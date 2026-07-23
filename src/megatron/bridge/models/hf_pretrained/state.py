# Copyright (c) 2025, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import fnmatch
import hashlib
import json
import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Collection, Mapping
from functools import lru_cache
from pathlib import Path, PureWindowsPath
from typing import (
    Dict,
    Iterable,
    List,
    Optional,
    Pattern,
    Set,
    Tuple,
    Union,
    overload,
)

import torch


logger = logging.getLogger(__name__)

_DEFAULT_HF_SHARD_SIZE_BYTES = 5_000_000_000


def _validate_safetensors_shard_filename(filename: object, *, tensor_key: str, index_file: Path) -> str:
    """Validate a shard filename loaded from a safetensors index."""
    if not isinstance(filename, str) or not filename:
        raise ValueError(
            f"Invalid shard filename for tensor '{tensor_key}' in {index_file}: expected a non-empty string."
        )

    path = Path(filename)
    windows_path = PureWindowsPath(filename)
    if (
        "\x00" in filename
        or "\\" in filename
        or path.is_absolute()
        or windows_path.is_absolute()
        or windows_path.drive
        or ".." in path.parts
    ):
        raise ValueError(
            f"Invalid shard filename for tensor '{tensor_key}' in {index_file}: {filename!r} must be a relative "
            "path within the checkpoint directory."
        )

    if path.suffix != ".safetensors":
        raise ValueError(
            f"Invalid shard filename for tensor '{tensor_key}' in {index_file}: {filename!r} must end with "
            "'.safetensors'."
        )

    return filename


def _resolve_output_shard_path(output_path: Path, filename: str) -> Path:
    """Resolve a shard output path and ensure it remains inside output_path."""
    output_root = output_path.resolve()
    output_file_path = (output_root / filename).resolve()
    try:
        output_file_path.relative_to(output_root)
    except ValueError:
        raise ValueError(f"Shard filename {filename!r} escapes output directory {output_root}.") from None
    return output_file_path


def save_generator_to_safetensors(
    generator: Iterable[Tuple[str, torch.Tensor]],
    output_path: Union[str, Path],
    *,
    distributed_save: bool = False,
    save_every_n_ranks: int = 1,
    max_shard_size_bytes: int = _DEFAULT_HF_SHARD_SIZE_BYTES,
    expected_keys: Collection[str] | None = None,
) -> None:
    """Stream destination-defined Hugging Face safetensors shards.

    Unlike :meth:`SafeTensorsStateSource.save_generator`, this writer does not
    require a source checkpoint or reuse a source shard map. Every rank drains
    the same collective conversion generator and computes identical shard
    boundaries; saver ranks retain and write only the shards assigned to them.

    Args:
        generator: Iterable of destination tensor names and tensors.
        output_path: Directory in which to write the Hugging Face checkpoint.
        distributed_save: Whether to distribute shards across saver ranks.
        save_every_n_ranks: Use every Nth rank as a saver when distributed save
            is enabled.
        max_shard_size_bytes: Target maximum shard size. A single tensor larger
            than this limit is written to its own shard.
        expected_keys: Optional exact destination key set for strict validation.

    Raises:
        KeyError: If yielded names do not match ``expected_keys``.
        RuntimeError: If ranks observe different generator contents.
        TypeError: If the generator yields a non-tensor value.
        ValueError: If the generator is empty, contains duplicate names, or an
            argument is invalid.
    """
    if save_every_n_ranks < 1:
        raise ValueError(f"save_every_n_ranks must be at least 1, got {save_every_n_ranks}.")
    if max_shard_size_bytes < 1:
        raise ValueError(f"max_shard_size_bytes must be positive, got {max_shard_size_bytes}.")

    is_distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
    rank = torch.distributed.get_rank() if is_distributed else 0
    world_size = torch.distributed.get_world_size() if is_distributed else 1
    saver_ranks = list(range(0, world_size, save_every_n_ranks)) if distributed_save and is_distributed else [0]

    destination = Path(output_path)
    if rank == 0:
        destination.mkdir(parents=True, exist_ok=True)
    if is_distributed:
        torch.distributed.barrier()

    from safetensors.torch import save_file

    seen_names: set[str] = set()
    shard_by_name: dict[str, int] = {}
    current_names: list[str] = []
    current_tensors: dict[str, torch.Tensor] = {}
    current_size = 0
    shard_index = 1
    total_size = 0
    content_digest = hashlib.sha256()
    pending_owned_shards: dict[int, dict[str, torch.Tensor]] = {}

    def _owner(shard_number: int) -> int:
        return saver_ranks[(shard_number - 1) % len(saver_ranks)]

    def _write_pending_wave() -> None:
        for pending_index, pending_tensors in pending_owned_shards.items():
            temporary_name = f".model-{pending_index:05d}.safetensors.tmp"
            save_file(pending_tensors, destination / temporary_name)
        pending_owned_shards.clear()
        if is_distributed:
            torch.distributed.barrier()

    def _flush_current_shard() -> None:
        nonlocal current_names, current_size, current_tensors, shard_index
        if not current_names:
            return

        completed_shard = shard_index
        for tensor_name in current_names:
            shard_by_name[tensor_name] = completed_shard

        if rank == _owner(completed_shard):
            pending_owned_shards[completed_shard] = current_tensors

        current_names = []
        current_tensors = {}
        current_size = 0
        shard_index += 1
        if completed_shard % len(saver_ranks) == 0:
            # Each saver owns at most one shard in a wave. All owners write
            # concurrently before conversion advances to the next wave.
            _write_pending_wave()

    for name, tensor in generator:
        if not isinstance(name, str) or not name:
            raise ValueError(f"Safetensors names must be non-empty strings, got {name!r}.")
        if name in seen_names:
            raise ValueError(f"Generator yielded duplicate tensor name: {name}.")
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Generator value for {name} must be a torch.Tensor, got {type(tensor).__name__}.")

        tensor_size = tensor.numel() * tensor.element_size()
        if current_names and current_size + tensor_size > max_shard_size_bytes:
            _flush_current_shard()

        seen_names.add(name)
        current_names.append(name)
        current_size += tensor_size
        total_size += tensor_size
        content_digest.update(name.encode("utf-8"))
        content_digest.update(b"\0")
        content_digest.update(str(tensor_size).encode("ascii"))
        content_digest.update(b"\0")
        content_digest.update(str(tensor.dtype).encode("ascii"))
        content_digest.update(b"\0")
        content_digest.update(str(tuple(tensor.shape)).encode("ascii"))
        content_digest.update(b"\0")
        if rank == _owner(shard_index):
            current_tensors[name] = tensor.contiguous().cpu()

        if current_size >= max_shard_size_bytes:
            _flush_current_shard()

    _flush_current_shard()
    shard_count = shard_index - 1
    if shard_count % len(saver_ranks):
        _write_pending_wave()
    if not seen_names:
        raise ValueError("Cannot save an empty Hugging Face weight generator.")

    local_signature = (len(seen_names), total_size, shard_count, content_digest.hexdigest())
    if is_distributed:
        gathered_signatures: list[tuple[int, int, int, str] | None] = [None] * world_size
        torch.distributed.all_gather_object(gathered_signatures, local_signature)
        if any(signature != local_signature for signature in gathered_signatures):
            raise RuntimeError("Ranks observed different destination tensors while writing Hugging Face shards.")
        torch.distributed.barrier()

    if expected_keys is not None:
        expected_names = set(expected_keys)
        missing_names = sorted(expected_names - seen_names)
        unexpected_names = sorted(seen_names - expected_names)
        if missing_names or unexpected_names:
            for index in range(1, shard_count + 1):
                if rank == _owner(index):
                    (destination / f".model-{index:05d}.safetensors.tmp").unlink(missing_ok=True)
            if is_distributed:
                torch.distributed.barrier()
            raise KeyError(
                "Destination Hugging Face key mismatch: "
                f"missing={missing_names[:20]}, unexpected={unexpected_names[:20]}."
            )

    def _final_name(shard_number: int) -> str:
        if shard_count == 1:
            return "model.safetensors"
        return f"model-{shard_number:05d}-of-{shard_count:05d}.safetensors"

    for index in range(1, shard_count + 1):
        if rank != _owner(index):
            continue
        temporary_path = destination / f".model-{index:05d}.safetensors.tmp"
        temporary_path.replace(destination / _final_name(index))

    if is_distributed:
        torch.distributed.barrier()

    if rank == 0 and shard_count > 1:
        weight_map = {name: _final_name(index) for name, index in shard_by_name.items()}
        index_data = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
        with open(destination / "model.safetensors.index.json", "w") as index_file:
            json.dump(index_data, index_file, indent=2, sort_keys=True)

    if is_distributed:
        torch.distributed.barrier()


class StateDict(Mapping[str, torch.Tensor]):
    """
    A state dict accessor that provides a unified interface for querying model
    checkpoints.

    `StateDict` allows for efficient and flexible access to tensor data from
    various sources, such as in-memory dictionaries or directories of
    `.safetensors` files. A key feature is its ability to query and load only
    the required tensors without loading the entire checkpoint into memory,
    making it highly memory-efficient for large models.

    It supports a flexible, pandas-like querying interface that allows for
    accessing tensors by exact name, a list of names, glob patterns, or regular
    expressions. This makes it easy to inspect and manipulate model
    checkpoints.

    Examples:
        >>> # Setup an example StateDict from an in-memory dictionary
        >>> import torch
        >>> import re
        >>> d = {
        ...     "model.layer.0.weight": torch.randn(10, 10),
        ...     "model.layer.0.bias": torch.randn(10),
        ...     "model.layer.1.weight": torch.randn(10, 10),
        ...     "model.layer.1.bias": torch.randn(10),
        ... }
        >>> state = StateDict(d)
        >>>
        >>> # 1. Access a single tensor by exact key
        >>> state["model.layer.0.weight"].shape
        torch.Size([10, 10])
        >>>
        >>> # 2. Access multiple tensors with a list of strings
        >>> list(state[["model.layer.0.weight", "model.layer.1.weight"]].keys())
        ['model.layer.0.weight', 'model.layer.1.weight']
        >>>
        >>> # 3. Access with a glob pattern
        >>> sorted(list(state["model.layer.*.bias"].keys()))
        ['model.layer.0.bias', 'model.layer.1.bias']
        >>>
        >>> # 4. Access with a compiled regex pattern
        >>> regex = re.compile(r"model\\\\.layer\\\\.0\\\\..*")
        >>> sorted(list(state[regex].keys()))
        ['model.layer.0.bias', 'model.layer.0.weight']

    The same querying flexibility applies to checkpoints on disk. The following
    is a conceptual example of using `StateDict` with a `SafetensorsStateSource`
    to query a sharded checkpoint without loading all of it into memory.

    .. code-block:: python

        # Assume SafetensorsStateSource is available
        # from megatron.bridge.models.state import SafetensorsStateSource

        # Imagine a directory 'my_model_checkpoint/' with sharded weights.
        state_from_disk = StateDict(SafetensorsStateSource('my_model_checkpoint/'))

        # You can query it just like the in-memory dictionary. Only the required
        # tensors (e.g., all weight tensors) will be loaded from disk.
        weights = state_from_disk["model.layer.*.weight"]
    """

    source: "StateSource"

    def __init__(self, source: Dict[str, torch.Tensor] | "StateSource"):
        """
        Initializes the StateDict query accessor.

        Args:
            source: The source of the tensor data. This can be a standard
                Python dictionary mapping tensor names to `torch.Tensor` objects,
                or an instance of a `StateSource` subclass (e.g.,
                `SafetensorsStateSource`) for more advanced, out-of-memory
                access.
        """
        if isinstance(source, dict):
            source = DictStateSource(source)

        if not isinstance(source, StateSource):
            raise TypeError(f"StateDict source must be a dict or a StateSource, got {type(source)}")

        self.source = source

    def _get_all_keys(self) -> List[str]:
        """
        Get all available tensor keys from the underlying source.
        """
        return self.source.get_all_keys()

    def _load_tensors(self, keys_to_load: List[str]) -> Dict[str, torch.Tensor]:
        """
        Load specified tensors from the underlying source.
        """
        return self.source.load_tensors(keys_to_load)

    def _match_keys(self, pattern: Union[str, Pattern]) -> List[str]:
        """Match keys against a glob pattern or regex."""
        all_keys = self._get_all_keys()

        if isinstance(pattern, Pattern):
            # Regex pattern
            return [k for k in all_keys if pattern.search(k)]
        elif "*" in pattern or "?" in pattern or "[" in pattern:
            # Glob pattern
            return [k for k in all_keys if fnmatch.fnmatch(k, pattern)]
        else:
            # Exact match
            return [pattern] if pattern in all_keys else []

    @overload
    def __getitem__(self, key: str) -> Union[torch.Tensor, Dict[str, torch.Tensor]]: ...

    @overload
    def __getitem__(self, key: List[str]) -> Dict[str, torch.Tensor]: ...

    @overload
    def __getitem__(self, key: Pattern) -> Dict[str, torch.Tensor]: ...

    def __getitem__(self, key: Union[str, List[str], Pattern]) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Accesses state dict entries using various key types.

        This method allows for retrieving tensors using:
        - A single string for an exact key match.
        - A list of strings for multiple exact key matches.
        - A string with glob-style wildcards (`*`, `?`, `[]`).
        - A compiled regular expression object.

        Args:
            key: A single key string, a list of keys, a glob pattern string, or a
                compiled regular expression.

        Returns:
            - A single `torch.Tensor` if `key` is a string that matches exactly one key
              and does not contain wildcards.
            - A `Dict[str, torch.Tensor]` for all other cases (list of keys, glob
              pattern, or regex), mapping the matched keys to their corresponding
              tensors.

        Raises:
            KeyError: If the key (or any key in a list) is not found, or if a
                pattern matches no keys.

        Examples:
            >>> d = {
            ...     "model.embed_tokens.weight": torch.randn(10, 1),
            ...     "model.layers.0.mlp.weight": torch.randn(10, 1),
            ...     "model.layers.0.self_attn.q_proj.weight": torch.randn(10, 1),
            ...     "lm_head.weight": torch.randn(10, 1),
            ... }
            >>> state = StateDict(d)
            >>>
            >>> # Exact match (returns a single tensor)
            >>> tensor = state["model.embed_tokens.weight"]
            >>> isinstance(tensor, torch.Tensor)
            True
            >>>
            >>> # List of keys (returns a dict of tensors)
            >>> tensors = state[["model.embed_tokens.weight", "lm_head.weight"]]
            >>> sorted(tensors.keys())
            ['lm_head.weight', 'model.embed_tokens.weight']
            >>>
            >>> # Glob pattern (returns a dict of tensors)
            >>> layer_0_weights = state["model.layers.0.*.weight"]
            >>> sorted(layer_0_weights.keys())
            ['model.layers.0.mlp.weight', 'model.layers.0.self_attn.q_proj.weight']
            >>>
            >>> # Regex pattern (returns a dict of tensors)
            >>> import re
            >>> attn_weights = state[re.compile(r".*self_attn.*")]
            >>> list(attn_weights.keys())
            ['model.layers.0.self_attn.q_proj.weight']
        """
        if isinstance(key, Pattern):
            matched_keys = self._match_keys(key)
            if not matched_keys:
                raise KeyError(f"No keys match regex pattern: {key.pattern}")
            return self._load_tensors(matched_keys)
        elif isinstance(key, str):
            if "*" in key or "?" in key or "[" in key:
                matched_keys = self._match_keys(key)
                if not matched_keys:
                    raise KeyError(f"No keys match pattern: {key}")
                return self._load_tensors(matched_keys)
            else:
                if key not in self._get_all_keys():
                    raise KeyError(f"Key not found: {key}")
                return self._load_tensors([key])[key]
        elif isinstance(key, list):
            all_keys_set = set(self._get_all_keys())
            missing_keys = [k for k in key if k not in all_keys_set]
            if missing_keys:
                raise KeyError(f"Keys not found: {missing_keys}")
            return self._load_tensors(key)
        else:
            raise TypeError(f"Key must be str, list of str, or compiled regex, got {type(key)}")

    def __call__(self) -> Dict[str, torch.Tensor]:
        """
        Loads and returns the entire state dict as a dictionary.

        Note:
            This method loads all tensors from the source into memory. For large
            models, this can be memory-intensive. Prefer using pattern-based
            or single-key lookups for more efficient access if you only need a
            subset of the state dict.

        Returns:
            A dictionary containing all tensor names and their corresponding
            `torch.Tensor` objects.
        """
        all_keys = self._get_all_keys()
        return self._load_tensors(all_keys)

    def keys(self) -> List[str]:
        """Get all state dict keys."""
        return self._get_all_keys()

    def __contains__(self, key: str) -> bool:
        """Check if a key exists in the state dict."""
        return key in self._get_all_keys()

    def __repr__(self) -> str:
        """String representation."""
        try:
            num_params = len(self)
            return f"<StateDict with {num_params} entries>"
        except Exception:
            return "<StateDict (not accessible)>"

    def get(self, key: str, default=None) -> Optional[torch.Tensor]:
        """
        Gets a tensor from the state dict.
        Returns `default` if the key is not found.
        Note: This method is for single key lookup and does not support patterns.
        """
        if key in self._get_all_keys():
            return self._load_tensors([key])[key]
        return default

    def __iter__(self) -> Iterable[str]:
        """Iterate over state dict keys."""
        return iter(self.keys())

    def __len__(self) -> int:
        """Get number of entries in the state dict."""
        return len(self.keys())

    def has_glob(self, pattern: str) -> bool:
        """
        Efficiently checks if any tensor key matches the given glob pattern.
        This is forwarded to the underlying StateSource which may have an
        optimized implementation that avoids iterating over all keys.

        Args:
            pattern: The glob pattern to match against tensor keys.

        Returns:
            True if a matching key is found, False otherwise.
        """
        return self.source.has_glob(pattern)


class StateSource(ABC, Mapping[str, torch.Tensor]):
    """
    Abstract base class for a source of model state.

    This class defines a standard interface for `StateDict` to access tensor
    data, abstracting away the details of how and where the data is stored.
    Subclasses can implement loading from different storage backends, such as
    in-memory dictionaries or files on disk. This allows `StateDict` to handle
    various checkpoint formats in a uniform way.
    """

    @abstractmethod
    def get_all_keys(self) -> List[str]:
        """Returns a list of all available tensor keys in the source."""
        pass

    @abstractmethod
    def load_tensors(self, keys: List[str]) -> Dict[str, torch.Tensor]:
        """Loads the specified tensors from the source."""
        pass

    def __getitem__(self, key: str) -> torch.Tensor:
        """Loads a single tensor by key."""
        tensors = self.load_tensors([key])
        if key not in tensors:
            raise KeyError(f"Key not found in source: {key}")
        return tensors[key]

    def __iter__(self) -> Iterable[str]:
        """Iterates over all tensor keys."""
        return iter(self.get_all_keys())

    def __len__(self) -> int:
        """Returns the total number of tensors in the source."""
        return len(self.get_all_keys())

    def has_glob(self, pattern: str) -> bool:
        """
        Checks if any tensor key matches the given glob pattern.
        This default implementation is not efficient for all sources, as it may
        load all keys. Subclasses should override this method if a more
        performant implementation is available.
        """
        import fnmatch

        for key in self.get_all_keys():
            if fnmatch.fnmatch(key, pattern):
                return True
        return False


class DictStateSource(StateSource):
    """
    A state source backed by an in-memory Python dictionary.

    This is the simplest `StateSource` implementation. It's used when the entire
    model state dict is already loaded into a dictionary in memory.

    Args:
        state_dict: A dictionary mapping tensor names (str) to `torch.Tensor` objects.
    """

    def __init__(self, state_dict: Dict[str, torch.Tensor]):
        self._dict = state_dict
        self._keys_cache: Optional[List[str]] = None

    def get_all_keys(self) -> List[str]:
        if self._keys_cache is None:
            self._keys_cache = sorted(list(self._dict.keys()))
        return self._keys_cache

    def load_tensors(self, keys: List[str]) -> Dict[str, torch.Tensor]:
        return {key: self._dict[key] for key in keys if key in self._dict}


class SafeTensorsStateSource(StateSource):
    """
    A state source backed by a directory of .safetensors files.

    This source is designed for efficiently loading tensors from checkpoints saved
    in the Safetensors format, which is common for large models that are often
    "sharded" into multiple files.

    It can handle two common scenarios:
    1. A directory containing multiple `.safetensors` files.
    2. A directory containing a `model.safetensors.index.json` file, which maps
       tensor names to the specific `.safetensors` file they reside in. This is
       the standard format used by Hugging Face Transformers.

    Using this source allows `StateDict` to query for tensor keys and load only
    the necessary files and tensors from disk, avoiding high memory usage.

    Args:
        path: The path to the directory containing the `.safetensors` files
              and/or the index file. Can also be a Hugging Face Hub model ID.
    """

    def __init__(self, path: Union[str, Path]):
        self.model_name_or_path = path
        self._resolved_path_cache: Optional[Path] = None
        self._keys_cache: Optional[List[str]] = None
        self._key_to_filename_map_cache: Optional[Dict[str, str]] = None

    @staticmethod
    def _ignore_source_key_prefixes(
        key_to_filename_map: Mapping[str, str] | None,
        ignored_source_key_prefixes: Iterable[str] | None,
    ) -> Dict[str, str]:
        if not key_to_filename_map:
            return {}
        if not ignored_source_key_prefixes:
            return dict(key_to_filename_map)

        prefixes = tuple(ignored_source_key_prefixes)
        return {key: filename for key, filename in key_to_filename_map.items() if not key.startswith(prefixes)}

    @property
    def path(self) -> Path:
        """
        The local path to the checkpoint files.
        If the initial path is a Hugging Face Hub model ID, this property
        will handle downloading the necessary files and return the local
        cache path.
        """
        if self._resolved_path_cache is None:
            self._resolved_path_cache = self._resolve_path(self.model_name_or_path)
        return self._resolved_path_cache

    @property
    def key_to_filename_map(self) -> Dict[str, str]:
        """
        Provides a mapping from tensor keys to the safetensor filename they
        are stored in.

        This map is constructed either from `model.safetensors.index.json` if
        it exists, or by scanning all `.safetensors` files in the directory.
        The result is cached for efficiency.
        """
        if self._key_to_filename_map_cache is not None:
            return self._key_to_filename_map_cache

        # First, try to load from the index file.
        key_map = self._cached_get_key_to_filename_map(self.path)
        if key_map:
            self._key_to_filename_map_cache = key_map
            return key_map

        # If no index, scan the directory.
        import os
        from glob import glob as file_glob

        from safetensors import safe_open

        key_map = {}
        safetensor_files = file_glob(str(self.path / "*.safetensors"))
        for file_path in safetensor_files:
            filename = os.path.basename(file_path)
            try:
                with safe_open(file_path, framework="pt", device="cpu") as f:
                    for key in f.keys():
                        if key in key_map:
                            # This is an issue. Same key in multiple files, and no index.
                            # How to resolve ambiguity? Let's just warn and overwrite. Last one wins.
                            print(
                                f"Warning: duplicate key '{key}' found in '{filename}' and '{key_map[key]}'. Using '{filename}'."
                            )
                        key_map[key] = filename
            except Exception as e:
                # Can be not a safetensor file, etc.
                print(f"Warning: could not open {filename} as a safetensors file: {e}")

        self._key_to_filename_map_cache = key_map
        return key_map

    @staticmethod
    def _resolve_path(model_name_or_path: Union[str, Path]) -> Path:
        """
        Resolves a model name or path to a local directory.
        If the path is not a local directory, it is treated as a Hugging
        Face Hub model ID, and the corresponding files are downloaded.
        """
        local_path = Path(model_name_or_path)
        if local_path.is_dir():
            return local_path

        try:
            from huggingface_hub import snapshot_download
            from huggingface_hub.utils import HfHubHTTPError

            # Not a local directory, so we assume it's a model ID
            # on the Hugging Face Hub.
            return Path(
                snapshot_download(
                    repo_id=str(model_name_or_path),
                    allow_patterns=[
                        "*.safetensors",
                        "model.safetensors.index.json",
                    ],
                    # Ignore other large files.
                    ignore_patterns=["*.bin", "*.pt", "*.pth"],
                )
            )
        except (ImportError, HfHubHTTPError, ValueError) as e:
            logger.warning(
                f"Failed to download '{model_name_or_path}' from HuggingFace Hub: {e}. "
                f"Falling back to treating it as a local path."
            )
            return local_path

    def get_all_keys(self) -> List[str]:
        if self._keys_cache is not None:
            return self._keys_cache

        from glob import glob as file_glob

        from safetensors import safe_open

        all_keys = set()
        key_to_filename_map = self.key_to_filename_map
        if key_to_filename_map:
            all_keys.update(key_to_filename_map.keys())

        if not all_keys:
            safetensor_files = file_glob(str(self.path / "*.safetensors"))
            if not safetensor_files and not key_to_filename_map:
                raise FileNotFoundError(f"No .safetensors files or index found in {self.model_name_or_path}")
            for safetensor_file in safetensor_files:
                with safe_open(safetensor_file, framework="pt", device="cpu") as f:
                    all_keys.update(f.keys())

        self._keys_cache = sorted(list(all_keys))
        return self._keys_cache

    def load_tensors(self, keys_to_load: List[str]) -> Dict[str, torch.Tensor]:
        if not keys_to_load:
            return {}

        import time
        from glob import glob as file_glob

        from safetensors import safe_open

        loaded_tensors = {}
        remaining_keys = set(keys_to_load)
        key_to_filename_map = self.key_to_filename_map

        max_retries = 3

        if key_to_filename_map:
            file_to_keys_map = defaultdict(list)
            for key in list(remaining_keys):
                if key in key_to_filename_map:
                    filename = key_to_filename_map[key]
                    file_to_keys_map[filename].append(key)

            for filename, keys_in_file in file_to_keys_map.items():
                file_path = self.path / filename
                if file_path.exists():
                    for attempt in range(max_retries):
                        with safe_open(file_path, framework="pt", device="cpu") as f:
                            file_keys = set(f.keys())
                            for key in keys_in_file:
                                if key in file_keys and key not in loaded_tensors:
                                    loaded_tensors[key] = f.get_tensor(key)
                                    remaining_keys.discard(key)
                        still_missing = [k for k in keys_in_file if k in remaining_keys]
                        if not still_missing:
                            break
                        if attempt < max_retries - 1:
                            logger.warning(
                                f"Retry {attempt + 1}/{max_retries}: {len(still_missing)} keys "
                                f"not found in {filename}, retrying after delay..."
                            )
                            time.sleep(1.0 * (attempt + 1))

        if remaining_keys:
            safetensor_files = file_glob(str(self.path / "*.safetensors"))
            if not safetensor_files and not key_to_filename_map and not loaded_tensors:
                raise FileNotFoundError(
                    f"No .safetensors files found in {self.model_name_or_path} to load keys: {remaining_keys}"
                )
            for safetensor_file_path in safetensor_files:
                if not remaining_keys:
                    break
                with safe_open(safetensor_file_path, framework="pt", device="cpu") as f:
                    current_file_keys = f.keys()
                    for key in list(remaining_keys):
                        if key in current_file_keys:
                            loaded_tensors[key] = f.get_tensor(key)
                            remaining_keys.remove(key)

        if remaining_keys:
            raise KeyError(f"Keys not found in safetensors from {self.model_name_or_path}: {remaining_keys}")

        return loaded_tensors

    def has_glob(self, pattern: str) -> bool:
        """
        Efficiently checks if any tensor key matches the given glob pattern.

        This method avoids loading all tensor keys into memory at once. It scans
        the checkpoint index or file headers and returns as soon as a match is
        found.

        Args:
            pattern: The glob pattern to match against tensor keys.

        Returns:
            True if a matching key is found, False otherwise.
        """
        import fnmatch
        from glob import glob as file_glob

        from safetensors import safe_open

        key_to_filename_map = self.key_to_filename_map
        if key_to_filename_map:
            for key in key_to_filename_map.keys():
                if fnmatch.fnmatch(key, pattern):
                    return True
            return False

        # If no index map, scan the files directly.
        safetensor_files = file_glob(str(self.path / "*.safetensors"))
        if not safetensor_files:
            return False

        for safetensor_file in safetensor_files:
            try:
                with safe_open(safetensor_file, framework="pt", device="cpu") as f:
                    for key in f.keys():
                        if fnmatch.fnmatch(key, pattern):
                            return True
            except Exception:
                # Ignore files that are not valid safetensors
                continue

        return False

    def save_generator(
        self,
        generator: Iterable[Tuple[str, torch.Tensor]],
        output_path: Union[str, Path],
        strict: bool = True,
        distributed_save: bool = False,
        save_every_n_ranks: int = 1,
        ignored_source_key_prefixes: Iterable[str] | None = None,
    ):
        """
        Saves tensors from a generator to `.safetensors` files, preserving the
        original sharding structure in a memory-efficient, streaming fashion.

        This method reads the sharding information (which tensor belongs to which
        file) from the source checkpoint. It then consumes a generator of tensors,
        buffering them in memory only until a complete file shard can be written to
        disk. This approach minimizes peak memory usage compared to collecting all
        tensors first.

        If the original checkpoint had a `model.safetensors.index.json` file, a new
        one will be created for the saved tensors.

        Args:
            generator: An iterable of (tensor_name, tensor) tuples.
            output_path: The directory where the new safetensor files and index
                         will be saved.
            strict: If True (default), raises a KeyError if the generator
                    yields a tensor name not found in the original model's
                    sharding structure. If False, it prints a warning and
                    skips the tensor.
            distributed_save: Whether to enable distributed saving mode where each rank saves
                part of weights independently.
            save_every_n_ranks: Interval for saving weights across ranks in distributed mode.
                For example, if set to 2, only ranks 0, 2, 4, ... will save weights.
            ignored_source_key_prefixes: Source tensor key prefixes to omit from the expected
                source sharding map when saving.

        """
        if distributed_save:
            return self._save_generator_distributed(
                generator,
                output_path,
                strict,
                save_every_n_ranks=save_every_n_ranks,
                ignored_source_key_prefixes=ignored_source_key_prefixes,
            )

        # In a distributed environment, only rank 0 should write to disk.
        # Other ranks must still exhaust the generator to participate in collectives.
        is_distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
        rank = torch.distributed.get_rank() if is_distributed else 0

        if rank != 0:
            # Other ranks must exhaust the generator to avoid hangs in collectives.
            for _ in generator:
                pass
            return

        # Rank 0 proceeds with saving.
        from safetensors.torch import save_file

        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        key_to_filename_map = self._ignore_source_key_prefixes(self.key_to_filename_map, ignored_source_key_prefixes)
        all_expected_keys = set(key_to_filename_map.keys())

        if not key_to_filename_map:
            buffered_tensors = dict(generator)
            if buffered_tensors:
                save_file(buffered_tensors, output_path / "model.safetensors")
            return

        filename_to_keys_map = defaultdict(set)
        for key, filename in key_to_filename_map.items():
            filename_to_keys_map[filename].add(key)

        files_to_save = dict(filename_to_keys_map)
        remaining_keys_by_file = {filename: set(keys) for filename, keys in files_to_save.items()}
        buffered_tensors = {}
        all_yielded_keys = set()
        all_saved_keys = set()
        total_saved_tensor_bytes = 0

        for name, tensor in generator:
            all_yielded_keys.add(name)
            if name not in all_expected_keys:
                if strict:
                    raise KeyError(
                        f"Tensor '{name}' from generator not found in the original model structure. "
                        "Re-run with strict=False to save the partial checkpoint instead of failing."
                    )
                else:
                    print(f"Warning: tensor '{name}' from generator not found in original model structure. Skipping.")
                    continue

            buffered_tensors[name] = tensor

            # Update only the shard containing this tensor. Scanning every shard
            # for every yielded tensor becomes prohibitively expensive for models
            # with tens of thousands of tensors.
            filename = key_to_filename_map[name]
            remaining_keys = remaining_keys_by_file.get(filename)
            if remaining_keys is None:
                # The shard was already saved. Preserve the previous duplicate-key
                # behavior by leaving the tensor buffered for final reporting.
                continue

            remaining_keys.discard(name)
            if not remaining_keys:
                keys_for_file = files_to_save[filename]
                tensors_to_save = {key: buffered_tensors[key] for key in keys_for_file}

                output_file_path = _resolve_output_shard_path(output_path, filename)
                output_file_path.parent.mkdir(parents=True, exist_ok=True)
                save_file(tensors_to_save, output_file_path)
                total_saved_tensor_bytes += sum(
                    tensor.numel() * tensor.element_size() for tensor in tensors_to_save.values()
                )

                # Free memory by removing saved tensors from the buffer.
                for key in keys_for_file:
                    del buffered_tensors[key]

                all_saved_keys.update(keys_for_file)
                del files_to_save[filename]
                del remaining_keys_by_file[filename]

        # --- Final Reporting ---
        if files_to_save:
            if strict:
                print(
                    "Warning: The following files could not be saved because the generator did not yield all of their tensors:"
                )
            else:
                print(
                    "Warning: The following files are different from the source because the generator did not yield all "
                    "of their tensors. However they are still saved because strict=False."
                )
            for filename in files_to_save:
                missing_for_file = remaining_keys_by_file[filename]
                if missing_for_file:
                    print(f"  - {filename}: missing {len(missing_for_file)} tensors:")
                    for key in sorted(list(missing_for_file)):
                        print(f"    - {key}")
            if not strict:
                for filename in list(files_to_save.keys()):
                    keys_for_file = files_to_save[filename]
                    tensors_to_save = {key: buffered_tensors[key] for key in keys_for_file if key in buffered_tensors}
                    output_file_path = _resolve_output_shard_path(output_path, filename)
                    output_file_path.parent.mkdir(parents=True, exist_ok=True)
                    save_file(tensors_to_save, output_file_path)
                    total_saved_tensor_bytes += sum(
                        tensor.numel() * tensor.element_size() for tensor in tensors_to_save.values()
                    )

                    # Free memory by removing saved tensors from the buffer.
                    for key in tensors_to_save.keys():
                        del buffered_tensors[key]

                    # Only the keys we actually wrote belong in the
                    # post-save index. Using ``keys_for_file`` here would
                    # claim unsaved keys as written and produce a
                    # ``model.safetensors.index.json`` that maps keys to a
                    # file which doesn't contain them — HF loading then
                    # crashes when it tries to read the missing tensor.
                    all_saved_keys.update(tensors_to_save.keys())
                    del files_to_save[filename]

        if buffered_tensors:
            print(
                f"Warning: {len(buffered_tensors)} tensors were yielded but not saved because their corresponding file shards were incomplete."
            )

        # Final check on whether all original tensors were written.
        unsaved_keys = all_expected_keys - all_saved_keys
        if unsaved_keys and strict:
            print(f"\nError: {len(unsaved_keys)} tensors from the original checkpoint were not written:")
            for key in sorted(unsaved_keys):
                print(f"  - {key}")
            raise RuntimeError(
                f"{len(unsaved_keys)} tensors from the original checkpoint were not written. "
                "Re-run with strict=False to save the partial checkpoint instead of failing."
            )
        if not unsaved_keys:
            extra_keys = all_yielded_keys - all_expected_keys
            if extra_keys:
                print(
                    f"\nSuccess: All tensors from the original checkpoint were written. "
                    f"({len(extra_keys)} extra tensors from generator were ignored as per strict=False)."
                )
            else:
                print("\nSuccess: All tensors from the original checkpoint were written.")
        else:
            print(
                f"\nError: {len(unsaved_keys)} tensors from the original checkpoint were not written. See warnings above for details."
            )

        # Create index file for the saved shards.
        original_index_file = self.path / "model.safetensors.index.json"
        if original_index_file.exists():
            with open(original_index_file, "r") as f:
                original_index_data = json.load(f)

            new_weight_map = {key: key_to_filename_map[key] for key in all_saved_keys}

            metadata = dict(original_index_data.get("metadata", {}))
            metadata["total_size"] = total_saved_tensor_bytes
            new_index_data = {"metadata": metadata, "weight_map": new_weight_map}

            output_index_file = output_path / "model.safetensors.index.json"
            if new_weight_map:
                with open(output_index_file, "w") as f:
                    json.dump(new_index_data, f, indent=4)

    @staticmethod
    @lru_cache(maxsize=None)
    def _cached_get_key_to_filename_map(model_name_or_path: Union[str, Path]) -> Optional[Dict[str, str]]:
        """Static, cached method to get the key-to-filename map."""
        index_file = Path(model_name_or_path) / "model.safetensors.index.json"
        if index_file.exists():
            with open(index_file, "r") as f:
                try:
                    index_data = json.load(f)
                    if "weight_map" in index_data and isinstance(index_data["weight_map"], dict):
                        return {
                            key: _validate_safetensors_shard_filename(
                                filename,
                                tensor_key=key,
                                index_file=index_file,
                            )
                            for key, filename in index_data["weight_map"].items()
                        }
                except json.JSONDecodeError:
                    return None
        return None

    def _save_generator_distributed(
        self,
        generator: Iterable[Tuple[str, torch.Tensor]],
        output_path: Union[str, Path],
        strict: bool = True,
        save_every_n_ranks: int = 1,
        ignored_source_key_prefixes: Iterable[str] | None = None,
    ):
        is_distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
        if is_distributed:
            world_size = torch.distributed.get_world_size()
            rank = torch.distributed.get_rank()
        else:
            world_size = 1
            rank = 0

        from safetensors.torch import save_file

        output_path = Path(output_path)

        # Calculate which ranks should participate in saving
        # Only rank % save_every_n_ranks == 0 will save
        num_nodes = (world_size + save_every_n_ranks - 1) // save_every_n_ranks
        is_saver_rank = rank % save_every_n_ranks == 0
        saver_ranks = [i * save_every_n_ranks for i in range(num_nodes) if i * save_every_n_ranks < world_size]
        num_savers = len(saver_ranks)
        saver_index = rank // save_every_n_ranks if is_saver_rank else -1

        if rank == 0:
            output_path.mkdir(parents=True, exist_ok=True)
        if is_distributed:
            torch.distributed.barrier()

        key_to_filename_map = self._ignore_source_key_prefixes(self.key_to_filename_map, ignored_source_key_prefixes)

        # Fallback: no sharding map, single-file save
        if not key_to_filename_map:
            if is_saver_rank and saver_index == 0:
                buffered_tensors = dict(generator)
                if buffered_tensors:
                    save_file(buffered_tensors, output_path / "model.safetensors")
            else:
                for _ in generator:
                    pass
            if is_distributed:
                torch.distributed.barrier()
            return

        all_expected_keys: Set[str] = set(key_to_filename_map.keys())
        all_yielded_keys = set()
        filename_to_keys_map: Dict[str, Set[str]] = defaultdict(set)
        for key, fname in key_to_filename_map.items():
            filename_to_keys_map[fname].add(key)

        all_filenames = sorted(filename_to_keys_map.keys())

        # Distribute files among saver ranks (one per node)
        if is_saver_rank:
            assigned_filenames = [fname for idx, fname in enumerate(all_filenames) if idx % num_savers == saver_index]
            assigned_filenames_set = set(assigned_filenames)
            assigned_expected_keys: Set[str] = (
                set().union(*(filename_to_keys_map[fname] for fname in assigned_filenames))
                if assigned_filenames
                else set()
            )
        else:
            assigned_filenames = []
            assigned_filenames_set = set()
            assigned_expected_keys = set()

        buffered_tensors: Dict[str, torch.Tensor] = {}
        actually_saved_keys: Set[str] = set()
        local_saved_tensor_bytes = 0

        for name, tensor in generator:
            all_yielded_keys.add(name)

            if name not in all_expected_keys:
                if strict:
                    raise KeyError(
                        f"Tensor '{name}' from generator not found in the original model structure. "
                        "Re-run with strict=False to save the partial checkpoint instead of failing."
                    )
                else:
                    print(f"Warning: tensor '{name}' from generator not found in original model structure. Skipping.")
                    continue

            if is_saver_rank:
                fname = key_to_filename_map[name]
                if fname not in assigned_filenames_set:
                    continue
                buffered_tensors[name] = tensor

        if is_saver_rank:
            missing_keys = assigned_expected_keys - set(buffered_tensors.keys())
            if missing_keys:
                missing_keys_sorted = sorted(missing_keys)
                missing_preview = ", ".join(missing_keys_sorted[:20])
                missing_suffix = ""
                if len(missing_keys_sorted) > 20:
                    missing_suffix = f", ... (+{len(missing_keys_sorted) - 20} more)"
                print(
                    f"Rank {rank}: Missing {len(missing_keys_sorted)} tensors for keys: "
                    f"{missing_preview}{missing_suffix}",
                    flush=True,
                )

            for fname in assigned_filenames:
                keys_for_file = filename_to_keys_map[fname]
                tensors_to_save = {k: buffered_tensors[k] for k in keys_for_file if k in buffered_tensors}
                if not tensors_to_save:
                    continue
                output_file_path = _resolve_output_shard_path(output_path, fname)
                output_file_path.parent.mkdir(parents=True, exist_ok=True)
                save_file(tensors_to_save, output_file_path)
                actually_saved_keys.update(tensors_to_save.keys())
                local_saved_tensor_bytes += sum(
                    tensor.numel() * tensor.element_size() for tensor in tensors_to_save.values()
                )

        # Strict-mode check: ensure all expected tensors were written. Aggregate
        # per-rank missing counts so all ranks raise consistently (avoids hangs
        # on the trailing barriers).
        if is_saver_rank:
            local_unsaved_keys = assigned_expected_keys - actually_saved_keys
        else:
            local_unsaved_keys = set()

        if is_distributed:
            gathered_save_status: list[tuple[int, int] | None] = [None] * world_size
            torch.distributed.all_gather_object(
                gathered_save_status,
                (len(local_unsaved_keys), local_saved_tensor_bytes),
            )
            total_unsaved_count = sum(status[0] for status in gathered_save_status if status is not None)
            total_saved_tensor_bytes = sum(status[1] for status in gathered_save_status if status is not None)
        else:
            total_unsaved_count = len(local_unsaved_keys)
            total_saved_tensor_bytes = local_saved_tensor_bytes

        if total_unsaved_count and strict:
            if local_unsaved_keys:
                keys_sorted = sorted(local_unsaved_keys)
                preview = ", ".join(keys_sorted[:20])
                suffix = f", ... (+{len(keys_sorted) - 20} more)" if len(keys_sorted) > 20 else ""
                print(
                    f"Rank {rank}: Error: {len(keys_sorted)} tensors not written: {preview}{suffix}",
                    flush=True,
                )
            raise RuntimeError(
                f"{total_unsaved_count} tensors from the original checkpoint were not written. "
                "Re-run with strict=False to save the partial checkpoint instead of failing."
            )

        # Rank 0 builds the index from the files that were actually written.
        # This avoids all_gather_object on very large key lists, which can
        # allocate excessive CUDA memory in distributed runs.
        if is_distributed:
            torch.distributed.barrier()

            if rank == 0:
                from safetensors import safe_open

                all_saved_keys_aggregated = set()
                for fname in all_filenames:
                    file_path = _resolve_output_shard_path(output_path, fname)
                    if not file_path.exists():
                        continue
                    with safe_open(file_path, framework="pt", device="cpu") as f:
                        all_saved_keys_aggregated.update(f.keys())
            else:
                all_saved_keys_aggregated = set()
        else:
            all_saved_keys_aggregated = actually_saved_keys

        if rank == 0:
            original_index_file = self.path / "model.safetensors.index.json"
            if original_index_file.exists():
                with open(original_index_file, "r") as f:
                    original_index_data = json.load(f)

                # Build weight_map only from actually saved keys, like the non-distributed path
                new_weight_map = {
                    key: key_to_filename_map[key] for key in key_to_filename_map if key in all_saved_keys_aggregated
                }

                metadata = dict(original_index_data.get("metadata", {}))
                metadata["total_size"] = total_saved_tensor_bytes
                new_index_data = {"metadata": metadata, "weight_map": new_weight_map}
                output_index_file = output_path / "model.safetensors.index.json"
                with open(output_index_file, "w") as f:
                    json.dump(new_index_data, f, indent=4)

        if is_distributed:
            torch.distributed.barrier()
