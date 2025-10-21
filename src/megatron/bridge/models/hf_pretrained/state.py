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
import json
import re
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from typing import (
    Iterable,
    Pattern,
    overload,
)

import torch


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
        >>> sorted(list(state.glob("model.layer.*.bias").keys()))
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
        weights = state_from_disk.glob("model.layer.*.weight")
    """

    source: "StateSource"

    def __init__(self, source: dict[str, torch.Tensor] | "StateSource"):
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

    def _get_all_keys(self) -> list[str]:
        """
        Get all available tensor keys from the underlying source.
        """
        return self.source.get_all_keys()

    def _load_tensors(self, keys_to_load: list[str]) -> dict[str, torch.Tensor]:
        """
        Load specified tensors from the underlying source.
        """
        return self.source.load_tensors(keys_to_load)

    def _match_keys(self, pattern: str | Pattern) -> list[str]:
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
    def __getitem__(self, key: str) -> torch.Tensor | dict[str, torch.Tensor]: ...

    @overload
    def __getitem__(self, key: list[str]) -> dict[str, torch.Tensor]: ...

    @overload
    def __getitem__(self, key: Pattern) -> dict[str, torch.Tensor]: ...

    def __getitem__(self, key: str | list[str], Pattern) -> torch.Tensor | dict[str, torch.Tensor]:
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
            - A `dict[str, torch.Tensor]` for all other cases (list of keys, glob
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

    def regex(self, pattern: str) -> dict[str, torch.Tensor]:
        """
        Queries the state dict with a regular expression pattern.

        This is a convenience method that compiles the pattern string and uses it
        to retrieve all matching tensors.

        Args:
            pattern: The regular expression string to match against tensor keys.

        Returns:
            A dictionary mapping matching tensor names to their `torch.Tensor` objects.

        Examples:
            >>> d = {
            ...     "model.layers.0.self_attn.weight": torch.randn(1, 1),
            ...     "model.layers.1.self_attn.weight": torch.randn(1, 1),
            ...     "model.layers.1.mlp.weight": torch.randn(1, 1)
            ... }
            >>> state = StateDict(d)
            >>> # Get all attention-related weights
            >>> attention_weights = state.regex(r"model\\.layers\\.\\d+\\.self_attn.*")
            >>> sorted(attention_weights.keys())
            ['model.layers.0.self_attn.weight', 'model.layers.1.self_attn.weight']
        """
        return self[re.compile(pattern)]

    def glob(self, pattern: str) -> dict[str, torch.Tensor]:
        """
        Queries the state dict with a glob pattern.

        This is a convenience method for pattern matching using Unix shell-style
        wildcards.

        Args:
            pattern: The glob pattern string to match against tensor keys.

        Returns:
            A dictionary mapping matching tensor names to their `torch.Tensor` objects.

        Examples:
            >>> d = {
            ...     "model.layers.0.mlp.weight": torch.randn(1, 1),
            ...     "model.layers.0.mlp.bias": torch.randn(1, 1),
            ...     "model.layers.1.mlp.weight": torch.randn(1, 1)
            ... }
            >>> state = StateDict(d)
            >>> # Get all mlp weights and biases from the first layer
            >>> layer_0_mlp = state.glob("model.layers.0.mlp.*")
            >>> sorted(layer_0_mlp.keys())
            ['model.layers.0.mlp.bias', 'model.layers.0.mlp.weight']
        """
        return self[pattern]

    def __call__(self) -> dict[str, torch.Tensor]:
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

    def keys(self) -> list[str]:
        """Get all state dict keys."""
        return self._get_all_keys()

    def items(self) -> list[tuple]:
        """Get all state dict items."""
        return list(self().items())

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

    def get(self, key: str, default=None) -> torch.Tensor | None:
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
    def get_all_keys(self) -> list[str]:
        """Returns a list of all available tensor keys in the source."""
        pass

    @abstractmethod
    def load_tensors(self, keys: list[str]) -> dict[str, torch.Tensor]:
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

    def __init__(self, state_dict: dict[str, torch.Tensor]):
        self._dict = state_dict
        self._keys_cache: list[str] | None = None

    def get_all_keys(self) -> list[str]:
        if self._keys_cache is None:
            self._keys_cache = sorted(list(self._dict.keys()))
        return self._keys_cache

    def load_tensors(self, keys: list[str]) -> dict[str, torch.Tensor]:
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

    def __init__(self, path: str | Path):
        self.model_name_or_path = path
        self._resolved_path_cache: Path | None = None
        self._keys_cache: list[str] | None = None
        self._key_to_filename_map_cache: dict[str, str] | None = None

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
    def key_to_filename_map(self) -> dict[str, str]:
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
    def _resolve_path(model_name_or_path: str | Path) -> Path:
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
        except (ImportError, HfHubHTTPError, ValueError):
            # If huggingface_hub is not installed, or if it's not a
            # valid model ID, we return the original path and let the
            # subsequent logic handle the file not found error.
            return local_path

    def get_all_keys(self) -> list[str]:
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

    def load_tensors(self, keys_to_load: list[str]) -> dict[str, torch.Tensor]:
        if not keys_to_load:
            return {}

        from glob import glob as file_glob

        from safetensors import safe_open

        loaded_tensors = {}
        remaining_keys = set(keys_to_load)
        key_to_filename_map = self.key_to_filename_map

        if key_to_filename_map:
            file_to_keys_map = defaultdict(list)
            for key in list(remaining_keys):
                if key in key_to_filename_map:
                    filename = key_to_filename_map[key]
                    file_to_keys_map[filename].append(key)

            for filename, keys_in_file in file_to_keys_map.items():
                file_path = self.path / filename
                if file_path.exists():
                    with safe_open(file_path, framework="pt", device="cpu") as f:
                        for key in keys_in_file:
                            if key in f.keys():
                                loaded_tensors[key] = f.get_tensor(key)
                                remaining_keys.discard(key)

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
        self, generator: Iterable[tuple[str, torch.Tensor]], output_path: str | Path, strict: bool = True
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
        """
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

        key_to_filename_map = self.key_to_filename_map
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
        buffered_tensors = {}
        all_yielded_keys = set()
        all_saved_keys = set()

        for name, tensor in generator:
            all_yielded_keys.add(name)
            if name not in all_expected_keys:
                if strict:
                    raise KeyError(
                        f"Tensor '{name}' from generator not found in the original model structure. "
                        "To ignore, set strict=False."
                    )
                else:
                    print(f"Warning: tensor '{name}' from generator not found in original model structure. Skipping.")
                    continue

            buffered_tensors[name] = tensor

            # Check if any file is complete and can be saved.
            # Iterate over a copy of keys since we might modify the dict.
            for filename in list(files_to_save.keys()):
                keys_for_file = files_to_save[filename]
                if keys_for_file.issubset(buffered_tensors.keys()):
                    # This shard is complete, save it.
                    tensors_to_save = {key: buffered_tensors[key] for key in keys_for_file}

                    output_file_path = output_path / filename
                    save_file(tensors_to_save, output_file_path)

                    # Free memory by removing saved tensors from the buffer.
                    for key in keys_for_file:
                        del buffered_tensors[key]

                    all_saved_keys.update(keys_for_file)
                    del files_to_save[filename]

        # --- Final Reporting ---
        if files_to_save:
            print(
                "Warning: The following files could not be saved because the generator did not yield all of their tensors:"
            )
            for filename, keys_for_file in files_to_save.items():
                missing_for_file = keys_for_file - all_yielded_keys
                if missing_for_file:
                    print(f"  - {filename}: missing {len(missing_for_file)} tensors:")
                    for key in sorted(list(missing_for_file)):
                        print(f"    - {key}")

        if buffered_tensors:
            print(
                f"Warning: {len(buffered_tensors)} tensors were yielded but not saved because their corresponding file shards were incomplete."
            )

        # Final check on whether all original tensors were written.
        unsaved_keys = all_expected_keys - all_saved_keys
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

            new_index_data = {
                "metadata": original_index_data.get("metadata", {}),
                "weight_map": new_weight_map,
            }

            output_index_file = output_path / "model.safetensors.index.json"
            if new_weight_map:
                with open(output_index_file, "w") as f:
                    json.dump(new_index_data, f, indent=4)

    def _get_key_to_filename_map(self) -> dict[str, str] | None:
        return self._cached_get_key_to_filename_map(self.path)

    @staticmethod
    @lru_cache(maxsize=None)
    def _cached_get_key_to_filename_map(model_name_or_path: str | Path) -> dict[str, str] | None:
        """Static, cached method to get the key-to-filename map."""
        index_file = Path(model_name_or_path) / "model.safetensors.index.json"
        if index_file.exists():
            with open(index_file, "r") as f:
                try:
                    index_data = json.load(f)
                    if "weight_map" in index_data and isinstance(index_data["weight_map"], dict):
                        return index_data["weight_map"]
                except json.JSONDecodeError:
                    return None
        return None
