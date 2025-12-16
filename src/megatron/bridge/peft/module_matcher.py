# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from megatron.core.tensor_parallel import ColumnParallelLinear, RowParallelLinear
from torch import nn

from megatron.bridge.peft.utils import wildcard_match
from megatron.bridge.utils.import_utils import safe_import_from


TEColumnParallelLinear, HAVE_TE_COL_LINEAR = safe_import_from(
    "megatron.core.extensions.transformer_engine", "TEColumnParallelLinear"
)
TELayerNormColumnParallelLinear, HAVE_TE_LN_COL_LINEAR = safe_import_from(
    "megatron.core.extensions.transformer_engine",
    "TELayerNormColumnParallelLinear",
)
TERowParallelLinear, HAVE_TE_ROW_LINEAR = safe_import_from(
    "megatron.core.extensions.transformer_engine", "TERowParallelLinear"
)
HAVE_TE = all((HAVE_TE_COL_LINEAR, HAVE_TE_LN_COL_LINEAR, HAVE_TE_ROW_LINEAR))


@dataclass
class ModuleMatcher:
    """
    Implements the LoRA (Low-Rank Adaptation) module for parameter-efficient fine-tuning.

    LoRA uses a low-rank projection to adapt the weights of a pre-trained model to a new downstream task.
    This class facilitates the application of LoRA to specific modules within the model architecture.

    Args:
        target_modules (List[str], optional): A list of module names to apply LoRA to.
            Defaults to all linear layers ['linear_qkv', 'linear_proj', 'linear_fc1', 'linear_fc2'].
                - 'linear_qkv': Apply LoRA to the fused linear layer used for query, key, and value projections
                                in self-attention.
                - 'linear_proj': Apply LoRA to the linear layer used for projecting the output of self-attention.
                - 'linear_fc1': Apply LoRA to the first fully-connected layer in MLP.
                - 'linear_fc2': Apply LoRA to the second fully-connected layer in MLP.
            Target modules can also contain wildcards. For example, you can specify
                target_modules=['*.layers.0.*.linear_qkv', '*.layers.1.*.linear_qkv'] to add LoRA to only linear_qkv
                on the first two layers.
    """

    target_modules: List[str] = field(
        default_factory=lambda: ["linear_qkv", "linear_proj", "linear_fc1", "linear_fc2"]
    )
    exclude_modules: List[str] = field(default_factory=list)
    canonical_mapping: Dict[str, Set] = field(default_factory=lambda: defaultdict(set))
    # pattern = canonical matcher string used internally (e.g., "linear_fc1")
    _pattern_to_alias: Dict[str, Set[str]] = field(default_factory=lambda: defaultdict(set), init=False, repr=False)
    # alias = user supplied target_modules entry that maps to a pattern
    _alias_to_pattern: Dict[str, str] = field(default_factory=dict, init=False, repr=False)
    _alias_matches: Dict[str, Set[str]] = field(default_factory=lambda: defaultdict(set), init=False, repr=False)

    def __post_init__(self) -> None:
        """Initialize target-module alias bookkeeping for validation."""
        for target in self.target_modules or []:
            self.register_target_alias(target, target)

    def match(
        self, m: nn.Module, name: Optional[str] = None, prefix: Optional[str] = None
    ) -> Optional[tuple[str, str]]:
        """
        Determines whether a given module matches specified target patterns.

        This function checks if the provided module `m` should be included based on predefined
        mapping rules (`canonical_mapping`, `target_modules`, and `exclude_modules`). It returns
        the matching pattern if a match is found; otherwise, it returns `None`.

        Args:
            m (nn.Module): The module being checked.
            name (str, optional): The module's name.
            prefix (str, optional): A prefix to be used in constructing `full_name`.

        Returns:
            Optional[Tuple[str, str]]: A tuple containing (matching_pattern, full_name) if a match
                is found; otherwise, `None`.

        Matching Logic:
        1) If `canonical_mapping` is defined, it checks:
        - Whether `name` exactly matches a pattern.
        - Whether `full_name` matches any regex pattern in `canonical_mapping`.
        2) If `target_modules` is defined, it follows the same logic as `canonical_mapping`.
        3) If neither `canonical_mapping` nor `target_modules` are defined, it ensures:
        - `name` is not in `exclude_modules`.
        - `full_name` does not match any `target_modules` patterns.
        - `m` is an instance of `nn.Linear`.

        Notes:
        - `exclude_modules` should only be non-empty if neither `canonical_mapping` nor `target_modules` are set.
        - The function asserts that `exclude_modules` is empty when using `canonical_mapping` or `target_modules`.
        """

        full_name = f"{prefix}.{name}" if prefix else name
        if len(self.canonical_mapping or []) > 0:
            """
            Find the element in canonical_mapping which
            1) matches the current `name` exactly, OR
            2) matches the current `full_name` with wildcard
            match is None if current module name doesn't match the specified targets.
            """
            assert len(self.exclude_modules) == 0, "exclude_modules should be empty when using canonical_mapping"
            for pattern in self.canonical_mapping:
                if name == pattern or wildcard_match(pattern, full_name):
                    self._record_match(pattern, full_name)
                    return (pattern, full_name)
        elif len(self.target_modules or []) > 0:
            assert len(self.exclude_modules) == 0, "exclude_modules should be empty when using target_modules"
            for pattern in self.target_modules:
                if name == pattern or wildcard_match(pattern, full_name):
                    self._record_match(pattern, full_name)
                    return (pattern, full_name)
        else:
            linear_types = [ColumnParallelLinear, RowParallelLinear, nn.Linear]
            if HAVE_TE_COL_LINEAR:
                linear_types.append(TEColumnParallelLinear)
            if HAVE_TE_LN_COL_LINEAR:
                linear_types.append(TELayerNormColumnParallelLinear)
            if HAVE_TE_ROW_LINEAR:
                linear_types.append(TERowParallelLinear)
            linear_types = tuple(linear_types)

            if (
                name not in self.exclude_modules
                and not any(wildcard_match(pattern, full_name) for pattern in self.exclude_modules)
                and isinstance(m, linear_types)
            ):
                self._record_match(name, full_name)
                return (name, full_name)

        return None

    def register_target_alias(self, alias: str, pattern: str) -> None:
        """Associate a user provided alias with the canonical pattern used for matching."""
        if alias is None or pattern is None:
            return

        previous_pattern = self._alias_to_pattern.get(alias)
        if previous_pattern:
            self._pattern_to_alias[previous_pattern].discard(alias)

        self._alias_to_pattern[alias] = pattern
        self._pattern_to_alias[pattern].add(alias)
        # Ensure alias has an entry in matches tracking so len(...) works without lookups later.
        self._alias_matches.setdefault(alias, set())

    def _record_match(self, pattern: str, full_name: Optional[str]) -> None:
        """Track which aliases successfully matched modules during traversal."""
        for alias in self._pattern_to_alias.get(pattern, []):
            self._alias_matches.setdefault(alias, set()).add(full_name or alias)

    def _reset_target_match_state(self) -> None:
        """Reset per-call match tracking."""
        for alias in self._alias_matches:
            self._alias_matches[alias].clear()

    def _validate_target_matches(self) -> None:
        """Raise an error if any requested target aliases failed to match a module."""
        if not self._alias_to_pattern:
            return

        unmatched = sorted(alias for alias, matches in self._alias_matches.items() if len(matches) == 0)
        if unmatched:
            raise ValueError("No modules matched the requested target_modules entries: " + ", ".join(unmatched))
