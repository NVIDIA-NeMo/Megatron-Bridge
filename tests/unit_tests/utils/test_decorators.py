# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Unit tests for utils/decorators.

Covers the ``experimental_fn`` decorator:

- emits a DeprecationWarning-style warning naming the wrapped function
- forwards positional / keyword args and the return value
- preserves the wrapped function's __name__ and __doc__ (functools.wraps)
- only warns from rank 0 in a distributed setting
"""

import warnings
from unittest.mock import patch

from megatron.bridge.utils.decorators import experimental_fn


class TestExperimentalFn:
    def test_returns_underlying_value(self):
        @experimental_fn
        def add(a, b):
            return a + b

        with patch("megatron.bridge.utils.decorators.get_rank_safe", return_value=0):
            assert add(2, 3) == 5

    def test_preserves_name_and_docstring(self):
        @experimental_fn
        def greet(name: str) -> str:
            """Say hi to someone."""
            return f"hi {name}"

        # functools.wraps preserves __name__, __doc__, and __wrapped__
        assert greet.__name__ == "greet"
        assert greet.__doc__ == "Say hi to someone."
        assert hasattr(greet, "__wrapped__")

    def test_emits_warning_on_rank_zero(self):
        @experimental_fn
        def widget():
            return 42

        with patch("megatron.bridge.utils.decorators.get_rank_safe", return_value=0):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                widget()
        assert len(caught) == 1
        # The message should name the wrapped function and call out experimental status.
        message = str(caught[0].message)
        assert "widget" in message
        assert "experimental" in message.lower()

    def test_no_warning_on_non_zero_rank(self):
        @experimental_fn
        def widget():
            return 42

        with patch("megatron.bridge.utils.decorators.get_rank_safe", return_value=1):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                widget()
        assert caught == []

    def test_forwards_positional_and_keyword_args(self):
        captured = {}

        @experimental_fn
        def echo(*args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs
            return ("ok", args, kwargs)

        with patch("megatron.bridge.utils.decorators.get_rank_safe", return_value=0):
            result = echo(1, 2, three=3, four=4)

        assert captured["args"] == (1, 2)
        assert captured["kwargs"] == {"three": 3, "four": 4}
        assert result == ("ok", (1, 2), {"three": 3, "four": 4})

    def test_warning_includes_correct_stacklevel(self):
        # stacklevel=2 makes the warning appear to originate at the caller
        # of the decorated function, not inside the wrapper. Check via the
        # warning's filename attribute.
        @experimental_fn
        def widget():
            return 0

        with patch("megatron.bridge.utils.decorators.get_rank_safe", return_value=0):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                widget()  # this file is the caller
        assert len(caught) == 1
        # The recorded warning's filename should be this test file, not decorators.py.
        assert "test_decorators" in caught[0].filename

    def test_each_call_re_emits_warning(self):
        # The wrapper is not memoized — every call triggers a fresh warning.
        @experimental_fn
        def widget():
            return 0

        with patch("megatron.bridge.utils.decorators.get_rank_safe", return_value=0):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                widget()
                widget()
                widget()
        assert len(caught) == 3
