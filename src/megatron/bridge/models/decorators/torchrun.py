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

import json
import os
import sys
import time
import traceback
from functools import wraps

import torch
from torch.distributed.elastic.multiprocessing.errors import record


def _write_torchelastic_error_file(exc: BaseException) -> None:
    """
    Write the same JSON shape as torch.distributed.elastic ErrorHandler.record_exception.

    Needed when the inner ``@record`` path does not run ``record_exception`` (e.g.
    ``SystemExit`` is not an ``Exception``), or when the worker exits in a way that
    would otherwise leave ``error.json`` missing so torchelastic reports
    ``error_file: <N/A>``.
    """
    path = (os.environ.get("TORCHELASTIC_ERROR_FILE") or "").strip()
    if not path:
        return
    try:
        data = {
            "message": {
                "message": f"{type(exc).__name__}: {exc}",
                "extraInfo": {
                    "py_callstack": traceback.format_exc(),
                    "timestamp": str(int(time.time())),
                },
            },
        }
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fp:
            json.dump(data, fp)
    except OSError:
        pass


def torchrun_main(fn):
    """
    A decorator that wraps the main function of a torchrun script. It uses
    the `torch.distributed.elastic.multiprocessing.errors.record` decorator
    to record any exceptions and ensures that the distributed process group
    is properly destroyed on successful completion. In case of an exception,
    it prints the traceback and exits so torchelastic can read ``error.json``.
    """
    recorded_fn = record(fn)

    @wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return_value = recorded_fn(*args, **kwargs)
            if torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()
            return return_value
        except (KeyboardInterrupt, GeneratorExit):
            raise
        except SystemExit as e:
            if e.code in (0, None, False):
                raise
            traceback.print_exc()
            print(f"{type(e).__name__}: {e}", file=sys.stderr)
            _write_torchelastic_error_file(e)
            sys.stderr.flush()
            sys.stdout.flush()
            code = e.code
            if isinstance(code, int):
                sys.exit(code)
            sys.exit(1)
        except BaseException as e:
            # The inner ``record`` decorator writes TORCHELASTIC_ERROR_FILE for
            # Exception, but we refresh it here so the stack includes outer frames.
            traceback.print_exc()
            _write_torchelastic_error_file(e)
            sys.stderr.flush()
            sys.stdout.flush()
            # Prefer sys.exit over os._exit so stdio and error.json are fully flushed
            # and torchelastic can surface tracebacks instead of ``error_file: <N/A>``.
            sys.exit(1)

    return wrapper
