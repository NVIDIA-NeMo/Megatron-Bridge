import os
import traceback
from functools import wraps

import torch
from torch.distributed.elastic.multiprocessing.errors import record


def torchrun_main(fn):
    """
    A decorator that wraps the main function of a torchrun script. It uses
    the `torch.distributed.elastic.multiprocessing.errors.record` decorator
    to record any exceptions and ensures that the distributed process group
    is properly destroyed on successful completion. In case of an exception,
    it prints the traceback and performs a hard exit, allowing torchrun to
    terminate all other processes.
    """
    recorded_fn = record(fn)

    @wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return_value = recorded_fn(*args, **kwargs)
            if torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()
            return return_value
        except Exception:
            # The 'record' decorator might only log the exception to a file.
            # Print it to stderr as well to make sure it's visible.
            traceback.print_exc()
            # Use os._exit(1) for a hard exit. A regular sys.exit(1) might
            # not be enough to terminate a process stuck in a bad C++ state
            # (e.g., after a NCCL error), which can cause the job to hang.
            os._exit(1)

    return wrapper
