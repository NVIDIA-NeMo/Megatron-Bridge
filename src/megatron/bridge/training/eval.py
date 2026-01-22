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

import math
import time
from collections.abc import Callable
from typing import Any

import torch
from megatron.core.full_cuda_graph import FullCudaGraphWrapper
from megatron.core.num_microbatches_calculator import get_num_microbatches
from megatron.core.pipeline_parallel import get_forward_backward_func
from megatron.core.pipeline_parallel.p2p_communication import P2PCommunicator
from megatron.core.pipeline_parallel.utils import is_pp_last_stage
from megatron.core.rerun_state_machine import RerunDataIterator, RerunMode, get_rerun_state_machine
from megatron.core.transformer import MegatronModule
from megatron.core.transformer.enums import CudaGraphScope
from megatron.core.utils import get_model_config
from modelopt.torch.distill.plugins.megatron import get_tensor_shapes_adjust_fn_for_distillation

from megatron.bridge.data.finetuning import prepare_finetuning_batch
from megatron.bridge.data.iterator_utils import make_data_iterator_list
from megatron.bridge.training import fault_tolerance
from megatron.bridge.training.callbacks import CallbackContext, CallbackManager, should_fire
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.forward_step_func_types import ForwardStepCallable
from megatron.bridge.training.state import GlobalState
from megatron.bridge.training.utils.pg_utils import get_pg_collection
from megatron.bridge.training.utils.train_utils import prepare_forward_step_func
from megatron.bridge.utils.common_utils import is_last_rank, print_rank_0, print_rank_last


def evaluate(
    state: GlobalState,
    forward_step_func: ForwardStepCallable,
    data_iterator: RerunDataIterator | list[RerunDataIterator] | None,
    model: list[MegatronModule],
    process_non_loss_data_func: Callable | None,
    config: ConfigContainer,
    verbose: bool = False,
    non_loss_data_func: Callable | None = None,
    callback_manager: CallbackManager | None = None,
    is_test: bool = False,
) -> tuple[dict[str, torch.Tensor] | None, Any | None, bool]:
    """Evaluation function.

    Args:
        state (GlobalState): The global state object.
        forward_step_func (Callable): The function that performs a forward step.
        data_iterator (RerunDataIterator | list[RerunDataIterator] | None): Iterator over evaluation data.
        model (list[MegatronModule]): list of model chunks.
        process_non_loss_data_func (Callable | None): Function to process non-loss data.
        config (ConfigContainer): Configuration container (potentially redundant).
        verbose (bool, optional): Whether to print evaluation progress. Defaults to False.
        non_loss_data_func (Callable | None, optional): Function to compute non-loss data. Defaults to None.
        callback_manager (CallbackManager | None): Optional callback manager for firing callbacks.
        is_test (bool, optional): Whether this is test evaluation (vs validation). Defaults to False.
            Controls which callback events are fired (on_test_* vs on_eval_*).

    Returns:
        tuple[dict[str, torch.Tensor] | None, Any | None, bool]: A tuple containing:
            - total_loss_dict: Dictionary of averaged losses.
            - collected_non_loss_data: Data collected by non_loss_data_func.
            - timelimit_hit: Boolean indicating if the time limit was reached.
    """
    # Determine callback event names based on whether this is test or eval
    step_start_event = "on_test_step_start" if is_test else "on_eval_step_start"
    step_end_event = "on_test_step_end" if is_test else "on_eval_step_end"
    # Prepare forward_step_func (check signature and inject state if needed)
    # This is done once to prevent creating new partial objects every eval iteration
    wrapped_forward_step = prepare_forward_step_func(forward_step_func, state)

    timers = state.timers
    timers("evaluate", log_level=0).start(barrier=True)

    # Turn on evaluation mode which disables dropout.
    for model_module in model:
        model_module.eval()

    # Retrieve process group collection and model config from the model
    pg_collection = get_pg_collection(model)
    model_config = get_model_config(model[0])

    # Disable result validation during evaluation
    rerun_state_machine = get_rerun_state_machine()
    rerun_mode = rerun_state_machine.get_mode()
    rerun_state_machine.set_mode(RerunMode.DISABLED)

    total_loss_dict = {}

    # make validation batch size independent from training batch size
    eval_batch_size = state.cfg.train.global_batch_size
    eval_num_microbatches = eval_batch_size // (state.cfg.train.micro_batch_size * state.cfg.data_parallel_size)

    if not state.cfg.dist.use_decentralized_pg:
        adjust_tensor_shapes_fn = get_tensor_shapes_adjust_fn_for_distillation(
            model,
            seq_length=state.cfg.model.seq_length,
            micro_batch_size=state.cfg.train.micro_batch_size,
            decoder_seq_length=state.cfg.model.seq_length,
        )
    else:
        adjust_tensor_shapes_fn = None

    with torch.no_grad():
        if verbose:
            print_rank_0(f"Evaluating on {state.cfg.validation.eval_iters * eval_batch_size} samples")

        if (
            state.cfg.model.cuda_graph_impl == "local"
            and CudaGraphScope.full_iteration in state.cfg.model.cuda_graph_scope
        ):
            forward_backward_func = FullCudaGraphWrapper(
                get_forward_backward_func(
                    pp_size=pg_collection.pp.size(),
                    vp_size=state.cfg.model.virtual_pipeline_model_parallel_size,
                ),
                cuda_graph_warmup_steps=state.cfg.model.cuda_graph_warmup_steps,
            )
        else:
            forward_backward_func = get_forward_backward_func(
                pp_size=pg_collection.pp.size(),
                vp_size=state.cfg.model.virtual_pipeline_model_parallel_size,
            )

        iteration = 0
        while iteration < state.cfg.validation.eval_iters:
            iteration += 1
            if verbose:
                print_rank_0(f"Evaluating iter {iteration}/{state.cfg.validation.eval_iters}")

            # Handle finetuning vs pretraining data consumption
            seq_length = state.cfg.model.seq_length  # Default for pretraining
            eval_data_iterator = data_iterator  # Default for pretraining

            if state.cfg.dataset.dataloader_type == "batch":
                # Finetuning path: prepare batch and extract dynamic seq_length
                eval_data_iterator, seq_length = prepare_finetuning_batch(
                    data_iterator=data_iterator,
                    num_microbatches=eval_num_microbatches,
                    default_seq_length=state.cfg.model.seq_length,
                    seq_key="tokens",
                )

            if len(model) > 1:
                # Convert to list of iterators for virtual pipeline parallelism
                # With virtual PP, each model chunk needs independent access to the same microbatch
                eval_data_iterator = make_data_iterator_list(
                    model=model,
                    data_iterator=eval_data_iterator,
                )

            # Don't care about timing during evaluation
            config.timers = None
            fault_tolerance.on_eval_step_start(state)
            p2p_communicator = P2PCommunicator(pp_group=pg_collection.pp, config=model_config)

            if should_fire(callback_manager, step_start_event):
                callback_manager.fire(
                    step_start_event,
                    CallbackContext(
                        state=state,
                        model=model,
                        user_state=callback_manager.user_state,
                    ),
                )

            loss_dicts = forward_backward_func(
                forward_step_func=wrapped_forward_step,
                data_iterator=eval_data_iterator,
                model=model,
                num_microbatches=eval_num_microbatches,
                seq_length=seq_length,
                micro_batch_size=state.cfg.train.micro_batch_size,
                forward_only=True,
                adjust_tensor_shapes_fn=adjust_tensor_shapes_fn,
                p2p_communicator=p2p_communicator,
                pg_collection=pg_collection,
            )
            fault_tolerance.on_eval_step_end(state)

            if should_fire(callback_manager, step_end_event):
                callback_manager.fire(
                    step_end_event,
                    CallbackContext(
                        state=state,
                        model=model,
                        user_state=callback_manager.user_state,
                    ),
                )

            config.timers = state.timers

            # Empty unused memory
            if state.cfg.train.empty_unused_memory_level >= 1:
                torch.cuda.empty_cache()

            if is_pp_last_stage(pg_collection.pp):
                # Reduce across processes.
                for key in loss_dicts[0].keys():
                    if key not in total_loss_dict:
                        total_loss_dict[key] = torch.tensor([0.0, 0.0], dtype=torch.float).cuda()
                    val = [x[key].view(-1) for x in loss_dicts]

                    if val[0].numel() == 2:
                        val = torch.vstack(val).sum(dim=0)
                        torch.distributed.all_reduce(val, group=pg_collection.dp_cp)
                        total_loss_dict[key] += val
                    elif val[0].numel() == 1:
                        val = torch.cat(val).sum()
                        total_loss_dict[key][0] += val
                        total_loss_dict[key][1] += len(loss_dicts)
                    else:
                        raise ValueError(f"Invalid value shape: {val[0].shape} for key {key}")

            state.train_state.consumed_valid_samples += eval_batch_size

            if state.cfg.train.exit_duration_in_mins:
                train_time = (time.time() - state.start_time) / 60.0
                done_cuda = torch.tensor(
                    [train_time > state.cfg.train.exit_duration_in_mins], dtype=torch.int, device="cuda"
                )
                torch.distributed.all_reduce(done_cuda, op=torch.distributed.ReduceOp.MAX)
                done = done_cuda.item()
                if done:
                    rerun_state_machine.set_mode(rerun_mode)
                    print_rank_0("Exiting during evaluation, timelimit reached")
                    return None, None, True

        collected_non_loss_data = None
        if non_loss_data_func is not None:
            collected_non_loss_data = non_loss_data_func(model)
        elif process_non_loss_data_func is not None and is_last_rank():
            # Handle finetuning vs pretraining for non-loss data collection
            non_loss_data_iterator = data_iterator
            non_loss_seq_length = state.cfg.model.seq_length

            if state.cfg.dataset.dataloader_type == "batch":
                # Finetuning path: prepare batch and wrap for VPP
                non_loss_microbatch_iterator, non_loss_seq_length = prepare_finetuning_batch(
                    data_iterator=data_iterator,
                    num_microbatches=get_num_microbatches(),
                    default_seq_length=state.cfg.model.seq_length,
                    seq_key="tokens",
                )
                non_loss_data_iterator = make_data_iterator_list(
                    model=model,
                    data_iterator=non_loss_microbatch_iterator,
                )

            p2p_communicator = P2PCommunicator(pp_group=pg_collection.pp, config=model_config)
            collected_non_loss_data = forward_backward_func(
                forward_step_func=wrapped_forward_step,
                data_iterator=non_loss_data_iterator,
                model=model,
                num_microbatches=get_num_microbatches(),
                seq_length=non_loss_seq_length,
                micro_batch_size=state.cfg.train.micro_batch_size,
                forward_only=True,
                collect_non_loss_data=True,
                p2p_communicator=p2p_communicator,
                pg_collection=pg_collection,
            )

    # Move model back to the train mode.
    for model_module in model:
        model_module.train()

    for key in total_loss_dict:
        numerator, denominator = total_loss_dict[key]
        total_loss_dict[key] = numerator / denominator

    timers("evaluate").stop()
    timers.log(["evaluate"])

    rerun_state_machine.set_mode(rerun_mode)

    return total_loss_dict, collected_non_loss_data, False


def evaluate_and_print_results(
    state: GlobalState,
    prefix: str,
    forward_step_func: ForwardStepCallable,
    data_iterator: RerunDataIterator | list[RerunDataIterator] | None,
    model: list[MegatronModule],
    config: ConfigContainer,
    verbose: bool = False,
    write_to_tensorboard: bool = True,
    process_non_loss_data_func: Callable | None = None,
    non_loss_data_func: Callable | None = None,
    callback_manager: CallbackManager | None = None,
    is_test: bool = False,
) -> None:
    """Helper function to evaluate and dump results on screen.

    Args:
        state (GlobalState): The global state object.
        prefix (str): Prefix for logging evaluation results.
        forward_step_func (Callable): The function that performs a forward step.
        data_iterator (RerunDataIterator | list[RerunDataIterator] | None): Iterator over evaluation data.
        model (list[MegatronModule]): list of model chunks.
        config (ConfigContainer): Configuration container (potentially redundant).
        verbose (bool, optional): Whether to print evaluation progress. Defaults to False.
        write_to_tensorboard (bool, optional): Whether to write results to TensorBoard. Defaults to True.
        process_non_loss_data_func (Callable | None, optional): Function to process non-loss data. Defaults to None.
        non_loss_data_func (Callable | None, optional): Function to compute non-loss data. Defaults to None.
        callback_manager (CallbackManager | None): Optional callback manager for firing callbacks.
        is_test (bool, optional): Whether this is test evaluation (vs validation). Defaults to False.
            Controls which callback events are fired (on_test_* vs on_eval_*).
    """
    # Determine callback event names based on whether this is test or eval
    start_event = "on_test_start" if is_test else "on_eval_start"
    end_event = "on_test_end" if is_test else "on_eval_end"

    if write_to_tensorboard:
        writer = state.tensorboard_logger
    else:
        writer = None

    wandb_writer = state.wandb_logger

    if should_fire(callback_manager, start_event):
        callback_manager.fire(
            start_event,
            CallbackContext(
                state=state,
                model=model,
                user_state=callback_manager.user_state,
            ),
        )

    total_loss_dict, collected_non_loss_data, timelimit = evaluate(
        state,
        forward_step_func,
        data_iterator,
        model,
        process_non_loss_data_func,
        config,
        verbose,
        non_loss_data_func,
        callback_manager=callback_manager,
        is_test=is_test,
    )

    # Timelimit hit during evaluation
    if timelimit:
        return
    string = f" validation loss at {prefix} | "
    for key in total_loss_dict:
        string += "{} value: {:.6E} | ".format(key, total_loss_dict[key].item())
        ppl = math.exp(min(20, total_loss_dict[key].item()))
        string += "{} PPL: {:.6E} | ".format(key, ppl)
        if writer:
            writer.add_scalar("{} validation".format(key), total_loss_dict[key].item(), state.train_state.step)
            writer.add_scalar(
                "{} validation vs samples".format(key),
                total_loss_dict[key].item(),
                state.train_state.consumed_train_samples,
            )
            if state.cfg.logger.log_validation_ppl_to_tensorboard:
                writer.add_scalar("{} validation ppl".format(key), ppl, state.train_state.step)
                writer.add_scalar(
                    "{} validation ppl vs samples".format(key), ppl, state.train_state.consumed_train_samples
                )

        if wandb_writer and is_last_rank():
            wandb_writer.log({"{} validation".format(key): total_loss_dict[key].item()}, state.train_state.step)
            if state.cfg.logger.log_validation_ppl_to_tensorboard:
                wandb_writer.log({"{} validation ppl".format(key): ppl}, state.train_state.step)

    if process_non_loss_data_func is not None and writer and is_last_rank():
        process_non_loss_data_func(collected_non_loss_data, state.train_state.step, writer)

    length = len(string) + 1
    print_rank_last("-" * length)
    print_rank_last(string)
    print_rank_last("-" * length)

    if should_fire(callback_manager, end_event):
        callback_manager.fire(
            end_event,
            CallbackContext(
                state=state,
                model=model,
                user_state=callback_manager.user_state,
                total_loss_dict=total_loss_dict,
            ),
        )
