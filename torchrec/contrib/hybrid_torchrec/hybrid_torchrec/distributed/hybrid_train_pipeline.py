#!/usr/bin/env python3
# Copyright (c) Huawei Platforms, Inc. and affiliates.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
# ==============================================================================
import logging
from dataclasses import dataclass, field
from typing import (
    Any,
    cast,
    Dict,
    Iterator,
    List,
    Optional,
    Tuple,
    TypeVar,
)

import torch
from torch.autograd.profiler import record_function
from torchrec.distributed.embedding_types import KJTList
from torchrec.distributed.types import Awaitable
from torchrec.streamable import Multistreamable, Pipelineable
from torchrec.distributed.train_pipeline.utils import (
    FusedKJTListSplitsAwaitable,
    _to_device,
    _rewrite_model,
    _wait_for_batch,
    _override_input_dist_forwards,
    _start_data_dist,
    PipelinedForward,
)
from torchrec.distributed.train_pipeline.train_pipelines import TrainPipelineSparseDist
import torch_npu

logger: logging.Logger = logging.getLogger(__name__)


In = TypeVar("In", bound=Pipelineable)
Out = TypeVar("Out")

FRONT_BATCH = 0
SECOND_BATCH = 1
THIRD_BATCH = 2


@dataclass
class HybridTrainPipelineContext:
    """
    Context information for a `TrainPipelineSparseDist` instance.

    Attributes:
        input_dist_splits_requests (Dict[str, Awaitable[Any]]): Stores input dist
            requests in the splits awaitable stage, which occurs after starting the
            input dist.
        input_dist_tensors_requests (Dict[str, Awaitable[Any]]): Stores input dist
            requests in the tensors awaitable stage, which occurs after calling `wait()`
            on the splits awaitable.
        module_contexts (Dict[str, Multistreamable]): Stores module contexts from the
            input dist for the current batch.
        module_contexts_next_batch (Dict[str, Multistreamable]): Stores module contexts
            from the input dist for the next batch.
        fused_splits_awaitables (List[Tuple[List[str], FusedKJTListSplitsAwaitable]]):
            List of fused splits input dist awaitable and the corresponding module names
            of each awaitable.
    """

    input_dist_splits_requests: Dict[str, Awaitable[Any]] = field(default_factory=dict)
    input_dist_tensors_requests: Dict[str, Awaitable[Any]] = field(default_factory=dict)
    input_dist_tensors_requests_post: Dict[str, Awaitable[Any]] = field(
        default_factory=dict
    )
    input_dist_tensors_requests_to_device: Dict[str, Awaitable[Any]] = field(
        default_factory=dict
    )
    input_dist_tensors_requests_forward: Dict[str, Awaitable[Any]] = field(
        default_factory=dict
    )

    module_contexts: Dict[str, Multistreamable] = field(default_factory=dict)
    module_contexts_next_batch: Dict[str, Multistreamable] = field(default_factory=dict)
    module_contexts_post: Dict[str, Multistreamable] = field(default_factory=dict)
    module_contexts_forward: Dict[str, Multistreamable] = field(default_factory=dict)

    fused_splits_awaitables: List[Tuple[List[str], FusedKJTListSplitsAwaitable]] = (
        field(default_factory=list)
    )
    preproc_fwd_results: Dict[str, Any] = field(default_factory=dict)
    index: Optional[int] = None
    version: int = 0


class HybridPipelinedForward(PipelinedForward):
    def __call__(self, *input, **kwargs) -> Awaitable:
        self._context: HybridTrainPipelineContext
        if self._name not in self._context.input_dist_tensors_requests_forward:
            raise ValueError(
                f"{self._name} is not in input_dist_tensors_requests_to_device"
            )
        data = self._context.input_dist_tensors_requests_forward[self._name]

        if self._stream is not None:
            torch_npu.npu.current_stream().wait_stream(self._stream)
            cur_stream = torch_npu.npu.current_stream()

            if not isinstance(data, (torch.Tensor, Multistreamable)):
                raise ValueError(
                    f"{type(data)} must implement Multistreamable interface"
                )

            data.record_stream(cur_stream)

            ctx = self._context.module_contexts_forward[self._name]
            ctx.record_stream(cur_stream)

        return self._module.compute_and_output_dist(
            self._context.module_contexts_forward[self._name], data
        )

    def set_current_context(self, context: HybridTrainPipelineContext):
        self._context = context


def kjt_list_to_device(batch: KJTList, device: torch.device, non_blocking: bool) -> In:
    if not isinstance(batch, (KJTList)):
        raise ValueError(f"{type(batch)} must be KJTList")
    result = KJTList(
        [
            f.pin_memory().to(device=device, non_blocking=non_blocking)
            for f in batch.features
        ]
    )
    return result


class HybridTrainPipelineSparseDist(TrainPipelineSparseDist[In, Out]):
    """
    This pipeline overlaps device transfer, and `ShardedModule.input_dist()` with
    forward and backward. This helps hide the all2all latency while preserving the
    training forward / backward ordering.

    stage 3: forward, backward - uses default CUDA stream
    stage 2: ShardedModule.input_dist() - uses data_dist CUDA stream
    stage 1: device transfer - uses memcpy CUDA stream

    `ShardedModule.input_dist()` is only done for top-level modules in the call graph.
    To be considered a top-level module, a module can only depend on 'getattr' calls on
    input.

    Input model must be symbolically traceable with the exception of `ShardedModule` and
    `DistributedDataParallel` modules.

    Args:
        model (torch.nn.Module): model to pipeline.
        optimizer (torch.optim.Optimizer): optimizer to use.
        device (torch.device): device where device transfer, sparse data dist, and
            forward/backward pass will happen.
        execute_all_batches (bool): executes remaining batches in pipeline after
            exhausting dataloader iterator.
        apply_jit (bool): apply torch.jit.script to non-pipelined (unsharded) modules.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        execute_all_batches: bool = True,
        apply_jit: bool = False,
        return_loss: bool = False,
        pipe_n_batch: int = 6,
    ) -> None:
        super().__init__(model, optimizer, device, execute_all_batches, apply_jit)
        self._return_loss = return_loss
        self._batch = [[None, None, None, None] for _ in range(pipe_n_batch)]

        self._context = [HybridTrainPipelineContext() for _ in range(pipe_n_batch)]
        self._current_line_id = 0
        self._batch_forward = None
        self._pipe_n_batch = pipe_n_batch

    def progress(self, dataloader_iter: Iterator[In]) -> Out:
        self._fill_pipeline(dataloader_iter)

        if self._model.training:
            with record_function("## zero_grad ##"):
                self._optimizer.zero_grad()

        cur_line_id = self._current_line_id % self._pipe_n_batch
        next_line_id = (self._current_line_id + 1) % self._pipe_n_batch

        cur_context_forward = self._context[cur_line_id]
        cur_batch_forward = self._batch_forward

        with record_function("## wait_for_batch ##"):
            self._wait_for_batch(
                cast(In, cur_batch_forward), self._memcpy_stream, cur_context_forward
            )

        # 此条线的顶端batch已经结束，下一条线还是拷贝到npu上
        self._batch[next_line_id][FRONT_BATCH] = self._copy_to_npu(
            self._batch[next_line_id][FRONT_BATCH], self._context[next_line_id]
        )

        # 此条线的顶端batch已经结束，此条线的下一个batch准备
        self._do_post_input_dist(self._context[cur_line_id])
        self._batch[cur_line_id][FRONT_BATCH] = self._batch[cur_line_id][SECOND_BATCH]

        with record_function("## forward ##"):
            losses, output = cast(
                Tuple[torch.Tensor, Out], self._model(cur_batch_forward)
            )

        # 每一个batch向前移动一个位置
        self._wait_sparse_data_dist(self._context[cur_line_id])
        self._batch[cur_line_id][SECOND_BATCH] = self._batch[cur_line_id][THIRD_BATCH]

        if self._model.training:
            # backward
            with record_function("## backward ##"):
                torch.sum(losses, dim=0).backward()

        # 每一个batch向前移动一个位置
        self._batch[cur_line_id][THIRD_BATCH] = self._get_next_batch(dataloader_iter)
        self._start_sparse_data_dist(
            self._batch[cur_line_id][THIRD_BATCH],
            self._context[cur_line_id],
        )

        if self._model.training:
            # update
            with record_function("## optimizer ##"):
                self._optimizer.step()

        self._current_line_id += 1
        self._batch_forward = self._batch[next_line_id][FRONT_BATCH]

        if self._return_loss:
            return output, losses
        else:
            return output

    def _get_next_batch(self, dataloader_iter):
        with record_function("## get_next_batch ##"):
            batch = next(dataloader_iter, None)
            if batch is None and not self._execute_all_batches:
                raise StopIteration
            return batch

    def _fill_pipeline(self, dataloader_iter: Iterator[In]) -> None:
        # executes last batch in pipeline
        if self._batch_forward and self._execute_all_batches:
            return

        start_id = self._current_line_id % self._pipe_n_batch

        # batch 1
        batch_start = self._get_next_batch(dataloader_iter)

        if batch_start is None:
            raise StopIteration
        self._init_pipelined_modules(batch_start, context=self._context[start_id])

        pipe_index = 0
        # 1级流水
        for batch_i in range(self._pipe_n_batch):
            if batch_i == start_id:
                self._batch_forward = batch_start
                self._batch[batch_i][pipe_index] = batch_start
            else:
                self._batch[batch_i][pipe_index] = self._get_next_batch(dataloader_iter)
            self._start_sparse_data_dist(
                self._batch[batch_i][pipe_index], context=self._context[batch_i]
            )
            self._wait_sparse_data_dist(context=self._context[batch_i])
            self._do_post_input_dist(context=self._context[batch_i])

        self._batch_forward = self._copy_to_npu(
            self._batch[start_id][pipe_index], self._context[start_id]
        )

        # 2级流水
        pipe_index = 1
        for next_cur_id in range(self._pipe_n_batch):
            self._batch[next_cur_id][pipe_index] = self._get_next_batch(dataloader_iter)
            self._start_sparse_data_dist(
                self._batch[next_cur_id][pipe_index], context=self._context[next_cur_id]
            )
            self._wait_sparse_data_dist(context=self._context[next_cur_id])

        # 2级流水
        pipe_index = 2
        for next_cur_id in range(self._pipe_n_batch):
            self._batch[next_cur_id][pipe_index] = self._get_next_batch(dataloader_iter)
            self._start_sparse_data_dist(
                self._batch[next_cur_id][pipe_index], context=self._context[next_cur_id]
            )

    def _wait_for_batch(
        self,
        batch: In,
        stream: Optional[torch.cuda.streams.Stream],
        context: HybridTrainPipelineContext,
    ) -> None:
        if batch is None:
            return
        if stream is not None:
            torch_npu.npu.current_stream().wait_stream(stream)

        context.input_dist_tensors_requests_forward.clear()
        for forward_name, module in zip(
            context.input_dist_tensors_requests_to_device.keys(),
            self._pipelined_modules,
        ):
            context.input_dist_tensors_requests_forward[forward_name] = (
                context.input_dist_tensors_requests_to_device[forward_name]
            )
            module.forward.set_current_context(context)

        if stream is None:
            return
        cur_stream = torch_npu.npu.current_stream()
        if not isinstance(batch, (torch.Tensor, Multistreamable)):

            raise ValueError(f"{type(batch)} must implement Multistreamable interface")
        batch.record_stream(cur_stream)
        for forward_name in context.input_dist_tensors_requests_forward.keys():
            context.input_dist_tensors_requests_forward[forward_name].record_stream(
                cur_stream
            )

    def _do_post_input_dist(self, context: HybridTrainPipelineContext):
        context.module_contexts_post = context.module_contexts.copy()
        context.input_dist_tensors_requests_post.clear()
        with record_function("## _post_input_dist ##"):
            for name, module in zip(
                context.input_dist_tensors_requests.keys(),
                self._pipelined_modules,
            ):
                awaitable = context.input_dist_tensors_requests[name]
                kjt_list = awaitable.wait()

                if hasattr(module, "post_input_dist"):
                    post_waitable = module.post_input_dist(
                        context.module_contexts_post[name], kjt_list
                    )
                    context.input_dist_tensors_requests_post[name] = post_waitable
                else:
                    raise RuntimeError(
                        "HybridTrainPipelineSparseDist can't be used for module with no post_input method"
                    )

    def _copy_to_npu(
        self, batch: In, context: HybridTrainPipelineContext
    ) -> Optional[In]:
        """
        Retrieves batch from dataloader and moves it to the provided device.

        Raises:
            StopIteration: if the dataloader iterator is exhausted; unless
                `self._execute_all_batches=True`, then returns None.
        """
        if batch is None:
            return None
        context.module_contexts_forward = context.module_contexts_post.copy()
        for preproc_mod in self._pipelined_preprocs:
            preproc_mod.set_context(context)
        with record_function("## _copy_to_npu ##"):
            with torch_npu.npu.stream(self._memcpy_stream):
                context.input_dist_tensors_requests_to_device.clear()

                for name in context.input_dist_tensors_requests_post.keys():
                    kjt_list = context.input_dist_tensors_requests_post[name].wait()
                    context.input_dist_tensors_requests_to_device[name] = (
                        kjt_list_to_device(kjt_list, self._device, non_blocking=True)
                    )

                if batch is not None:
                    batch = _to_device(batch, self._device, non_blocking=True)
                elif not self._execute_all_batches:
                    raise StopIteration
                return batch

    def _init_pipelined_modules(
        self, batch: In, context: HybridTrainPipelineContext
    ) -> None:
        """
        Retrieves the pipelined modules after overriding their forwards, initializes the
        modules' input dists, and overrides the input dist forwards to support fusing
        the splits collective in the input dist.
        """
        if self._pipelined_modules:
            return
        (
            self._pipelined_modules,
            self._model,
            self._original_forwards,
            self._pipelined_preprocs,
            self._non_pipelined_sharded_modules,
        ) = _rewrite_model(
            model=self._model,
            context=context,
            dist_stream=None,
            batch=batch,
            apply_jit=self._apply_jit,
            pipelined_forward=HybridPipelinedForward,
        )
        # initializes input dist, so we can override input dist forwards
        self._start_sparse_data_dist(batch, context)
        _override_input_dist_forwards(self._pipelined_modules)

    def _start_sparse_data_dist(
        self, batch: Optional[In], context: HybridTrainPipelineContext
    ) -> None:
        """
        Waits for batch to finish getting copied to GPU, then starts the input dist.
        """
        if batch is None:
            return
        with record_function("## start_sparse_data_dist ##"):
            # Temporarily set context for next iter to populate cache
            for preproc_mod in self._pipelined_preprocs:
                preproc_mod.set_context(context)
            _start_data_dist(self._pipelined_modules, batch, context)

    def _wait_sparse_data_dist(self, context: HybridTrainPipelineContext) -> None:
        """
        Waits on the input dist splits requests to get the input dist tensors requests,
        and populates the context with them.
        """
        with record_function("## wait_sparse_data_dist ##"):
            context.module_contexts = context.module_contexts_next_batch.copy()
            context.input_dist_tensors_requests.clear()
            for names, awaitable in context.fused_splits_awaitables:
                for name, request in zip(names, awaitable.wait()):
                    context.input_dist_tensors_requests[name] = request
