import torch
from typing import List
from torch import nn
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import torch.distributed as dist


class Awaitable:
    def __init__(self, a_function, *args):
        self.result = executor.submit(a_function, *args)

    def wait(self):
        return self.result.result()


executor = ThreadPoolExecutor(6)


class PostInpuDistAwaitable(Awaitable):
    pass


class LookupAndOutputDist(Awaitable):
    def __init__(self, post_awaitable, lookup_and_out_dist_function, *args):
        self.post_awaitable = post_awaitable
        self.lookup_and_out_dist_function = lookup_and_out_dist_function
        self.args = args

    def wait(self):
        post_result = self.post_awaitable.wait()
        return self.lookup_and_out_dist_function(post_result, *self.args)


class LookupAwaitable:
    def __init__(self, lookup_function, *args):
        self.result = executor.submit(lookup_function, *args)

    def wait(self):
        return self.result.result()


@dataclass
class EmbeddingConfig:
    num_embedding: int = 0
    embedding_dim: int = 0
    optimizer: torch.optim.Optimizer = 0
    world_sie: int = 0
    rank: int = 0


class AllGatherEmbedding(torch.autograd.Function):
    @staticmethod
    def forward(ctx, fwd_pg, bwd_pg, embedding: torch.Tensor):
        ctx.fwd_gp = fwd_pg
        ctx.bwd_pg = bwd_pg
        ctx.embedding = embedding.data
        result_list = [torch.empty_like(embedding) for i in range(2)]
        fwd_pg.allgather(result_list, embedding)
        return tuple(result_list)

    @staticmethod
    def backward(ctx, *grad_output: torch.Tensor):
        grad_output = [g.data for g in grad_output]
        result = torch.empty_like(grad_output[0])
        if grad_output[0].device != torch.device("cpu"):
            ctx.bwd_pg.reduce_scatter(result, grad_output_list)
        else:
            # 模拟reduce scatter
            rank = ctx.bwd_pg.rank()
            grad_output_concat = torch.concat(grad_output)
            ctx.bwd_pg.allreduce(grad_output_concat, op=dist.reduce_op.SUM)
            result = grad_output_concat.split(6)[rank]
        return None, None, result


class HashEmbeddingModule(nn.Module):
    def __init__(self, config=None, pipe_n_batch=6):
        super().__init__()
        self.pg = dist.new_group(backend="gloo")
        self.rank = config.rank
        self.post_input_dist_module = self.create_post_input_dist()
        self.lookup_module = self.create_lookup()

    def create_post_input_dist(self):
        return

    def create_lookup(self):
        return

    # 示例代码
    def fids2indices(self, fids: List[torch.Tensor]):
        return fids

    # 示例代码
    def lookup(self, fids: List[torch.Tensor]):
        result = []
        test_num = 10
        for a_id in fids:
            result.append([a_id / test_num, a_id / test_num])
        return nn.Parameter(torch.Tensor(result), requires_grad=True)

    def post_input_dist(self, input_fid_list_each_rank):
        fid_need_this_rank_lookup = [
            input_fid_list[self.rank] for input_fid_list in input_fid_list_each_rank
        ]
        fid_need_this_rank_lookup = torch.concat(fid_need_this_rank_lookup)
        indices = self.fids2indices(fid_need_this_rank_lookup)
        return indices

    def lookup_and_post_dist(self, indices):
        embedding = self.lookup(indices)
        result = AllGatherEmbedding().apply(self.pg, self.pg, embedding)
        return result

    def forward(self, input_fids_list: List[torch.Tensor], is_full_pipe=True):
        post_awaitable = PostInpuDistAwaitable(self.post_input_dist, input_fids_list)
        lookup_and_outdist_awaitable = LookupAndOutputDist(
            post_awaitable, self.lookup_and_post_dist
        )
        return lookup_and_outdist_awaitable
