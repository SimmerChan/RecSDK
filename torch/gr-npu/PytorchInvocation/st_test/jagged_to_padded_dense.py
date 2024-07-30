import torch
import torch_npu
import fbgemm_gpu
import numpy as np


DENSE_DIM = (128, 210, 1)
denses = np.random.randn(*DENSE_DIM).astype(np.float32)
offsets = np.random.randint(0, DENSE_DIM[1], DENSE_DIM[0])


def get_grad(device):
    dense_torch = torch.nn.Parameter(torch.from_numpy(denses).to(torch.float32)).to(device)
    dense_torch.retain_grad()

    offsets_torch = torch.from_numpy(offsets).to(torch.int64).to(device)

    jagged_id_offset = torch.ops.fbgemm.asynchronous_complete_cumsum(offsets_torch)

    outputSize = jagged_id_offset[-1]

    jagged_embeding=torch.ops.fbgemm.dense_to_jagged(dense_torch, [jagged_id_offset], outputSize)[0]

    output_embeddings=torch.ops.fbgemm.jagged_to_padded_dense(jagged_embeding, [jagged_id_offset], max_lengths=[DENSE_DIM[1]], padding_value=0.0)

    loss = torch.mean(output_embeddings)
    loss.backward()

    return dense_torch.grad.cpu().clone()


gloden = get_grad(torch.device("cpu"))
npu_result = get_grad(torch.device("npu"))
result = torch.abs(gloden-npu_result)<0.0001
print(result.all().item())