import argparse
import time
import random
import sysconfig

import torch
import torch_npu
import numpy as np

torch.npu.config.allow_internal_format = False
torch.ops.load_library(f"{sysconfig.get_path('purelib')}/libfbgemm_npu_api.so")

batch_size = 1000


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=mode_library.keys(), default='prof')
    parser.add_argument('--npu', choices=list(range(8)), default=0, type=int)
    return parser.parse_args()


def fix_random_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch_npu.npu.manual_seed(seed)
    torch_npu.npu.manual_seed_all(seed)


def estimate_batch(grady_batch, index_batch, device):
    n = grady_batch.shape[0]
    grady_batch = torch.from_numpy(grady_batch).to(torch.float32).to(device)
    index_batch = torch.from_numpy(index_batch).to(torch.int64).to(device)
    x = torch.zeros(1, dtype=torch.float32).to(device)

    for _, grady, index in zip(range(n // 10), grady_batch, index_batch):
        torch.ops.mxrec.index_select_for_rank1_backward(grady, x, index)
    torch_npu.npu.synchronize()

    start = time.time_ns()
    for grady, index in zip(grady_batch, index_batch):
        torch.ops.mxrec.index_select_for_rank1_backward(grady, x, index)
    torch_npu.npu.synchronize()
    end = time.time_ns()

    return end - start


def profiling(device_id):
    device = f"npu:{device_id}"
    torch.npu.set_device(device)

    index_shape, x_dim = 102400, 128
    grady_batch = np.random.randn(batch_size, index_shape)
    index_batch = np.random.randint(x_dim, size=(batch_size, index_shape))
    with torch_npu.profiler.profile():
        estimate_batch(grady_batch, index_batch, device)


def costime(device_id):
    device = f"npu:{device_id}"
    torch.npu.set_device(device)

    print("x_dim\tindex_shape\tcost")
    for x_dim in [128, 256, 512, 1024]:
        for index_shape in [1024 * (10**i) for i in range(4)]:
            grady_batch = np.random.randn(batch_size, index_shape)
            index_batch = np.random.randint(x_dim, size=(batch_size, index_shape))
            cost = estimate_batch(grady_batch, index_batch, device)

            print(f"{x_dim}\t{index_shape}\t{cost/batch_size:.3f}")


def main():
    fix_random_seeds()
    args = parse_args()
    mode_library[args.mode](args.npu)


if __name__ == '__main__':
    # msprof --output=./ --app="python3 estimate_gather_for_rank1.py --mode prof --npu 0"
    mode_library = {
        "prof": profiling,
        "costime": costime,
    }
    main()
