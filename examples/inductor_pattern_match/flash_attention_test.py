import math
import numpy as np
import torch
import torch_npu


def supported_op_exec(query_states1, past_key, past_value, head_dim, num_heads):
    query_states1 = query_states1.permute(0, 2, 1, 3)
    past_key = past_key.permute(0, 2, 1, 3)
    past_value = past_value.permute(0, 2, 1, 3)
    attn_weights1 = torch.matmul(query_states1, past_key.transpose(2, 3)) * (
        1.0 / math.sqrt(head_dim)
    )
    attn_weights1 = torch.max(
        attn_weights1,
        torch.full(
            (1, 1), torch.finfo(attn_weights1.dtype).min, device=attn_weights1.device
        ),
    )
    attn_weights1 = torch.nn.functional.softmax(
        attn_weights1, dim=-1, dtype=torch.float32
    ).to(query_states1.dtype)
    attn_output1 = torch.matmul(attn_weights1, past_value)
    return attn_output1.permute(0, 2, 1, 3)


def custom_op_exec(query, key, value, head_dim, num_heads):
    scale = 1.0 / math.sqrt(head_dim)
    return torch_npu.npu_prompt_flash_attention(
        query,
        key,
        value,
        num_heads=num_heads,
        input_layout="BSND",
        scale_value=scale,
        pre_tokens=65535,
        next_tokens=65535,
    )


def test_npu_prompt_flash_attention(device="npu"):
    head_dim = 128
    num_heads = 32
    batch_size = 10
    seq_len_q = 1
    seq_len_kv = 2048
    query = torch.randn(
        batch_size, seq_len_q, num_heads, head_dim, dtype=torch.float16, device=device
    )
    key = torch.randn(
        batch_size, seq_len_kv, num_heads, head_dim, dtype=torch.float16, device=device
    )
    value = torch.randn(
        batch_size, seq_len_kv, num_heads, head_dim, dtype=torch.float16, device=device
    )

    supported_output = (
        supported_op_exec(query, key, value, head_dim, num_heads).cpu().numpy()
    )
    custom_output = custom_op_exec(query, key, value, head_dim, num_heads).cpu().numpy()
    print("supported_output shape: \n", supported_output.shape)
    print("supported_output: \n", supported_output.reshape(-1)[0])
    print("custom_output shape: \n", custom_output.shape)
    print("custom_output: \n", custom_output.reshape(-1)[0])
    assert np.allclose(
        supported_output,
        custom_output,
        rtol=1e-05,
        atol=1e-05,
    )


def perf_flash_attention():
    head_dim = 128
    num_heads = 32
    batch_size = 10
    seq_len_q = 1
    seq_len_kv = 2048
    query = torch.randn(
        batch_size, seq_len_q, num_heads, head_dim, dtype=torch.float16, device="npu"
    )
    key = torch.randn(
        batch_size, seq_len_kv, num_heads, head_dim, dtype=torch.float16, device="npu"
    )
    value = torch.randn(
        batch_size, seq_len_kv, num_heads, head_dim, dtype=torch.float16, device="npu"
    )
    import time

    times = 10000
    start = time.time()
    for _ in range(times):
        custom_op_exec(query, key, value, head_dim, num_heads)
    end = time.time()
    print(f"{times}x fused op time cost: {end - start}s")
    start = time.time()
    for _ in range(times):
        supported_op_exec(query, key, value, head_dim, num_heads)
    end = time.time()
    print(f"{times}x supported op time cost: {end - start}s")


if __name__ == "__main__":
    test_npu_prompt_flash_attention()
    perf_flash_attention()
