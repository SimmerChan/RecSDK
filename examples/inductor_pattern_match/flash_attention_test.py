import numpy as np
import torch
import torch_npu


def supported_op_exec(query_states1, past_key, past_value, head_dim):
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
    return attn_output1


def custom_op_exec(query, key, value, head_dim):
    scale = 1.0 / math.sqrt(head_dim)
    return torch_npu.npu_prompt_flash_attention(
        query,
        key,
        value,
        num_heads=32,
        input_layout="BNSD",
        scale_value=scale,
        pre_tokens=65535,
        next_tokens=65535,
    )


def test_npu_prompt_flash_attention(device="npu"):
    query = torch.randn(1, 32, 2048, 128, dtype=torch.float16, device=device)
    key = torch.randn(1, 32, 2048, 128, dtype=torch.float16, device=device)
    value = torch.randn(1, 32, 2048, 128, dtype=torch.float16, device=device)

    head_dim = 128

    supported_output = supported_op_exec(query, key, value, head_dim)
    custom_output = custom_op_exec(query, key, value, head_dim)
    assert np.allclose(
        supported_output.cpu().numpy(),
        custom_output.cpu().numpy(),
        rtol=1e-05,
        atol=1e-05,
    )


if __name__ == "__main__":
    test_npu_prompt_flash_attention()
