#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2025. Huawei Technologies Co.,Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


import ast
import logging
import math
import os
from typing import Dict, Optional, Tuple

import pandas as pd
import numpy as np
import torch
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib


import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(filename)s %(lineno)d [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("test_benchmark.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

matplotlib.use("Agg")
INDEX_STR = "index"
DATASETS = os.path.join(os.path.realpath(config.NFS_DIR), "datasets")
benchmark_csv = os.path.join(os.path.realpath(config.NFS_DIR), "benchmark.csv")
result_csv = os.path.join(os.path.realpath(config.NFS_DIR), "result.csv")

column_names = [
    "index",
    "shape_info",
    "total_len",
    "batch_size",
    "heads",
    "heads_rab",
    "max_seq_len_q",
    "max_seq_len_k",
    "max_context_len",
    "max_target_len",
    "target_group_size",
    "attn_dim",
    "hidden_dim",
    "alpha",
    "has_rab",
    "has_drab",
    "window_size",
    "run_benchmark",
    "dtype",
    "full_batch",
    "is_delta_q",
    "format",
    "npu_fw_time",
    "npu_bw_time",
    "gpu_fw_time",
    "gpu_bw_time",
    "precision",
    "npu_fw/gpu_fw",
    "npu_bw/gpu_bw",
    "npu_fw+bw/gpu_fw+bw",
    "npu_fw/benchmark",
    "npu_bw/benchmark",
    "npu_fw+bw/benchmark",
]

column_left = column_names[: column_names.index("format") + 1]

hstu_required_params = {
    "total_len": int,
    "batch_size": int,
    "heads": int,
    "heads_rab": int,
    "max_seq_len_q": int,
    "max_seq_len_k": int,
    "max_context_len": int,
    "max_target_len": int,
    "target_group_size": int,
    "attn_dim": int,
    "hidden_dim": int,
    "alpha": float,
    "has_rab": bool,
    "has_drab": bool,
    "window_size": tuple,
    "run_benchmark": int,
    "dtype": torch.dtype,
    "full_batch": bool,
    "is_delta_q": bool,
}

generate_params = [
    "total_len",
    "batch_size",
    "heads",
    "heads_rab",
    "max_seq_len_q",
    "max_seq_len_k",
    "max_context_len",
    "max_target_len",
    "target_group_size",
    "attn_dim",
    "hidden_dim",
    "window_size",
    "dtype",
    "full_batch",
    "has_drab",
    "is_delta_q",
]


def convert_value(value, required_type):
    if isinstance(value, str):
        if value == "torch.float16":
            return torch.float16
        elif value == "torch.bfloat16":
            return torch.bfloat16
        else:
            try:
                value = ast.literal_eval(value)
            except ValueError:
                pass
    try:
        return required_type(value)
    except (ValueError, TypeError):
        return value


def read_and_validate_parameters(index, csv_file_path=benchmark_csv):
    try:
        # 尝试读取CSV文件
        df_benchmark = pd.read_csv(csv_file_path, encoding="utf-8")

        # 检查索引是否存在
        if INDEX_STR not in df_benchmark.columns:
            logger.error("CSV文件中缺少索引列: %s", INDEX_STR)
            return None, None

        # 过滤指定索引的行
        df_benchmark = df_benchmark.loc[df_benchmark[INDEX_STR] == index]
        if df_benchmark.empty:
            logger.info("索引 %d 对应的行为空", index)
            return None, None

        # 提取所需参数
        try:
            row = df_benchmark[list(hstu_required_params.keys())]
        except KeyError as e:
            logger.error("CSV文件中缺少必要列: %s", e)
            return None, None

        # 转换为字典并验证类型
        params = row.iloc[0].to_dict()
        for key, required_type in hstu_required_params.items():
            try:
                params[key] = convert_value(params[key], required_type)
            except (ValueError, TypeError) as e:
                logger.error("参数 %s 类型转换失败: %s", key, e)
                return None, None

        logger.info("索引 %d 的参数: %s", index, params)
        return df_benchmark, params

    except Exception as e:
        logger.error("错误: %s", e)
        return None, None


def init_result_csv_index(index):
    try:
        # 读取基准CSV文件并确保索引列为整数类型
        benchmark_df = pd.read_csv(benchmark_csv)
        benchmark_df[INDEX_STR] = benchmark_df[INDEX_STR].astype(int)

        # 如果结果文件不存在则创建空文件
        if not os.path.exists(result_csv):
            pd.DataFrame(columns=column_names).to_csv(result_csv, index=False)

        # 读取结果文件
        df_res = pd.read_csv(result_csv)

        # 检查索引是否存在于基准数据中
        benchmark_row = benchmark_df.loc[benchmark_df[INDEX_STR] == index, column_left]
        if benchmark_row.empty:  # 明确处理空情况
            logger.warning(f"索引 {index} 不存在于基准文件中，跳过添加操作")
            raise ValueError(f"row {index} not in {benchmark_csv}")

        # 检查索引是否已存在于结果文件中
        if index in df_res[INDEX_STR].values:
            return

        # 添加新行并保存
        new_row = pd.DataFrame([benchmark_row.iloc[0]], columns=column_left)
        df_res = pd.concat([df_res, new_row], ignore_index=True)
        df_res.to_csv(result_csv, index=False)
        logger.info(f"成功添加索引 {index} 到结果文件 {result_csv}")

    except Exception as e:
        logger.error(f"初始化结果文件时发生错误: {str(e)}")
        raise e


def construct_mask(
    seqlen_c,
    seqlen,
    seqlen_t=0,
    target_group_size=1,
    window_size=(-1, -1),  # -1 means infinite window size
    seq_offsets=None,
    num_contexts=None,
    device=None,
):
    seqlen = seqlen_c + seqlen + seqlen_t
    bs = seq_offsets.size(0) - 1

    mask = torch.zeros((seqlen, seqlen), device=device, dtype=torch.bool)
    if window_size[0] < 0 and window_size[1] == 0:
        # causal mask
        for i in range(seqlen):
            mask[i, : i + 1] = True

        # context mask
        if seqlen_c != 0:
            mask = mask.unsqueeze(0).unsqueeze(0).repeat(bs, 1, 1, 1)
            for i in range(bs):
                target_start = (
                    num_contexts[i] + seq_offsets[i + 1] - seq_offsets[i]
                ).item()
                mask[i, 0, : num_contexts[i], :target_start] = True

        # target mask
        if seqlen_t != 0:
            mask = (
                mask.unsqueeze(0).unsqueeze(0).repeat(bs, 1, 1, 1)
                if mask.ndim == 2
                else mask
            )
            for i in range(bs):
                target_start = (
                    num_contexts[i] + seq_offsets[i + 1] - seq_offsets[i]
                ).item()
                # target group mask
                if target_group_size > 1:
                    group_num = math.ceil((seqlen - target_start) / target_group_size)
                    for j in range(group_num):
                        for k in range(
                            min(
                                target_group_size,
                                seqlen - target_start - j * target_group_size,
                            )
                        ):
                            mask[
                                i,
                                0,
                                target_start + j * target_group_size + k,
                                target_start : target_start + j * target_group_size,
                            ] = False
                else:
                    for j in range(target_start, seqlen):
                        mask[i, 0, j, target_start:j] = False

    # local mask
    else:
        window_size_0 = window_size[0] if window_size[0] > 0 else seqlen
        window_size_1 = window_size[1] if window_size[1] > 0 else seqlen
        for i in range(seqlen):
            mask[i, max(0, i - window_size_0) : min(seqlen, i + window_size_1 + 1)] = (
                True
            )
    return mask


def gen_seq(length, max_value, total_sum):
    if max_value * length < total_sum:
        raise ValueError("total_sum error %d", total_sum)
    logger.info(f"gen_seq with total_sum {total_sum}")
    if length == 1:
        return np.array([total_sum])
    if length == 2:
        return np.array([max_value, total_sum - max_value])
    remaining_sum = total_sum - max_value
    mean_value = remaining_sum // (length - 2)
    min_val = remaining_sum - mean_value * (length - 2)
    sequence = [mean_value] * (length - 2)
    if min_val == 0 and sequence[-1] > 1:
        min_val += 1
        sequence[-1] -= 1
    sequence.extend([max_value, min_val])
    return np.array(sequence)


def adjust_ratio(total_sum, max_context_len, max_seq_len_k, max_target_len):
    # 初始化结果变量
    total_content, total_k, total_target = 0, 0, 0

    # 检查分母是否为0
    denominator = max_context_len + max_seq_len_k + max_target_len
    if denominator == 0:
        return 0, 0, 0  # 所有比例项为0时直接返回0

    # 检查每个分子是否为0
    if max_context_len == 0:
        total_content = 0
    if max_seq_len_k == 0:
        total_k = 0
    if max_target_len == 0:
        total_target = 0

    # 计算剩余需要分配的总和
    remaining_sum = total_sum - (total_content + total_k + total_target)
    if remaining_sum == 0:
        return total_content, total_k, total_target  # 无需分配剩余值

    # 计算剩余项的有效分母（排除分子为0的项）
    valid_denominator = 0
    if max_context_len > 0:
        valid_denominator += max_context_len
    if max_seq_len_k > 0:
        valid_denominator += max_seq_len_k
    if max_target_len > 0:
        valid_denominator += max_target_len

    # 按剩余比例分配
    if max_context_len > 0:
        total_content += int(round(remaining_sum * max_context_len / valid_denominator))
    if max_seq_len_k > 0:
        total_k += int(round(remaining_sum * max_seq_len_k / valid_denominator))
    if max_target_len > 0:
        total_target += int(round(remaining_sum * max_target_len / valid_denominator))

    # 处理四舍五入导致的误差（强制修正总和）
    diff = total_sum - (total_content + total_k + total_target)
    if diff != 0:
        total_target += diff  # 默认将差值调整到target（可根据需求修改）

    return total_k, total_content, total_target


def generate_input(
    total_len: int,
    batch_size: int,
    heads: int,
    heads_rab: Optional[int],
    max_seq_len_q: int,
    max_seq_len_k: int,
    max_context_len: int,
    max_target_len: int,
    target_group_size: int,
    attn_dim: int,
    hidden_dim: int,
    window_size: Tuple[int, int],
    dtype: torch.dtype,
    full_batch: bool,
    has_drab: bool,
    is_delta_q: bool,
):
    device_str = "cpu"
    has_context = max_context_len > 0
    has_target = max_target_len > 0
    target_group_size > 1
    logger.info(f"generate with total_len {total_len}")
    total_k, total_content, total_target = adjust_ratio(
        total_len, max_context_len, max_seq_len_k, max_target_len
    )
    lengths_k = (
        torch.from_numpy(gen_seq(batch_size, max_seq_len_k, total_k))
        .to(device_str)
        .int()
    )
    num_contexts = (
        torch.from_numpy(gen_seq(batch_size, max_context_len, total_content))
        .to(device_str)
        .int()
    )
    num_targets = (
        torch.from_numpy(gen_seq(batch_size, max_target_len, total_target))
        .to(device_str)
        .int()
    )

    # Generate lengths for context

    seq_offsets_c = torch.zeros(
        (batch_size + 1,), dtype=torch.int32, device=torch.device(device_str)
    )
    seq_offsets_c[1:] = torch.cumsum(num_contexts, dim=0)

    # Generate lengths for historial qkv

    seq_offsets_k = torch.zeros(
        (batch_size + 1,), dtype=torch.int32, device=torch.device(device_str)
    )
    seq_offsets_k[1:] = torch.cumsum(lengths_k, dim=0)

    # Generate lengths for target qkv
    seq_offsets_t = torch.zeros(
        (batch_size + 1,), dtype=torch.int32, device=torch.device(device_str)
    )
    seq_offsets_t[1:] = torch.cumsum(num_targets, dim=0)

    # Generate lengths for delta q
    if is_delta_q:
        if full_batch:
            lengths_q = (
                torch.ones(
                    (batch_size,), device=torch.device(device_str), dtype=torch.int32
                )
                * max_seq_len_q
            )
        else:
            # lengths_q[i] is an integer between 1 and min(max_seq_len_q, lengths_k[i])
            lengths_q = torch.zeros(
                (batch_size,), device=torch.device(device_str), dtype=torch.int32
            )
            for i in range(batch_size):
                lengths_q[i] = torch.randint(
                    1,
                    min(max_seq_len_q, lengths_k[i]) + 1,
                    size=(1,),
                    device=torch.device(device_str),
                )
        seq_offsets_q = torch.zeros(
            (batch_size + 1,), dtype=torch.int32, device=torch.device(device_str)
        )
        seq_offsets_q[1:] = torch.cumsum(lengths_q, dim=0)
    else:
        seq_offsets_q = seq_offsets_k

    # Lengths for whole q, kv
    seq_offsets_q_wt = torch.zeros(
        (batch_size + 1,), dtype=torch.int32, device=torch.device(device_str)
    )
    seq_offsets_q_wt = seq_offsets_c + seq_offsets_q + seq_offsets_t
    seq_offsets_k_wt = torch.zeros(
        (batch_size + 1,), dtype=torch.int32, device=torch.device(device_str)
    )
    seq_offsets_k_wt = seq_offsets_c + seq_offsets_k + seq_offsets_t

    l_q = int(seq_offsets_q_wt[-1].item())
    l_k = int(seq_offsets_k_wt[-1].item())
    if dtype == torch.float8_e4m3fn:
        dtype_init = torch.float16
    else:
        dtype_init = dtype

    # Generate q, k, v for history + target
    q = (
        torch.empty(
            (l_q, heads, attn_dim), dtype=dtype_init, device=torch.device(device_str)
        )
        .uniform_(-1, 1)
        .requires_grad_()
    ).to(dtype)
    k = (
        torch.empty(
            (l_k, heads, attn_dim), dtype=dtype_init, device=torch.device(device_str)
        )
        .uniform_(-1, 1)
        .requires_grad_()
    ).to(dtype)
    v = (
        torch.empty(
            (l_k, heads, hidden_dim), dtype=dtype_init, device=torch.device(device_str)
        )
        .uniform_(-1, 1)
        .requires_grad_()
    ).to(dtype)
    rab = None
    if has_drab:
        rab = torch.empty(
            (
                batch_size,
                heads if heads_rab is None else heads_rab,
                max_context_len + max_seq_len_k + max_target_len,
                max_context_len + max_seq_len_k + max_target_len,
            ),
            dtype=dtype_init,
            device=torch.device(device_str),
        ).uniform_(-1, 1)
        rab = rab.requires_grad_()

    if window_size[0] == -1 and window_size[1] == -1:
        attn_mask = None
    else:
        attn_mask = (
            construct_mask(
                seqlen_c=max_context_len,
                seqlen=max_seq_len_k,
                seqlen_t=max_target_len,
                target_group_size=target_group_size,
                window_size=window_size,
                num_contexts=num_contexts,
                seq_offsets=seq_offsets_k,
            )
            .cpu()
            .to(torch.float32)
        )
    grad = torch.rand_like(v)
    params = {
        "l_q": l_q,
        "l_k": l_k,
        "num_contexts": num_contexts if has_context else None,
        "seq_offsets_q_wt": seq_offsets_q_wt,
        "seq_offsets_k_wt": seq_offsets_k_wt,
        "num_targets": num_targets if has_target else None,
        "q": q,
        "k": k,
        "v": v,
        "rab": rab,
        "attn_mask": attn_mask,
        "grad": grad,
    }
    return params


def save_mask(matrix, title="matrix"):
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "CustomMap", [(1, 1, 1), (0.8, 0.902, 0.8)], N=256
    )

    max_pixels = 65536
    dpi = 100

    width_inches = min(12, max(6, matrix.shape[1] * 0.05))
    height_inches = min(12, max(6, matrix.shape[0] * 0.05))

    width_pixels = width_inches * dpi
    height_pixels = height_inches * dpi

    if width_pixels > max_pixels or height_pixels > max_pixels:
        scale_factor = min(max_pixels / width_pixels, max_pixels / height_pixels)
        width_inches *= scale_factor
        height_inches *= scale_factor

    fig, ax = plt.subplots(figsize=(width_inches, height_inches))

    img = ax.imshow(
        matrix, cmap=cmap, origin="lower", vmin=0, vmax=1, interpolation="nearest"
    )

    ax.xaxis.set_ticks_position("top")
    ax.yaxis.set_ticks_position("left")
    ax.invert_yaxis()

    plt.colorbar(img)
    plt.title(title)
    plt.xlabel("seq_len")
    plt.ylabel("seq_len")
    plt.savefig(f"{title}.png", bbox_inches="tight", dpi=dpi)
    plt.close(fig)


PARAM_META = {
    # Tensor parameters (name: (type, required, description))
    "l_q": ("Tensor", True, "Query sequence length"),
    "l_k": ("Tensor", True, "Key sequence length"),
    "num_contexts": ("Tensor", True, "Number of contexts"),
    "seq_offsets_q_wt": ("Tensor", True, "Query sequence weight offsets"),
    "seq_offsets_k_wt": ("Tensor", True, "Key sequence weight offsets"),
    "num_targets": ("Tensor", True, "Number of targets"),
    "q": ("Tensor", True, "Query tensor"),
    "k": ("Tensor", True, "Key tensor"),
    "v": ("Tensor", True, "Value tensor"),
    "rab": ("Tensor", True, "Relative attention bias"),
    "attn_mask": ("Tensor", True, "Attention mask"),
    "grad": ("Tensor", True, "Gradient tensor"),
    # Configuration parameters
    "dtype": ("Config", True, "Data type"),
    "max_context_len": ("Config", True, "Max context length"),
    "max_seq_len_q": ("Config", True, "Max query sequence length"),
    "max_target_len": ("Config", True, "Max target length"),
    "alpha": ("Config", True, "silu scale tensor"),
    # Optional parameters
    "image_name": ("Optional", False, "Attention matrix image filename"),
}


def _get_save_paths(save_dir: str) -> Dict[str, str]:
    """Generate parameter save paths dictionary (Windows path safe)"""
    return {
        name: os.path.normpath(os.path.join(save_dir, f"{name}.pth"))
        for name in PARAM_META
        if not name.startswith("_")
    }


def save_params(save_dir=DATASETS, **kwargs):
    """
    Save all parameters to specified directory (Windows path safe)
    Usage: save_params(r"c:\zengxiong\saved", l_q=tensor1, attn_mask=matrix, image_name="mask.png")
    """
    # Normalize path
    save_dir = os.path.normpath(save_dir)
    os.makedirs(save_dir, exist_ok=True)

    # Validate required parameters
    missing = [
        name
        for name, (_, required, _) in PARAM_META.items()
        if required and name not in kwargs
    ]
    if missing:
        logger.error(f"Missing required parameters: {missing}")
        raise ValueError(f"Required parameters missing: {', '.join(missing)}")

    # Get all save paths
    paths = _get_save_paths(save_dir)

    # Save parameters with logging
    logger.info("\n" + "=" * 50)
    logger.info(f"[SAVE] Target directory: {save_dir}")
    for name, path in paths.items():
        if name in kwargs:
            torch.save(kwargs[name], path)
            param = kwargs[name]
            log_msg = f"{name.ljust(15)}: "
            if torch.is_tensor(param):
                log_msg += f"shape={str(param.shape).ljust(18)} | dtype={param.dtype}"
            else:
                log_msg += str(param)
            logger.info(log_msg + f" → {path}")

    # Save attention matrix image if provided
    if "attn_mask" in kwargs and "image_name" in kwargs:
        mask = kwargs["attn_mask"]
        if mask.dim() == 4:
            save_matrix = mask[0, 0].cpu()
        elif mask.dim() == 3:
            save_matrix = mask[0].cpu()
        else:
            save_matrix = mask.cpu()

        img_path = os.path.join(save_dir, kwargs["image_name"])
        save_mask(save_matrix, img_path)

    # Create completion flag
    flag_path = os.path.join(save_dir, "complete.flag")
    torch.save(torch.tensor(1), flag_path)
    logger.info("=" * 50 + "\n[SUCCESS] All parameters saved\n" + "=" * 50)


def load_params(save_dir=DATASETS, device="cpu") -> Dict[str, object]:
    """
    Load all parameters from directory (Windows path safe)
    Returns: Parameter dictionary {'l_q': tensor1, 'attn_mask': tensor2, ...}
    """
    save_dir = os.path.normpath(save_dir)
    paths = _get_save_paths(save_dir)

    # Validate directory integrity
    if not os.path.exists(paths["l_q"]):
        logger.error(f"Invalid parameter directory: {save_dir}")
        raise FileNotFoundError(f"Invalid parameter directory: {save_dir}")

    # Load parameters with logging
    logger.info("\n" + "=" * 50)
    logger.info(f"[LOAD] Source directory: {save_dir}")
    params = {}
    for name, path in paths.items():
        if os.path.exists(path):
            params[name] = torch.load(path, map_location=device)
            log_msg = f"{name.ljust(15)}: "
            if torch.is_tensor(params[name]):
                log_msg += f"shape={str(params[name].shape).ljust(18)} | dtype={params[name].dtype}"
            else:
                log_msg += str(params[name])
            logger.info(log_msg + f" ← {path}")

    logger.info("=" * 50 + "\n[SUCCESS] All parameters loaded\n" + "=" * 50)
    return params


def create_and_save_params(bx):
    df, bparams = read_and_validate_parameters(bx)
    bparams["image_name"] = f"{bx}_{df.shape_info.item()}"
    init_result_csv_index(bx)
    df_resg = pd.read_csv(result_csv)
    logger.info(df_resg)
    gene_param_dict = {k: bparams[k] for k in generate_params}
    iparams = generate_input(**gene_param_dict)
    iparams["image_name"] = bparams["image_name"]

    extra_params = [
        "dtype",
        "max_context_len",
        "max_seq_len_q",
        "max_target_len",
        "alpha",
    ]
    for param_name in extra_params:
        if param_name not in iparams:
            iparams[param_name] = bparams[param_name]

    save_params(**iparams)


if __name__ == "__main__":
    benchmark_df = pd.read_csv(benchmark_csv)
    benchmark_df[INDEX_STR] = benchmark_df[INDEX_STR].astype(int)
    all_indices = benchmark_df[INDEX_STR].tolist()

    for bx in all_indices:
        create_and_save_params(bx)
        # Load example
        loaded = load_params()
