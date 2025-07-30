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
import os
import subprocess
import argparse
import pandas as pd
import torch
import config
import paramiko




DATASETS = os.path.join(config.NFS_DIR, "datasets")
benchmark_csv = os.path.join(config.NFS_DIR, "benchmark.csv")
result_csv = os.path.join(config.NFS_DIR, "result.csv")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(filename)s %(lineno)d [%(levelname)s] %(message)s',
    handlers=[logging.FileHandler("test_benchmark.log"), logging.StreamHandler()])
logger = logging.getLogger(__name__)


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
    "npu_fw+bw/benchmark"]

column_left = column_names[:column_names.index("format") + 1]

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
    "is_delta_q": bool
}
    

def read_and_validate_parameters(index_to_find, csv_file_path=benchmark_csv):
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
           
    try: 
        df = pd.read_csv(csv_file_path, encoding="utf-8") 
    except UnicodeDecodeError as e:
        logger.error(f"An error occurred: {e}")
        return None
    
    df = df.loc[df["index"] == index_to_find]
    row = df[list(hstu_required_params.keys())]

    if row.empty:
        logger.info(f"row {index_to_find} is empty.")
        return None
    
    params = row.iloc[0].to_dict()

    for key, required_type in hstu_required_params.items():
        params[key] = convert_value(params[key], required_type)

    params["image_name"] = f"{index_to_find}_{df.shape_info.item()}"
    logger.info(f"{index_to_find}: {params}")
    
    return df, params


def init_result_csv_index(index_to_find):
    benchmark_df = pd.read_csv(benchmark_csv)
    if not os.path.exists(result_csv):
        df_res = pd.DataFrame(columns=column_names)
        df_res.to_csv(result_csv, index=False)
    
    df_res = pd.read_csv(result_csv)
    benchmark_row = benchmark_df.loc[benchmark_df['index'] == index_to_find, column_left]
    if benchmark_row.shape[0] > 0:
        benchmark_row = benchmark_row.iloc[0]

    if index_to_find in df_res['index'].tolist():
        return
    
    new_row = pd.DataFrame([benchmark_row], columns=column_left, index=[index_to_find])
    df_res = pd.concat([df_res, new_row])
    df_res.to_csv(result_csv, index=False)
    logger.info(f"Create index {index_to_find} in {result_csv}")    


if __name__ == "__main__":
    index_to_find = 1
    df, params = read_and_validate_parameters(index_to_find)
    init_result_csv_index(index_to_find)
    df_res = pd.read_csv(result_csv)
    logger.info(df_res)