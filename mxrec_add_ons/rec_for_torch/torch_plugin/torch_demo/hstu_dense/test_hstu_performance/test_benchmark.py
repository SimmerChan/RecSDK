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

 
import argparse
import os
import subprocess
import torch
import paramiko
import pandas as pd
import config
from test_read_benchmark import benchmark_csv, result_csv, logger, DATASETS, init_result_csv_index



INDEX_STR = 'index'
GPU_FW_TIME = 'gpu_fw_time'
NPU_FW_TIME = 'npu_fw_time'
NPU_BW_TIME = 'npu_bw_time'
GPU_BW_TIME = 'gpu_bw_time'


def execute_and_process(bx):
    cmd = f"python3 test_msprof_npu_hstu.py --index={bx}"
    logger.info(f"Executing local scirpt: {cmd}")
    try:
        result = subprocess.run(
            cmd.split(' '),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        logger.info(f"stdout: {result.stdout}")
        logger.info(f"stderr: {result.stderr}")
        return True
    except Exception as e:
        logger.error(f"Failed to execute local script: {cmd}")
        logger.error(e)
        return False


def transfer_and_execute(bx):
    remote_host = config.GPU_IP
    remote_user = config.GPU_USER
    remote_password = config.GPU_PASSWORD
    remote_dir = config.RECSYS_DIR

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    cmd = f"cd {remote_dir} && source ~/.bashrc && {config.PYTHON3} test_gpu_hstu.py --index={bx}"

    try:
        client.connect(remote_host, port=22, username=remote_user, password=remote_password)
        sftp = client.open_sftp()
        files_to_transfer = ["test_gpu_hstu.py", "test_read_benchmark.py", "config.py"]
        for file in files_to_transfer:
            sftp.put(file, os.path.join(remote_dir, file))
        sftp.close()

        
        logger.info(f"Executing remote script: {cmd}")
        stdin, stdout, stderr = client.exec_command(cmd)
        exit_status = stdout.channel.recv_exit_status()
        logger.info(f"stdout: {stdout.read().decode('utf-8')}")
        logger.info(f"stderr: {stderr.read().decode('utf-8')}")
        return exit_status == 0
    except Exception as e:
        logger.error(f"Failed to execute remote script: {cmd}")
        logger.error(e)
        return False
    finally:
        logger.info("Closing connection")
        if client:
            client.close()


def compare_npu_gpu_precision(save_dir=DATASETS, device='cpu'):
    logger.info(f"Compating npu and gpu results of {save_dir}")
    try:
        data_type = torch.load(os.path.join(save_dir, "data_type.pth"), map_location=device)
        npu_out = torch.load(os.path.join(save_dir, "npu_out.pth"), map_location=device).to(dtype=data_type)
        gpu_out = torch.load(os.path.join(save_dir, "gpu_out.pth"), map_location=device).to(dtype=data_type)
        gpu_out = gpu_out.view(gpu_out.shape[0], -1)

        npu_q = torch.load(os.path.join(save_dir, "npu_q.pth"), map_location=device).to(dtype=data_type)
        gpu_q = torch.load(os.path.join(save_dir, "gpu_q.pth"), map_location=device).to(dtype=data_type)

        npu_k = torch.load(os.path.join(save_dir, "npu_k.pth"), map_location=device).to(dtype=data_type)
        gpu_k = torch.load(os.path.join(save_dir, "gpu_k.pth"), map_location=device).to(dtype=data_type)

        npu_v = torch.load(os.path.join(save_dir, "npu_v.pth"), map_location=device).to(dtype=data_type)
        gpu_v = torch.load(os.path.join(save_dir, "gpu_v.pth"), map_location=device).to(dtype=data_type)
   
    except Exception as e:
        logger.error(f"error : {e}")
        return False
        
    if data_type == torch.bfloat16:
        eps = 1e-2
    elif data_type == torch.float16:
        eps = 1e-3  
    else:
        logger.error(f"error type : {data_type}")
        return False
  
    try:
        out_close = torch.allclose(npu_out, gpu_out, eps, eps)
        out_q = torch.allclose(npu_q, gpu_q, eps, eps)
        out_k = torch.allclose(npu_k, gpu_k, eps, eps)
        out_v = torch.allclose(npu_v, gpu_v, eps, eps)
    except Exception as e:
        logger.error(f"error : {e}")
        return False

    logger.info(f"npu_out vs gpu_out: {out_close}")
    logger.info(f"npu_q vs gpu_q: {out_q}")
    logger.info(f"npu_k vs gpu_k: {out_k}")
    logger.info(f"npu_v vs gpu_v: {out_v}")

    ret = out_close and out_q and out_k and out_v
    return ret


def update_index_csv(df_res, benchmark_df, bx, precision):
    mask_res = df_res[INDEX_STR] == bx
    mask_benchmark = benchmark_df[INDEX_STR] == bx

    df_res.loc[mask_res, 'precision'] = precision
    df_res.loc[mask_res, 'npu_fw/gpu_fw'] = (
        df_res.loc[mask_res, GPU_FW_TIME] / df_res.loc[mask_res, NPU_FW_TIME]
    )
    df_res.loc[mask_res, 'npu_bw/gpu_bw'] = (
        df_res.loc[mask_res, GPU_BW_TIME] / df_res.loc[mask_res, NPU_BW_TIME]
    )
    
    df_res.loc[mask_res, 'npu_fw+bw/gpu_fw+bw'] = (
         df_res.loc[mask_res, [GPU_FW_TIME, GPU_BW_TIME]].sum(axis=1) / 
         df_res.loc[mask_res, [NPU_FW_TIME, NPU_BW_TIME]].sum(axis=1)
    )
    df_res.loc[mask_res, 'npu_fw/benchmark'] = (
        benchmark_df.loc[mask_benchmark, NPU_FW_TIME] / df_res.loc[mask_res, NPU_FW_TIME]
    )
    df_res.loc[mask_res, 'npu_bw/benchmark'] = (
      benchmark_df.loc[mask_benchmark, NPU_BW_TIME] / df_res.loc[mask_res, NPU_BW_TIME]
    )
    df_res.loc[mask_res, 'npu_fw+bw/benchmark'] = (
        benchmark_df.loc[mask_benchmark, [NPU_FW_TIME, NPU_BW_TIME]].sum(axis=1) /
        df_res.loc[mask_res, [NPU_FW_TIME, NPU_BW_TIME]].sum(axis=1)
    )


def retry_operation(operation, operation_name, bx, max_retries=2):
    for attempt in range(max_retries):
        ret = operation(bx)
        if ret:
            return True
        logger.warning(f"{operation_name} failed for benchmark {bx}, attempt {attempt + 1}/{max_retries}")
    logger.error(f"{operation_name} failed for benchmark {bx} after {max_retries} retries")
    return False


def main(index=None):
    benchmark_df = pd.read_csv(benchmark_csv)
    benchmark_df[INDEX_STR] = benchmark_df[INDEX_STR].astype(int)
    all_indices = benchmark_df[INDEX_STR].tolist()
    
    if index is not None:
        all_indices = [index]

    for bx in all_indices:
        logger.info(f"benchmark {bx} testing")
        init_result_csv_index(bx)
        df_res = pd.read_csv(result_csv)
        
        if df_res[df_res[INDEX_STR] == bx].notna().all().all():
            logger.info(f"benchmark {bx} already, pass")
            continue
        
        remote_success = retry_operation(transfer_and_execute, "Remote execution", bx)
        if not remote_success:
            continue
        
        local_success = retry_operation(execute_and_process, "Local execution", bx)
        if not local_success:
            continue
        
        df_res = pd.read_csv(result_csv)
        precision = compare_npu_gpu_precision()
        update_index_csv(df_res, benchmark_df, bx, precision)
        df_res.to_csv(result_csv, index=False)
        logger.info(f"benchmark {bx} tested")
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run benchmark tests.")
    parser.add_argument('--index', type=int, help="benchmark to run. Default run all.")
    args = parser.parse_args()
    main(args.index)