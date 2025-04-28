# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2024. Huawei Technologies Co.,Ltd. All rights reserved.
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

import os
import sys
from enum import Enum

import logging
import glob
import sqlite3
import torch
import numpy as np
import pandas as pd

CONST_16 = 16
RECORD_COUNT = 10
DATA_RANGE = (-1.0, 1.0)
WORKSPACE = os.getcwd()
WORKSPACE_PROF_LIST_FILE = os.path.join(WORKSPACE, 'log', f'prof_list.csv')
WORKSPACE_RESULT_FILE = os.path.join(WORKSPACE, 'log', f'res.csv')

os.environ["WORKSPACE"] = WORKSPACE
os.environ["ASCEND_GLOBAL_LOG_LEVEL"] = "3"
os.environ["ASCEND_SLOG_PRINT_TO_STDOUT"] = "0"


class CubeFormat(Enum):
    ND = 0
    NZ = 1
    ZN = 2
    ZZ = 3
    NN = 4
    VECTOR = 5

    def __repr__(self) -> str:
        return self.__name__


class OpParam:

    CSV_HEADER_B = "B"
    CSV_HEADER_M = "M"
    CSV_HEADER_K = "K"
    CSV_HEADER_N = "N"
    CSV_HEADER_TRANSA = "Transpose A"
    CSV_HEADER_TRANSB = "Transpose B"
    CSV_HEADER_FORMATA = "Data Format A"
    CSV_HEADER_FORMATB = "Data Format B"
    
    def __init__(self, b=0, m=0, k=0, n=0,
                 trans_a=False, trans_b=False, en_bias=False,
                 en_scale=False, en_residual=False,
                 format_a=CubeFormat.ND, format_b=CubeFormat.ND,
                 format_c=CubeFormat.ND) -> None:
        self.b = b
        self.m = m
        self.k = k
        self.n = n
        self.trans_a = trans_a
        self.trans_b = trans_b
        self.en_bias = en_bias
        self.en_scale = en_scale
        self.en_residual = en_residual
        self.format_a = format_a
        self.format_b = format_b
        self.format_c = format_c
    
    def __str__(self) -> str:
        return f"Shape: ({self.b}, {self.m}, {self.k}, {self.n}) \n" + \
               f"Transpose: A {self.trans_a}, B {self.trans_b} \n" + \
               f"(De)Quant: Bias {self.en_bias}, Scale {self.en_scale}, Residual {self.en_residual} \n" + \
               f"CubeFormat: format_a {self.format_a}, format_b {self.format_b}, format_c {self.format_c}"


def read_op_param(testcase):
    op_param = OpParam()
    op_param.b = int(testcase[OpParam.CSV_HEADER_B])
    op_param.m = int(testcase[OpParam.CSV_HEADER_M])
    op_param.k = int(testcase[OpParam.CSV_HEADER_K])
    op_param.n = int(testcase[OpParam.CSV_HEADER_N])
    op_param.trans_a = testcase[OpParam.CSV_HEADER_TRANSA]
    op_param.trans_b = testcase[OpParam.CSV_HEADER_TRANSB]
    op_param.format_a = testcase[OpParam.CSV_HEADER_FORMATA]
    op_param.format_b = testcase[OpParam.CSV_HEADER_FORMATB]
    return op_param


def check_and_add_outputpath_header(output_path):
    if not os.path.exists(output_path):
        flags = os.O_WRONLY  
        modes = stat.S_IWUSR | stat.S_IRUSR 
        with os.fdopen(os.open(output_path, flags, modes), 'w') as f:
            f.write("B,M,K,N,"
                    "task_duration,aic_time(us),aic_cube_time(us),"
                    "aic_cube_ratio,"
                    "aic_mte1_ratio,aic_mte2_ratio,aic_mte3_ratio,"
                    "aic_fixpipe_ratio"
                    "\n")


def write_profop_data(res_path: str, op_param: OpParam) -> None:

    task_duration = 0
    op_basic_info_csv = os.path.join(res_path, "OpBasicInfo.csv")
    if not os.path.exists(op_basic_info_csv):
        logging.info("[debuf] check file: %s", op_basic_info_csv)
        logging.info("Can not find OpBasicInfo.csv in %s", res_path)
        raise Exception(f"Can not find OpBasicInfo.csv in {res_path}")
    else:
        data = pd.read_csv(op_basic_info_csv)
        for _, row in data.iterrows():
            task_duration = row['Task Duration(us)']
            break

    aic_time_us = 0
    aic_cube_time_us = 0
    aic_cube_ratio = 0
    aic_mte1_ratio = 0
    aic_mte2_ratio = 0
    aic_mte3_ratio = 0
    aic_fixpipe_ratio = 0
    pipe_util_csv = os.path.join(res_path, "PipeUtilization.csv")
    if not os.path.exists(pipe_util_csv):
        logging.info("Can not find PipeUtilization.csv in %s", res_path)
        raise Exception(f"Can not find PipeUtilization.csv in {res_path}")
    else:
        data = pd.read_csv(pipe_util_csv)
        for _, row in data.iterrows():
            aic_time_us = row['aic_time(us)']
            aic_cube_time_us = row['aic_cube_time(us)']
            aic_cube_ratio = row['aic_cube_ratio']
            aic_mte1_ratio = row['aic_mte1_ratio']
            aic_mte2_ratio = row['aic_mte2_ratio']
            aic_mte3_ratio = row['aic_mte3_ratio']
            aic_fixpipe_ratio = row['aic_fixpipe_ratio']
            break

    output_path = WORKSPACE_RESULT_FILE
    check_and_add_outputpath_header(output_path)

    flags = os.O_WRONLY  
    modes = stat.S_IWUSR | stat.S_IRUSR 
    with os.fdopen(os.open(output_path, flags, modes), 'w') as f:
        f.write(f"{op_param.b},{op_param.m},{op_param.k},{op_param.n},"
                f"{task_duration},{aic_time_us},{aic_cube_time_us},"
                f"{aic_cube_ratio},"
                f"{aic_mte1_ratio},{aic_mte2_ratio},{aic_mte3_ratio},"
                f"{aic_fixpipe_ratio}"
                "\n")


def performance_test(csv: str = "test_case.csv", device_id: int = 0):
    data = pd.read_csv(WORKSPACE + "/csv/" + csv, delimiter=",")
    for _, testcase in data.iterrows():
        op_param = read_op_param(testcase)
        cmd = f"./test_aclnn_msprofop {op_param.m} {op_param.k} {op_param.n} {op_param.trans_a} " \
              f" {op_param.trans_b} {op_param.format_a} {op_param.format_b} {device_id}"
        os.system(f"msprof op --launch-count={5} --launch-skip-before-match=1 --application=\"{cmd}\" "\
                  f" --output={WORKSPACE}/prof | tee {WORKSPACE}/log/output.log")
        with open(f"{WORKSPACE}/log/output.log", 'r') as f:
            lines = f.readlines()  # 读取所有行
            prof_res_path = lines[-3].split(" ")[-1].replace('\n', '')
            logging.info("lines[-3]", lines[-3])

        output_path = WORKSPACE_PROF_LIST_FILE
        if not os.path.exists(output_path):
            logging.info("Create file: %s", output_path)
            flags = os.O_WRONLY  
            modes = stat.S_IWUSR | stat.S_IRUSR 
            with os.fdopen(os.open(output_path, flags, modes), 'w') as f:
                f.write("B,M,K,N,prof_res_path\n")    

        flags = os.O_WRONLY  
        modes = stat.S_IWUSR | stat.S_IRUSR 
        with os.fdopen(os.open(output_path, flags, modes), 'w') as f:
            f.write(f"{op_param.b},{op_param.m},{op_param.k},{op_param.n},"
                    f"{prof_res_path}\n")


def clean_space() :
    if os.path.exists(WORKSPACE + "/prof"):
        os.system("rm -rf {}/prof/*".format(WORKSPACE))
    if os.path.exists(WORKSPACE + "/log"):
        os.system("rm -f {}/log/*".format(WORKSPACE))


def gen_test_data(csv: str = "test_case.csv", gen_algo_type: int = 0):
    # 定义列名
    columns = [
                OpParam.CSV_HEADER_B,
                OpParam.CSV_HEADER_M,
                OpParam.CSV_HEADER_K,
                OpParam.CSV_HEADER_N,
                OpParam.CSV_HEADER_TRANSA,
                OpParam.CSV_HEADER_TRANSB,
                OpParam.CSV_HEADER_FORMATA,
                OpParam.CSV_HEADER_FORMATB
    ]

    # 创建一个空的DataFrame，用于存储数据
    df = pd.DataFrame(columns=columns)

    if gen_algo_type == 0:
        data = []
        n = 300
        # 逐行添加数据
        for i in range(n):
            m = i * 32 + 32
            k = i * 32 + 32
            n = i * 32 + 32
            data.append({OpParam.CSV_HEADER_B: 1,
                         OpParam.CSV_HEADER_M: m,
                         OpParam.CSV_HEADER_K: k,
                         OpParam.CSV_HEADER_N: n,
                         OpParam.CSV_HEADER_TRANSA: 1,
                         OpParam.CSV_HEADER_TRANSB: 1,
                         OpParam.CSV_HEADER_FORMATA: 0,
                         OpParam.CSV_HEADER_FORMATB: 0})
            
        logging.info("Append data. size: %d", len(data))
        new_df = pd.DataFrame(data)
        df = pd.concat([df, new_df], ignore_index=True)  # 忽略索引进行合并

        # 将DataFrame写入CSV文件
        df.to_csv(WORKSPACE + "/csv/" + csv, sep=",", index=False)
    else:
        logging.info("Not support gen_algo_type: %s", gen_algo_type)
        raise Exception(f"Not support gen_algo_type: {gen_algo_type}")


def analyze_result():
    prof_list_file = WORKSPACE_PROF_LIST_FILE
    if not os.path.exists(prof_list_file):
        logging.info("prof file not exist. %s", prof_list_file)
        raise Exception(f"prof file not exist. {prof_list_file}")
    data = pd.read_csv(prof_list_file)
    for _, row in data.iterrows():
        op_param = OpParam()
        op_param.b = row[OpParam.CSV_HEADER_B]
        op_param.m = row[OpParam.CSV_HEADER_M]
        op_param.k = row[OpParam.CSV_HEADER_K]
        op_param.m = row[OpParam.CSV_HEADER_N]
        prof_res_path = row["prof_res_path"].strip()
        write_profop_data(prof_res_path, op_param)

if __name__ == "__main__":
    op = str(sys.argv[1])
    logging.info("WORKSPACE: %s, OP: %s", WORKSPACE, op)
    if op == "clean":
        clean_space()
    elif op == "gen_test_data":
        input_output_csv_file = sys.argv[2]
        input_gen_algo_type = int(sys.argv[3])
        gen_test_data(input_output_csv_file, input_gen_algo_type)
    elif op == "prof":
        os.makedirs(WORKSPACE + "/log", exist_ok=True)
        os.makedirs(WORKSPACE + "/prof", exist_ok=True)
        input_csv_file = sys.argv[2]
        input_device_id = int(sys.argv[3])
        performance_test(input_csv_file, input_device_id)
    elif op == "analyze_result":
        analyze_result()
    else:
        logging.info("Unknow op, OP: [%s]", op)
