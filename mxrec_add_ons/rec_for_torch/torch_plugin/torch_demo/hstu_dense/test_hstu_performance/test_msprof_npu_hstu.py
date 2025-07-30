#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2025. Huawei Technologies Co.,Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import argparse
import glob
import subprocess
import pandas as pd
import os
import config

from test_read_benchmark import logger, read_and_validate_parameters, result_csv, init_result_csv_index


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Read CSV file and run a specific index benchmark')
    parser.add_argument('--index', type=int, required=True, help='index of the benchmark to run')
    args = parser.parse_args()

    init_result_csv_index(args.index)
    df_res = pd.read_csv(result_csv)
    if df_res.loc[df_res['index'] == args.index, 'npu_fw_time'].notna().all() and \
        df_res.loc[df_res['index'] == args.index, 'npu_bw_time'].notna().all():
        logger.info(f'Benchmark with index {args.index} is already done. Exit.')
        exit(0)
    
    _, params = read_and_validate_parameters(args.index)
    cmd = f'rm profnpu/ -rf ; msprof --application=\"python3 test_npu_hstu.py --index={args.index}\" --output=profnpu'
    subprocess.run(cmd.split(' '))
    
    search_dir = os.path.join(config.NFS_DIR, 'profnpu')
    csv_file = glob.glob(f'{search_dir}/PROF_*/mindstudio_profiler_output/op_stati*.csv')[0]

    logger.info(f'profile located at: {csv_file}')
    df_op_stati = pd.read_csv(csv_file)

    forward_row = df_op_stati[df_op_stati['OP Type'] == 'HstuDenseForward']
    backward_row = df_op_stati[df_op_stati['OP Type'] == 'HstuDenseBackward']
    
    df_res.loc[df_res['index'] == args.index, 'npu_fw_time'] = forward_row['Avg Time(us)'].squeeze() / 1000
    df_res.loc[df_res['index'] == args.index, 'npu_bw_time'] = backward_row['Avg Time(us)'].squeeze() / 1000

    df_res.to_csv(result_csv, index=False)
    
    logger.info(f'Forward time {df_res.loc[df_res['index'] == args.index, 'npu_fw_time']} ms')
    logger.info(f'Backward time {df_res.loc[df_res['index'] == args.index, 'npu_bw_time']} ms')
