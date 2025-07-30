#!/usr/bin/env python3
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
import stat
import tensorflow as tf

MIN_SIZE = 1
MAX_FILE_SIZE = 500 * 1024 * 1024 * 1024
HDFS_FILE_PREFIX = ["viewfs://", "hdfs://"]
UNSUPPORTED_FILE_MODE_MASK = 0o022

def check_file_system_is_hdfs(file_path):
    for prefix in HDFS_FILE_PREFIX:
        if file_path.startswith(prefix):
            return True
    return False

def validate_read_file(read_file_path):
    """
    Validate file before reading，including validating soft link, file size
    :param read_file_path: the file path to be validated
    """
    # file size check
    file_stat = tf.io.gfile.stat(read_file_path)
    if not (MIN_SIZE < file_stat.length <= MAX_FILE_SIZE):
        raise ValueError(f"file size: {file_stat.length} is invalid, not in ({MIN_SIZE}, {MAX_FILE_SIZE}]")

    # file system check
    if check_file_system_is_hdfs(read_file_path):
        return
    
    # link file check
    if (os.path.abspath(read_file_path) != os.path.realpath(read_file_path)):
        raise ValueError(f"soft link or relative path: {read_file_path} should not be in the path parameter")

    stat_info = os.stat(read_file_path)
    # user group check
    process_uid = os.geteuid()
    process_gid = os.getegid()
    if not ((process_uid == stat_info.st_uid) or (process_gid == stat_info.st_gid)):
        raise ValueError(f"Invalid log file user or group, path: {read_file_path}.")

    # file mode check
    mode = stat.S_IMODE(stat_info.st_mode)
    if ((mode & UNSUPPORTED_FILE_MODE_MASK) != 0):
        raise ValueError(f"Current file:{read_file_path}, mode {oct(mode)} is unsupported")
