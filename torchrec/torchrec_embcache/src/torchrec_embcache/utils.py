#!/usr/bin/env python3
# Copyright (c) Huawei Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import os
from pathlib import Path
from typing import List

_STRING_MIN_LEN = 0
_STRING_MAX_LEN = 1024
_FILE_SIZE_MAX = 1 * 1024 * 1024 * 1024 * 1024  # 1TB

_DEFAULT_BLACK_DIRS = ["/usr/bin", "/usr/bin", "/usr/sbin", "/etc", "/usr/lib", "/usr/lib64", "/usr/local"]
_DEFAULT_SENSITIVE_WORDS = ["Key", "password", "privatekey"]


def check_str(string_value: str, min_length: int, max_length: int) -> None:
    if not isinstance(string_value, str):
        raise TypeError(f"expected param type string but got {type(string_value)}")
    if len(string_value) < min_length or len(string_value) > max_length:
        raise ValueError(f"string param length is invalid, length limit:[{min_length}, {max_length}]")


def check_path(value: str, min_len: int = _STRING_MIN_LEN, max_len: int = _STRING_MAX_LEN,
               need_exist: bool = False, file_size_min: int = None, file_size_max: int = None, is_dir: bool = False,
               black_dirs: List[str] = None, sensitive_words: List[str] = None) -> None:
    check_str(value, min_len, max_len)
    if os.path.abspath(value) != os.path.realpath(value):
        raise ValueError(f"soft link or relative path can't be a path param, got:{value}")
    if need_exist and not os.path.exists(os.path.realpath(value)):
        raise ValueError(f"expected path exist, but got:{value}")

    black_dirs = black_dirs or _DEFAULT_BLACK_DIRS
    is_start_with_black_dirs = any([value.startswith(item) for item in black_dirs])
    if is_start_with_black_dirs:
        raise ValueError(f"path can't start with black dirs, but got:{value}")

    sensitive_words = sensitive_words or _DEFAULT_SENSITIVE_WORDS
    contains_sensitive_word = any([item in value for item in sensitive_words])
    if contains_sensitive_word:
        raise ValueError(f"path can't contains sensitive words, but got:{value}")

    file_exist = os.path.exists(os.path.realpath(value))
    # 检查权限
    if file_exist:
        process_uid = os.geteuid()
        process_gid = os.getegid()
        stat_info = os.stat(value)
        file_uid = stat_info.st_uid
        file_gid = stat_info.st_gid
        if not (process_uid == file_uid or process_gid == file_gid):
            raise ValueError(f"current user don't have access permission for the path:{value}")

    current_is_dir = file_exist and os.path.isdir(value)
    if is_dir and not current_is_dir:
        raise ValueError(f"expected path param is a directory, but file not exist or not a directory")

    if file_exist and not os.path.isdir(value) and (file_size_min or file_size_max):
        file_bytes = Path(value).stat().st_size
        if file_size_min and file_bytes < file_size_min or file_size_max and file_bytes > file_size_max:
            min_size = file_size_min or 0
            max_size = file_size_max or _FILE_SIZE_MAX
            raise ValueError(f"file size in byte exceeds limit:[{min_size}, {max_size}]")
