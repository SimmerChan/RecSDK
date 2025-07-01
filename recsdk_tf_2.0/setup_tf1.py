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
import subprocess

from setuptools import find_packages, setup


def modify_version():
    default_version = "6.0.T200"
    env_version = os.getenv("VERSION")
    ver = env_version if env_version is not None else default_version

    init_file = "mxrec/__init__.py"
    with open(init_file, 'r') as file:
        lines = file.readlines()
        for idx, line in enumerate(lines):
            if "__version__ = " not in line:
                continue
            lines[idx] = f"__version__ = '{ver}'\n"
            break

    flag = os.O_WRONLY | os.O_TRUNC
    mode = stat.S_IWUSR | stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH
    with os.fdopen(os.open(init_file, flag, mode), 'w') as out:
        out.writelines(lines)
    return ver


if __name__ == "__main__":
    try:
        with open("README.md") as md_file:
            long_description = md_file.read()
    except IOError:
        long_description = ""

    version = modify_version()

    # compile so files
    script_path = os.getcwd()
    tf1_script = os.path.join(script_path, "./ci/scripts/build_tf1.sh")
    res = subprocess.run([tf1_script], shell=False)
    if res.returncode:
        raise RuntimeError("compile so files failed!")

    setup(
        name='mxrec_for_lingqu',
        version=version,
        author='HUAWEI Inc',
        description='MindX SDK Recommend for Lingqu 2.0',
        long_description=long_description,
        packages=find_packages(include=["mxrec*"]),
        # other file
        package_data={'': ['*.yml', '*.so*']},
        python_requires='>=3.7.5'
    )

    move_whl_script = os.path.join(script_path, "./ci/scripts/move_whl_file_2_pkg_dir.sh")
    res = subprocess.run([move_whl_script, "tf1"], shell=False)
    if res.returncode:
        raise RuntimeError("move tf1 whl file to pkg dir failed!")
