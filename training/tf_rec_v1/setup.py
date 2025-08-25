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

import os
import argparse
import sys
from setuptools import setup, find_packages
from setuptools.command.install import install
from packaging.version import Version
import shutil
import subprocess

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default="7.2.RC1")
    parser.add_argument("--discription", default="")

    # 分离 setuptools 参数和自定义参数
    args, unknown = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + unknown  # 将剩余参数传回 setuptools
    return args

args = parse_args()

if os.path.exists("mx_rec"):
    shutil.rmtree("mx_rec")
shutil.copytree("python", "mx_rec")

common_version = Version(args.version)

class PostInstallCommand(install):
    def run(self):
        install.run(self)
        install_dir = self.install_lib
        whl_path = os.path.join(install_dir, f"../../../../common/dist/rec_sdk_common-{common_version}-py3-none-any.whl")
        if not os.path.exists(whl_path):
            raise FileNotFoundError(f"not find：{whl_path}")

        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", whl_path],
                stdout=subprocess.DEVNULL,  # 隐藏输出（可选）
                stderr=subprocess.STDOUT
            )
            print(f"success install ：{whl_path}")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"install fail：{e}")

setup(
    name='mx_rec',
    version=args.version,
    author='HUAWEI Inc',
    description='MindSDK Recommend',
    long_description=args.discription,
    # include mx_rec
    packages=find_packages(
        where=".",
        include=["mx_rec*"]
    ),
    # other file
    package_data={'': ['tools/*', 'tools/*/*', '*.yml', '*.sh',
                        '*.so*', f"../../../../common/dist/rec_sdk_common-{common_version}-py3-none-any.whl"]},
    cmdclass={'install':PostInstallCommand},
    # dependency
    python_requires='>=3.7.5'
)
shutil.rmtree("mx_rec")