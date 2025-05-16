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

import pkg_resources
from setuptools import find_packages, setup

PKG_NAME = "tf_npu_xla"
VERSION = "0.1.0"
DESCRIPTION = "NPU XLA for TensorFlow"
REQUIRES_PYTHON = ">=3.9.0"


def host_tf_version():
    for pkg in pkg_resources.working_set:
        if (
            "tensorflow-io" in pkg.project_name
            or "tensorflow-estimator" in pkg.project_name
        ):
            continue
        if "tensorflow" in pkg.project_name:
            return f"{pkg.project_name}=={pkg.version}"


REQUIRES = [host_tf_version()]

setup(
    name=PKG_NAME,
    version=VERSION,
    description=DESCRIPTION,
    python_requires=REQUIRES_PYTHON,
    packages=find_packages("python", exclude=("tests",)),
    install_requires=REQUIRES,
    license="Apache License 2.0",
    package_dir={"tf_npu_xla": "python/tf_npu_xla"},
    package_data={
        "tf_npu_xla": [
            "*.so",
            "tf-mlir-opt",
            "inference-mlir-opt",
            "inference-mlir-compiler",
        ]
    },
    zip_safe=False,
)
