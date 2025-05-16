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
import tensorflow as tf
from tensorflow.python.framework import load_library as _ll

DEVICE_SO = "libnpu_device.so"
XLA_OP_SO = "libnpu_xla.so"
TF_MLIR_MAIN = "tf-mlir-opt"
MLIR_OPT = "inference-mlir-opt"
MLIR_COMPILER = "inference-mlir-compiler"
_ROOT = os.path.abspath(os.path.dirname(__file__))


# Main switch function to enable Ascend NPU XLA functionality
# After enabling, NPU will be used for XLA accelerated computation
def enable(persistent_cache: str):
    device_path = os.path.join(_ROOT, DEVICE_SO)
    xla_op_path = os.path.join(_ROOT, XLA_OP_SO)
    tf_mlir_path = os.path.join(_ROOT, TF_MLIR_MAIN)
    if not os.path.exists(device_path):
        raise FileNotFoundError("Device library %s not found" % device_path)
    if not os.path.exists(xla_op_path):
        raise FileNotFoundError("NPU_XLA library %s not found" % xla_op_path)
    if not os.path.exists(tf_mlir_path):
        raise FileNotFoundError("TF MLIR executable %s not found" % tf_mlir_path)
    if not os.path.exists(MLIR_OPT):
        raise FileNotFoundError("IG MLIR opt executable %s not found" % MLIR_OPT)
    if not os.path.exists(MLIR_COMPILER):
        raise FileNotFoundError(
            "IG MLIR compiler executable %s not found" % MLIR_COMPILER
        )
    _ll.load_pluggable_device_library(device_path)
    tf.load_op_library(xla_op_path)
    os.environ.setdefault("TF_MLIR_BIN_PATH", tf_mlir_path)
    persistent_cache_path = os.path.join(persistent_cache, "cache")
    compilation_production_path = os.path.join(
        persistent_cache, "compilation_production"
    )
    os.environ.setdefault("CACHE_PATH", persistent_cache_path)
    os.environ.setdefault("COMPILE_PRODUCT_PATH", compilation_production_path)
    os.environ.setdefault("INFERENCE_GRAPH_PATH", _ROOT)

    # Add _ROOT to LD_LIBRARY_PATH
    ld_library_path = os.environ.get("LD_LIBRARY_PATH", "")
    if _ROOT not in ld_library_path.split(os.pathsep):
        new_ld_library_path = (
            os.pathsep.join([_ROOT, ld_library_path]) if ld_library_path else _ROOT
        )
        os.environ["LD_LIBRARY_PATH"] = new_ld_library_path

    print("NPU XLA is enabled.")
