#!/usr/bin/env python3
# Copyright (c) Huawei Platforms, Inc. and affiliates.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import os
import torch

from hybrid_torchrec.modules.hash_embeddingbag import HashEmbeddingBagCollection, HashEmbeddingBagConfig, \
    HybridHashTable


torch.ops.load_library(os.path.join(os.path.dirname(__file__), "libfbgemm_npu_api.so"))