#!/bin/bash
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
mkdir -p AttentionFusionGrad/data
cp -r ../aclnn_attention_fusion_grad/input/* AttentionFusionGrad/data
cp -r ../aclnn_attention_fusion_grad/output/* AttentionFusionGrad/data
/usr/local/Ascend/ascend-toolkit/latest/tools/ascendc_tools/ascendebug kernel --backend cpu --json-file attention_fusion_grad_debug.json --repo-type customize --customize-path \
    /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/attention_fusion_grad/  --chip-version Ascend910B2  --core-type MixCore --install-path /usr/local/Ascend/ascend-toolkit/ --work-dir .