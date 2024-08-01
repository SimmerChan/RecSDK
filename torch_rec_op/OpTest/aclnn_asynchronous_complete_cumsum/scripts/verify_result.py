# coding: UTF-8
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

import numpy as np

LOSS = 1e-3
MINIMUM = 10e-10

def verify_result(real_result_file, golden_file):
    real_result = np.fromfile(real_result_file, dtype=np.int64)
    golden = np.fromfile(golden_file, dtype=np.int64)
    print("real result, example-> ", real_result_file, real_result[:129])
    print("golden result, example-> ", golden_file, golden[:129])

    result = np.abs(real_result - golden)
    deno = np.maximum(np.abs(real_result), np.abs(golden))
    result_atol = np.less_equal(result, loss)
    result_rtol = np.less_equal(result / np.add(deno, minimum), loss)
    if not result_rtol.all() and not result_atol.all():
        if np.sum(result_rtol == False) > real_result.size * loss and np.sum(result_atol == False) > real_result.size * loss:
            print("[ERROR] result error")
            return False
    print("test pass")
    return True

if __name__ == '__main__':
    print("=============================x============")
    verify_result("./output/y.bin", "./output/golden.bin")