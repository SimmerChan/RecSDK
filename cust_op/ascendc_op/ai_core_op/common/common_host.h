/* Copyright 2025. Huawei Technologies Co.,Ltd. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
        limitations under the License.
==============================================================================*/
#ifndef COMMON_HOST_H
#define COMMON_HOST_H

#include <cstdint>

int64_t GetBatchSizeFromJaggedOffset(const int64_t *seqOffsetData, int32_t seqOffsetLens)
{
    if (seqOffsetData == nullptr || offset_size <= 0) {
        return 0;
    }
    
    // 二分法找出有效batch
    int64_t maxValue = seqOffsetData[seqOffsetLens - 1];
    int32_t left = 0;
    int32_t right = seqOffsetLens - 1;
    int32_t firstMaxIdx = seqOffsetLens - 1;
    while (left <= right) {
        int32_t mid = left + (right - left) / 2;
        if (seqOffsetData[mid] == maxValue) {
            firstMaxIdx = mid;
            right = mid - 1;
        } else if (seqOffsetData[mid] < maxValue) {
            left = mid + 1;
        }
    }

    int64_t batchSize = static_cast<int64_t>(firstMaxIdx);
    return batchSize;
}

#endif // COMMON_HOST_H