/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * Copyright (c) huawei Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include "bucketize.h"

#include <cstdint>
#include <vector>
#include <chrono>

namespace Embcache {

void Mod(const int64_t* lengths, int64_t lenthsNum, const int64_t* values, int64_t valuesNum, int64_t numBuckets,
         int64_t* outBucketLengths, int64_t outBucketLenthsNum, int64_t* outBucketValues, int64_t outBucketValuesNum)
{
    int64_t startOff = 0;
    int64_t index = 0;
    // [bucket][batch*num]
    std::vector<std::vector<int64_t>> allBuckets(numBuckets);
    std::vector<int64_t> bucketLengths(numBuckets * lenthsNum);
    std::vector<std::vector<int64_t>> tmpBuckets(numBuckets);

    // O(valuesNum)
    for (size_t i = 0; i < lenthsNum; ++i) {
        int64_t length = lengths[i];
        for (int64_t j = startOff; j < startOff + length && j < valuesNum; j++) {
            auto value = values[j];
            auto bucket = value % numBuckets;
            allBuckets[bucket].push_back(value);
            tmpBuckets[bucket].push_back(value);
        }
        for (size_t n = 0; n < numBuckets; ++n) {
            bucketLengths[n * lenthsNum + index] = tmpBuckets[n].size();
            tmpBuckets[n].clear();
        }
        startOff += length;
        index++;
    }
    // Memory copy to output.
    memcpy(outBucketLengths, bucketLengths.data(), bucketLengths.size() * sizeof(int64_t));
    size_t memcpyOff = 0;
    for (size_t n = 0; n < numBuckets; ++n) {
        memcpy(outBucketValues + memcpyOff, allBuckets[n].data(), allBuckets[n].size() * sizeof(int64_t));
        memcpyOff += allBuckets[n].size();
    }
}

std::tuple<at::Tensor, at::Tensor> ModBucketize(const at::Tensor& lengths, const at::Tensor& values, int64_t numBuckets)
{
    TORCH_CHECK(lengths.dtype() == torch::kInt64);
    TORCH_CHECK(values.dtype() == torch::kInt64);

    int64_t* lengthsPtr = lengths.data_ptr<int64_t>();
    int64_t lenthsNum = lengths.numel();

    int64_t* valuesPtr = values.data_ptr<int64_t>();
    int64_t valuesNum = values.numel();
    auto bucketLengths = at::empty({lengths.numel() * numBuckets}, torch::kInt64);
    auto bucketValues = at::empty({values.numel()}, torch::kInt64);
    Mod(lengthsPtr, lenthsNum, valuesPtr, valuesNum, numBuckets, bucketLengths.data_ptr<int64_t>(),
        bucketLengths.numel(), bucketValues.data_ptr<int64_t>(), bucketValues.numel());

    return std::make_tuple(bucketLengths, bucketValues);
}

}  // namespace Embcache