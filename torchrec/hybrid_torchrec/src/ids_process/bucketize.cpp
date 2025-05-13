/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * Copyright (c) huawei Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <ATen/ops/empty_like.h>
#include <c10/util/Exception.h>
#include <c10/util/flat_hash_map.h>

#include <cstdint>
#include <vector>

#include "torch/torch.h"
#include "bucketize.h"

namespace hybrid {

// 计算前缀和，preSum数组长度应为length+1
template <typename T>
void PrefixSum(const int length, const T* array, T* preSum)
{
    preSum[0] = 0;
    for (const auto i : c10::irange(length)) {
        preSum[i + 1] = array[i] + preSum[i];
    }
}
template <bool Sequence, typename OffsetT, typename IndexT, bool DoUnique>
void ComputeNewLengths(const OffsetT* offsetsData, const IndexT* indicesData, OffsetT* newLengthsData,
                       int32_t numFeatures, int32_t batchSize, int64_t mySize, int64_t lengthsSize)
{
    for (const auto featureIdx : c10::irange(numFeatures)) {
        const auto blockSize = mySize;
        for (const auto batchIdx : c10::irange(batchSize)) {
            const auto linearIndex = featureIdx * batchSize + batchIdx;
            const OffsetT start = offsetsData[linearIndex];
            const OffsetT end = offsetsData[linearIndex + 1];
            for (const auto i : c10::irange(start, end)) {
                const IndexT idx = indicesData[i];
                const IndexT bucket = idx % blockSize;
                newLengthsData[bucket * lengthsSize + linearIndex]++;
            }
        }
    }
}

template <bool Sequence, typename OffsetT, typename IndexT, bool DoUnique>
void FillNewIndices(const OffsetT* offsetsData, const IndexT* indicesData, OffsetT* newOffsetsData,
                    IndexT* newIndicesData, IndexT* unbucketizePermuteData, int32_t numFeatures, int32_t batchSize,
                    int64_t mySize, int64_t lengthsSize)
{
    for (const auto featureIdx : c10::irange(numFeatures)) {
        const auto blockSize = mySize;
        for (const auto batchIdx : c10::irange(batchSize)) {
            const auto linearIndex = featureIdx * batchSize + batchIdx;
            const OffsetT start = offsetsData[linearIndex];
            const OffsetT end = offsetsData[linearIndex + 1];
            for (const auto i : c10::irange(start, end)) {
                const IndexT idx = indicesData[i];
                const IndexT bucket = idx % blockSize;
                const IndexT pos = newOffsetsData[bucket * lengthsSize + linearIndex];
                newIndicesData[pos] = idx;
                if constexpr (Sequence) {
                    unbucketizePermuteData[i] = pos;
                }
                newOffsetsData[bucket * lengthsSize + linearIndex]++;
            }
        }
    }
}

template <typename OffsetT, typename IndexT>
void Deduplicate(OffsetT* newLengthsData, const OffsetT* newOffsetsData, const OffsetT* offsetsData,
                 const IndexT* indicesData, IndexT* newIndicesData, IndexT* unbucketizePermuteData, int32_t numFeatures,
                 int32_t batchSize, int64_t mySize)
{
    int32_t uniqueOffset = 0;
    OffsetT curOffset = 0;
    std::vector<ska::flat_hash_map<IndexT, int32_t>> uniqueMaps(numFeatures * mySize);
    IndexT lastOffset = 0;
    for (const auto featureBucketIdx : c10::irange(numFeatures * mySize)) {
        auto& uniqueMap = uniqueMaps[featureBucketIdx];
        for (const auto batchIdx : c10::irange(batchSize)) {
            const auto linearIndex = featureBucketIdx * batchSize + batchIdx;
            const OffsetT start = lastOffset;
            const OffsetT end = newOffsetsData[linearIndex];
            lastOffset = end;
            for (const auto i : c10::irange(start, end)) {
                const IndexT idx = newIndicesData[i];
                auto it = uniqueMap.find(idx);
                if (it == uniqueMap.end()) {
                    uniqueMap.emplace(idx, uniqueOffset);
                    newIndicesData[uniqueOffset] = idx;
                    uniqueOffset++;
                } else {
                    newLengthsData[linearIndex]--;
                }
            }
        }
    }

    for (const auto featureIdx : c10::irange(numFeatures)) {
        const auto blockSize = mySize;
        for (const auto batchIdx : c10::irange(batchSize)) {
            const auto linearIndex = featureIdx * batchSize + batchIdx;
            const OffsetT rowStart = offsetsData[linearIndex];
            const OffsetT rowEnd = offsetsData[linearIndex + 1];
            for (const auto i : c10::irange(rowStart, rowEnd)) {
                const IndexT idx = indicesData[i];
                const IndexT bucket = idx % blockSize;
                const auto hashMapIndex = bucket * numFeatures + featureIdx;
                // 为了性能考虑不做边界检查，因为一定在其中
                unbucketizePermuteData[i] = uniqueMaps[hashMapIndex].find(idx)->second;
            }
        }
    }
}

// 核心分桶逻辑模板
template <bool Sequence,             // 是否序列模式
          bool HasWeight,            // 是否包含权重
          bool ReturnBucketMapping,  // 是否返回分桶映射
          typename OffsetT,          // 偏移量类型
          typename IndexT,           // 索引类型
          typename ScalarT,          // 标量类型
          bool DoUnique              // 是否去重
          >
void BlockBucketizeSparseFeaturesCpuKernel(const at::Tensor& lengths, const at::Tensor& indices,
                                           const std::optional<at::Tensor>& weights, const bool bucketizePos,
                                           const at::Tensor& blockSizes,
                                           const std::optional<at::Tensor>& totalNumBlocks, const int64_t mySize,
                                           at::Tensor newLengths, at::Tensor newIndices,
                                           std::optional<at::Tensor> newWeights, std::optional<at::Tensor> newPos,
                                           const std::optional<at::Tensor>& unbucketizePermute,
                                           const std::optional<at::Tensor>& batchSizePerFeature,
                                           const std::optional<std::vector<at::Tensor>>& blockBucketizePos,
                                           const std::optional<at::Tensor>& bucketMapping, const bool keepOrigIdx)
{
    // 基本参数校验
    const auto lengthsSize = lengths.numel();
    const auto newLengthsSize = lengthsSize * mySize;
    const int32_t numFeatures = blockSizes.numel();
    if (numFeatures == 0) {
        return;
    }
    const int32_t batchSize = lengthsSize / numFeatures;

    // 预分配偏移量数组
    auto offsets = at::empty({lengthsSize + 1}, lengths.options());
    auto newOffsets = at::empty({newLengthsSize + 1}, lengths.options());

    // 数据指针获取
    OffsetT* lengthsData = lengths.data_ptr<OffsetT>();
    OffsetT* offsetsData = offsets.data_ptr<OffsetT>();
    const IndexT* indicesData = indices.data_ptr<IndexT>();

    // 计算原始偏移量
    PrefixSum(lengthsSize, lengthsData, offsetsData);
    TORCH_CHECK(offsetsData[lengthsSize] == indices.numel(),
                "Offset validation failed: offsets[lengthsSize] = ", offsetsData[lengthsSize],
                ", indices.numel() = ", indices.numel());

    // 新数据结构指针初始化
    OffsetT* newLengthsData = newLengths.data_ptr<OffsetT>();
    OffsetT* newOffsetsData = newOffsets.data_ptr<OffsetT>();
    IndexT* newIndicesData = newIndices.data_ptr<IndexT>();

    // 可选参数处理
    IndexT* unbucketizePermuteData = nullptr;
    if constexpr (Sequence) {
        unbucketizePermuteData = unbucketizePermute.value().data_ptr<IndexT>();
    }

    // 第一阶段: 计算新长度
    ComputeNewLengths<Sequence, OffsetT, IndexT, DoUnique>(offsetsData, indicesData, newLengthsData, numFeatures,
                                                           batchSize, mySize, lengthsSize);

    // 计算新偏移量
    PrefixSum(newLengthsSize, newLengthsData, newOffsetsData);

    // 第二阶段: 填充新索引
    FillNewIndices<Sequence, OffsetT, IndexT, DoUnique>(offsetsData, indicesData, newOffsetsData, newIndicesData,
                                                        unbucketizePermuteData, numFeatures, batchSize, mySize,
                                                        lengthsSize);

    // 去重逻辑 (需要时启用)
    if constexpr (DoUnique) {
        Deduplicate<OffsetT, IndexT>(newLengthsData, newOffsetsData, offsetsData, indicesData, newIndicesData,
                                     unbucketizePermuteData, numFeatures, batchSize, mySize);
    }
}
#include <optional>
#include <vector>

struct BucketizeOptions {
    const at::Tensor& lengths;
    const at::Tensor& indices;
    const at::Tensor& blockSizes;
    const int64_t mySize;
    bool bucketizePos = false;
    bool sequence = false;
    std::optional<at::Tensor> totalNumBlocks = std::nullopt;
    std::optional<at::Tensor> weights = std::nullopt;
    std::optional<at::Tensor> batchSizePerFeature = std::nullopt;
    int64_t maxBatchSize = 0;
    std::optional<std::vector<at::Tensor>> blockBucketizePos = std::nullopt;
    bool returnBucketMapping = false;
    bool keepOrigIdx = false;

    BucketizeOptions(
        const at::Tensor& l,
        const at::Tensor& i,
        const at::Tensor& b,
        const int64_t m,
        bool bp = false,
        bool s = false,
        std::optional<at::Tensor> tnb = std::nullopt,
        std::optional<at::Tensor> w = std::nullopt,
        std::optional<at::Tensor> bpf = std::nullopt,
        int64_t mb = 0,
        std::optional<std::vector<at::Tensor>> bbp = std::nullopt,
        bool rbm = false,
        bool ko = false
    ) : lengths(l), indices(i), blockSizes(b), mySize(m), bucketizePos(bp), sequence(s), totalNumBlocks(tnb), weights(w), batchSizePerFeature(bpf), maxBatchSize(mb), blockBucketizePos(bbp), returnBucketMapping(rbm), keepOrigIdx(ko) {}
};

BucketTensorResult BlockBucketizeSparseFeaturesCpu(const BucketizeOptions& options) {
    // 参数校验
    TORCH_CHECK(options.lengths.scalar_type() == at::kLong,
                "Lengths tensor must be int64 type, got: ", options.lengths.scalar_type());
    TORCH_CHECK(options.indices.scalar_type() == at::kLong,
                "Indices tensor must be int64 type, got: ", options.indices.scalar_type());
    TORCH_CHECK(!options.weights, "Weighted KJT is currently not supported");
    TORCH_CHECK(!options.bucketizePos, "Bucket position tracking is not implemented");
    TORCH_CHECK(!options.returnBucketMapping, "Bucket mapping return is not supported");

    // 初始化输出张量
    const auto lengthsSize = options.lengths.numel();
    const auto newLengthsSize = lengthsSize * options.mySize;

    auto newLengths = at::zeros({newLengthsSize}, options.lengths.options());
    auto newIndices = at::empty_like(options.indices);
    auto unbucketizePermute = at::empty(options.indices.sizes(), options.indices.options());

    // 根据序列模式选择不同内核
    if (options.sequence) {
        BlockBucketizeSparseFeaturesCpuKernel<true, false, false, int64_t, int64_t, int64_t, false>(
            options.lengths, options.indices, options.weights, options.bucketizePos, options.blockSizes,
            options.totalNumBlocks, options.mySize, newLengths, newIndices, std::nullopt, std::nullopt,
            unbucketizePermute, options.batchSizePerFeature, options.blockBucketizePos, std::nullopt,
            options.keepOrigIdx);
    } else {
        BlockBucketizeSparseFeaturesCpuKernel<false, false, false, int64_t, int64_t, int64_t, false>(
            options.lengths, options.indices, options.weights, options.bucketizePos, options.blockSizes,
            options.totalNumBlocks, options.mySize, newLengths, newIndices, std::nullopt, std::nullopt,
            unbucketizePermute, options.batchSizePerFeature, options.blockBucketizePos, std::nullopt,
            options.keepOrigIdx);
    }

    return {newLengths, newIndices, std::nullopt, std::nullopt, unbucketizePermute, std::nullopt};
}

}  // namespace hybrid