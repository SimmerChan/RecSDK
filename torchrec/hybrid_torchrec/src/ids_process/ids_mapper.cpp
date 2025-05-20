#include "ids_mapper.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <utility>

#include "torch/torch.h"

namespace hybrid {
void IdsMapper::UniqueAndLookupOut(const torch::Tensor& globalIds, const torch::Tensor& hashIndices,
                                   const torch::Tensor& offset, const torch::Tensor& unique,
                                   const torch::Tensor& uniqueInverse, const torch::Tensor& uniqueOffset,
                                   int64_t tableId, bool isUnique)
{
    at::ThreadLocalStateGuard tlsGrad(state);
    RECORD_FUNCTION(c10::str("hybrid::UniqueAndLookupOut"), c10::ArrayRef<const c10::IValue>());

    int64_t* hashIndicesPtr = hashIndices.data_ptr<int64_t>();
    int64_t* globalIdsPtr = globalIds.data_ptr<int64_t>();
    int64_t* offsetPtr = offset.data_ptr<int64_t>();
    int64_t start = offsetPtr[tableId];
    int64_t end = offsetPtr[tableId + 1];
    int64_t* uniqueOffsetPtr = uniqueOffset.data_ptr<int64_t>();
    if (start == end) {
        uniqueOffsetPtr[tableId + 1] = uniqueOffsetPtr[tableId];
        return;
    }
    for (int64_t i = start; i < end; i++) {
        int64_t key = globalIdsPtr[i];
        auto findResult = ids2indicesMap.find(key);
        if (findResult == ids2indicesMap.end()) {
            std::lock_guard<std::mutex> lock(insertMute);
            TORCH_CHECK(key < initMaxIndex,
                "indices = ", key,
                "must smaller than table Size", initMaxIndex);
            auto findResult = ids2indicesMap.find(key);
            if (findResult == ids2indicesMap.end()) {
                int64_t r = maxIndex++;
                auto findResult = ids2indicesMap.find(key);
                ids2indicesMap.insert_or_assign(key, r);
                hashIndicesPtr[i] = r;
            } else {
                hashIndicesPtr[i] = findResult->second;
            }
        } else {
            hashIndicesPtr[i] = findResult->second;
        }
    }

    // Unique
    if (isUnique) {
        UniqueProcessing(hashIndices, offset, unique, uniqueInverse, uniqueOffset, tableId);
    }
}

void IdsMapper::UniqueProcessing(const torch::Tensor& hashIndices, const torch::Tensor& offset,
                                 const torch::Tensor& unique, const torch::Tensor& uniqueInverse,
                                 const torch::Tensor& uniqueOffset, int64_t tableId)
{
    at::ThreadLocalStateGuard tlsGrad(state);
    RECORD_FUNCTION(c10::str("hybrid::UniqueProcessing"), c10::ArrayRef<const c10::IValue>());

    int64_t* hashIndicesPtr = hashIndices.data_ptr<int64_t>();
    int64_t* offsetPtr = offset.data_ptr<int64_t>();
    int64_t start = offsetPtr[tableId];
    int64_t end = offsetPtr[tableId + 1];
    int64_t* uniqueOffsetPtr = uniqueOffset.data_ptr<int64_t>();
    if (start == end) {
        uniqueOffsetPtr[tableId + 1] = uniqueOffsetPtr[tableId];
        return;
    }

    auto aHashMap = AllocFullHashMap();
    auto aHashMapPtr = aHashMap->data();
    int64_t* uniquePtr = unique.data_ptr<int64_t>();
    int64_t* uniqueInversePtr = uniqueInverse.data_ptr<int64_t>();
    if (uniqueOffsetPtr == nullptr || uniquePtr == nullptr || uniqueInversePtr == nullptr) {
        printf("[ERROR] unique, uniqueOffset and uniqueInverse cannot be None!");
        return;
    }
    int64_t globalUniqueOffset = uniqueOffsetPtr[tableId];
    int64_t thisUniqueOffset = 0;

    for (const auto i : c10::irange(start, end)) {
        int64_t key = hashIndicesPtr[i];
        if (aHashMapPtr[key] == -1) {
            aHashMapPtr[key] = thisUniqueOffset;
            uniquePtr[globalUniqueOffset + thisUniqueOffset] = hashIndicesPtr[i];
            thisUniqueOffset++;
        }
    }

    if (tableId == 0) {
        uniqueOffsetPtr[tableId] = 0;
    }
    uniqueOffsetPtr[tableId + 1] = uniqueOffsetPtr[tableId] + thisUniqueOffset;

    for (const auto i : c10::irange(globalUniqueOffset, globalUniqueOffset + thisUniqueOffset)) {
        int64_t key = uniquePtr[i];
        aHashMapPtr[key] = i - globalUniqueOffset;
    }

    for (const auto i : c10::irange(start, end)) {
        int64_t key = hashIndicesPtr[i];
        uniqueInversePtr[i] = aHashMapPtr[key];
    }

    for (const auto i : c10::irange(globalUniqueOffset, globalUniqueOffset + thisUniqueOffset)) {
        int64_t key = uniquePtr[i];
        aHashMapPtr[key] = -1;
    }

    DeallocFullHashMap(std::move(aHashMap));
}
}  // namespace hybrid