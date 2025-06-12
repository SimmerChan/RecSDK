/*
* Copyright (c) Meta Platforms, Inc. and affiliates.
* Copyright (c) huawei Platforms, Inc. and affiliates.
* All rights reserved.
*
* This source code is licensed under the BSD-style license found in the
* LICENSE file in the root directory of this source tree.
 */
#ifndef HASHMAP_DYNAMIC_LOAD_H
#define HASHMAP_DYNAMIC_LOAD_H
#include <dlfcn.h>
#include <torch/torch.h>

#include "parallel_hash_map.h"

namespace hybrid {
class HashMapDynamicLoad {
public:
    HashMapDynamicLoad() noexcept;
    std::unique_ptr<ParallelHashMap> GetHashmapInstance(int64_t n);
    ~HashMapDynamicLoad();
    void* handle = nullptr;
    static HashMapDynamicLoad gInstance;
};

class DefaultHashmap : public ParallelHashMap {
public:
    explicit DefaultHashmap(int64_t n)
    {
        map.reserve(n);
    };
    std::pair<bool, int64_t> Find(int64_t key) override
    {
        auto findResult = map.find(key);
        if (findResult == map.end()) {
            return {false, -1};
        } else {
            return {true, findResult->second};
        }
    };
    bool CheckOrExpansion(int64_t insertNum) override
    {
        return false;
    }
    bool Insert(int64_t k, int64_t v) override
    {
        return map.insert_or_assign(k, v).second;
    };
    int64_t Size() override
    {
        return map.size();
    }
    void Clear() override
    {
        map.clear();
    }
    int64_t Capacity() override
    {
        return INT_MAX;
    }
    ~DefaultHashmap() override{};

private:
    ska::flat_hash_map<int64_t, int64_t> map;
    std::mutex mtx;
};

}  // namespace hybrid
#endif