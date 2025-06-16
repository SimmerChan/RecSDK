/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * Copyright (c) huawei Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef EMBEDDING_CACHE_SWAP_MANAGER_H
#define EMBEDDING_CACHE_SWAP_MANAGER_H

#include <vector>
#include <unordered_set>
#include <cstdint>
#include <stdexcept>
#include <c10/util/flat_hash_map.h>

namespace Embcache {

class LimitedSet {
public:
    explicit LimitedSet(int64_t maxRange)
    {
        capacity_ = nextPowerOfTwo(maxRange);
        sparse_ = std::vector<int64_t>(capacity_, -1);
        dense_ = std::vector<int64_t>(capacity_);
        capacity_mask_ = capacity_ - 1;
        front_ = 0;
        size_ = 0;
    }

    void insert(int64_t value)
    {
        int64_t& idx = sparse_[value];
        if (idx != -1)
            return;
        if (size_ == capacity_)
            return;
        int64_t pos = (front_ + size_) & capacity_mask_;
        dense_[pos] = value;
        idx = pos;
        size_++;
    }

    void remove(int64_t value)
    {
        int64_t& idx = sparse_[value];
        if (idx == -1)
            return;
        int64_t last_pos = (front_ + size_ - 1) & capacity_mask_;
        int64_t last = dense_[last_pos];
        if (value != last) {
            dense_[idx] = last;
            sparse_[last] = idx;
        }
        size_--;
        idx = -1;
    }

    int64_t pop_front()
    {
        int64_t value = dense_[front_];
        sparse_[value] = -1;
        front_ = (front_ + 1) & capacity_mask_;
        size_--;
        return value;
    }

    bool find(int64_t value) const
    {
        return sparse_[value] != -1;
    }

    bool empty() const
    {
        return size_ == 0;
    }

    // 自定义迭代器（保持逻辑顺序）
    class Iterator {
        const LimitedSet* set_;
        int64_t index_;
        int64_t count_;

    public:
        Iterator(const LimitedSet* set, int64_t index, int64_t count) : set_(set), index_(index), count_(count) {}
        int64_t operator*() const
        {
            return set_->dense_[(set_->front_ + index_) & set_->capacity_mask_];
        }
        Iterator& operator++()
        {
            index_++;
            count_--;
            return *this;
        }
        bool operator!=(const Iterator& other) const
        {
            return count_ != other.count_;
        }
    };

    Iterator begin() const
    {
        return Iterator(this, 0, size_);
    }
    Iterator end() const
    {
        return Iterator(this, size_, 0);
    }

private:
    std::vector<int64_t> sparse_;
    std::vector<int64_t> dense_;
    int64_t front_;
    int64_t size_;
    int64_t capacity_;
    int64_t capacity_mask_;

    static int64_t nextPowerOfTwo(int64_t n)
    {
        if (n <= 0)
            return 1;
        n--;  // 处理n已经是2的幂的情况
        n |= n >> 1;
        n |= n >> 2;
        n |= n >> 4;
        n |= n >> 8;
        n |= n >> 16;
        n |= n >> 32;
        return n + 1;
    }
};

class SwapManager {
public:
    explicit SwapManager(int64_t cacheSize, int64_t memStartOffset = 0);

    std::tuple<std::vector<int64_t>, std::vector<int64_t>, std::vector<int64_t>, std::vector<int64_t>,
               std::vector<int64_t>>
    ComputeSwapInfo(const std::vector<int64_t>& keys);

    int64_t GetKey(int64_t off);
    int64_t GetOccupiedNum()
    {
        return occupiedNum;
    };
    void RemoveKeys(const std::vector<int64_t>& keys, std::vector<int64_t>& evictFeatures);
    int64_t GetMemStartOffset() const;

private:
    // 被淘汰的key的version给一个特殊标记，用以表示该位置可用；
    const int64_t CAN_REUSE_KEY_VERSION = -2;
    const int64_t INVALID_KEY = -1;
    const int64_t OFFSET_OF_INVALID_KEY = 0;
    // device中的start offset；若开启准入，start offset将被初始化为1
    int64_t memStartOffset = 0;

    int64_t cacheSize;
    int64_t occupiedNum = memStartOffset;
    ska::flat_hash_map<int64_t, int64_t> key2off;
    struct KeyVersion {
        int64_t key;
        int64_t version;
    };
    std::vector<KeyVersion> cache;
    int64_t nowVersion = 0;
    int64_t swapIdx = memStartOffset;
};
}  // namespace Embcache

#endif  // EMBEDDING_CACHE_SWAP_MANAGER_H
