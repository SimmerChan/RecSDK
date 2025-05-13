#ifndef DEFAULT_HASHMAP_H
#define DEFAULT_HASHMAP_H

#include <c10/util/Exception.h>
#include <c10/util/flat_hash_map.h>
#include <dlfcn.h>

#include <cstdint>
#include <iostream>
#include <mutex>
#include <unordered_map>

namespace hybrid {
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
}
#endif