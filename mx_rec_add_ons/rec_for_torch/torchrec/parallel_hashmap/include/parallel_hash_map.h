#ifndef HYBRID_HASHMAP_H
#define HYBRID_HASHMAP_H

#include <memory.h>

#include <cstdint>
#include <cstdio>
#include <utility>

namespace hybrid {

class ParallelHashMap {
public:
    ParallelHashMap() {}
    explicit ParallelHashMap(int64_t capacity){};
    virtual std::pair<bool, int64_t> Find(int64_t key) = 0;
    virtual bool Insert(int64_t k, int64_t v) = 0;
    virtual int64_t Size() = 0;
    virtual bool CheckOrExpansion(int64_t insertNum)= 0;
    virtual int64_t Capacity() = 0;
    virtual void Clear() = 0;
    virtual ~ParallelHashMap(){};
};

// Create
using CreateMap = ParallelHashMap*(int n);
using DestroyMap = void(ParallelHashMap*);

}  // namespace hybrid
#endif