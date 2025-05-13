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
}  // namespace hybrid
#endif