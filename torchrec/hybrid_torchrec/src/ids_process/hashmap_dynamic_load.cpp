#include <c10/util/Exception.h>
#include <c10/util/flat_hash_map.h>
#include <dlfcn.h>
#include <hashmap_dynamic_load.h>

#include "default_hashmap.h"

#include <cstdint>
#include <iostream>
#include <mutex>
#include <unordered_map>

namespace hybrid {
HashMapDynamicLoad::HashMapDynamicLoad() noexcept
{
    const char* pathOfSo = getenv("PARALLEL_HASH_MAP_SO");
    if (pathOfSo == nullptr) {
        TORCH_WARN_ONCE("PARALLEL_HASH_MAP_SO is None, Use DefaultHashmap, it may be cause low performance");
        return;
    }
    std::string pathOfSoStr = pathOfSo;
    if (pathOfSoStr.empty()) {
        TORCH_WARN_ONCE("PARALLEL_HASH_MAP_SO is None, Use DefaultHashmap, it may be cause low performance");
        return;
    }
}

std::unique_ptr<ParallelHashMap> HashMapDynamicLoad::GetHashmapInstance(int64_t n)
{
    if (handle == nullptr) {
        TORCH_WARN_ONCE("Handle is None, Use DefaultHashmap, it may be cause low performance\n");
        return std::make_unique<DefaultHashmap>(n);
    }
    CreateMap* create = (CreateMap*)dlsym(handle, "create");
    DestroyMap* destroy = (DestroyMap*)dlsym(handle, "destroy");
    if (create == nullptr or destroy == nullptr) {
        TORCH_WARN_ONCE(
            "Hashmap.so don't contain create method, Use DefaultHashmap, it may be cause low performance\n");
        return std::make_unique<DefaultHashmap>(n);
    }
    ParallelHashMap* p = create(n);
    std::unique_ptr<ParallelHashMap> map_ptr(p);
    return map_ptr;
}

HashMapDynamicLoad::~HashMapDynamicLoad()
{
    if (handle != nullptr) {
        dlclose(handle);
    }
}
HashMapDynamicLoad HashMapDynamicLoad::gInstance;
}  // namespace hybrid
