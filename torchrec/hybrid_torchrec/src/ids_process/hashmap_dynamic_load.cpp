#include <c10/util/Exception.h>
#include <c10/util/flat_hash_map.h>
#include <dlfcn.h>
#include <hashmap_dynamic_load.h>

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

HashMapDynamicLoad::HashMapDynamicLoad() noexcept
{
    char* pathOfSo = getenv("PARALLEL_HASH_MAP_SO");
    if (pathOfSo == nullptr) {
        TORCH_WARN_ONCE(" PARALLEL_HASH_MAP_SO is None, Use DefaultHashmap, it may be cause low performance\n");
        return;
    }
    handle = dlopen(pathOfSo, RTLD_LAZY | RTLD_GLOBAL);
};

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
