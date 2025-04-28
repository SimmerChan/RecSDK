#include <folly/AtomicHashMap.h>
#include <parallel_hash_map.h>

#include <cstdint>
#include <cstdio>
#include <memory>
#include <utility>

constexpr float EXPAND_FACTOR = 0.75;
namespace hybrid {
class FollyAtomic : public ParallelHashMap {
public:
    FollyAtomic() = delete;
    explicit FollyAtomic(int n)
    {
        map = std::make_unique<folly::AtomicHashMap<int64_t, int64_t>>(n);
    };
    std::pair<bool, int64_t> Find(int64_t key) override
    {
        auto findResult = map->find(key);
        if (findResult == map->end()) {
            return {false, -1};
        } else {
            return {true, findResult->second};
        }
    };
    bool Insert(int64_t k, int64_t v) override
    {
        bool insertSucess = map->insert(k, v).second;
        return insertSucess;
    };
    int64_t Size() override
    {
        return map->size();
    }
    bool CheckOrExpansion(int64_t insertNum) override
    {
        if (map->size() + insertNum > map->capacity() * EXPAND_FACTOR) {
            printf("Warning HashMap Expansion");
            auto mapNew = std::make_unique<folly::AtomicHashMap<int64_t, int64_t>>(map->capacity() * 8);
            for (auto it = map->begin(); it != map->end(); it++) {
                mapNew->insert(it->first, it->second);
            }
            map = std::move(mapNew);
            return true;
        }
        return false;
    }
    int64_t Capacity() override
    {
        return map->size();
    }
    void Clear() override
    {
        map->clear();
    }
    ~FollyAtomic() = default;

private:
    std::unique_ptr<folly::AtomicHashMap<int64_t, int64_t>> map;
};

extern "C" ParallelHashMap* create(int n)
{
    return new FollyAtomic(n);
}
extern "C" void destroy(ParallelHashMap* p)
{
    delete p;
}
}  // namespace hybrid
