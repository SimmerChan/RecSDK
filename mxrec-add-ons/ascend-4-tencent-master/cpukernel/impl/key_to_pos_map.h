#pragma once
#include <memory>
#include <unordered_map>
#include <vector>

namespace pctr{
namespace util{
    
constexpr float USING_MAP_OVERLOAD_RATIO = 0.3;

template <typename T>
class BaseKeyToPosMap{
    public:
        virtual bool Find(const T& key, int64_t* pos) const = 0;
        virtual bool Build(const std::vector<T>& keys, const T max) = 0;
        virtual ~BaseKeyToPosMap(){}
};

template <typename T>
class KeyToPosMapper : public BaseKeyToPosMap<T> {
    public:
        KeyToPosMapper() = default;
        ~KeyToPosMapper() = default;
        bool Find(const T& key, int64_t* pos) const override;
        bool Build(const std::vector<T>& keys, const T max) override;
    private:
        std::unordered_map<T, int64_t> map_;
};

template <typename T>
class KeyToPosVector : public BaseKeyToPosMap<T> {
    public:
        KeyToPosVector() = default;
        ~KeyToPosVector() = default;
        bool Find(const T& key, int64_t* pos) const override;
        bool Build(const std::vector<T>& keys, const T max) override;
    private:
        std::vector<int64_t> vec_;
};

template <typename T>
class KeyToPosMap{
    public:
        KeyToPosMap() = default;
        ~KeyToPosMap() = default;
        bool Find(const T& key, int64_t* pos) const;
        bool Build(const std::vector<T>& keys);
    private:
        float ratio_{USING_MAP_OVERLOAD_RATIO};
        std::shared_ptr<BaseKeyToPosMap<T> > map_ptr_;
};
} // namespace util

} // namespace pctr