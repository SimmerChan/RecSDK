#include "key_to_pos_map.h"

namespace pctr {

namespace util {

template <typename T>
bool KeyToPosMapper<T>::Find(const T& key, int64_t* pos) const {
    auto iter = map_.find(key);
    if (iter == map_.end()){
        return false;
    }
    *pos = iter->second;
    return true;
}

template <typename T>
bool KeyToPosMapper<T>::Build(const std::vector<T>& keys, const T max) {
    for (int64_t i = 0; i < keys.size(); ++i){
        map_[keys[i]] = i;
    }
    return true;
}

template <typename T>
bool KeyToPosVector<T>::Find(const T& key, int64_t* pos) const {
    if (key < vec_.size() && vec_[key] != -1) {
        *pos = vec_[key];
        return true;
    }
    return false;
}

template <typename T>
bool KeyToPosVector<T>::Build(const std::vector<T>& keys, const T max) {
    vec_.resize(max, -1);
    for (int64_t i = 0; i < keys.size(); ++i) {
        vec_[keys[i]] = i;
    }
    return true;
}

template <typename T>
bool KeyToPosMap<T>::Build(const std::vector<T>& keys) {
    T key_max;
    for (auto& key : keys) {
        key_max = (key > key_max) ? key : key_max;
    }

    if (key_max > (keys.size() * (1 + ratio_))) {
        map_ptr_ = std::make_shared<KeyToPosMapper<T>>();
    } else {
        map_ptr_ = std::make_shared<KeyToPosVector<T>>();
    }

    return map_ptr_->Build(keys, key_max);
}

template <typename T>
bool KeyToPosMap<T>::Find(const T& key, int64_t* pos) const {
    return map_ptr_->Find(key, pos);
}

template class KeyToPosMap<int64_t>;

} // namespace util

} // namespace pctr