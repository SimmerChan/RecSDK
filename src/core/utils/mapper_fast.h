//
// Created by z00576261 on 2024/4/15.
//

#ifndef FAST_MAPPERFAST_H
#define FAST_MAPPERFAST_H

#include <atomic>
#include <mutex>
#include <algorithm>
#include <functional>
#include <cmath>
#include <iostream>
#include "securec.h"

namespace RecMapper {
    constexpr size_t BUCKCAPACITY = 3;

    enum BuckStatus {
        BUCK_POS_0 = 0,
        BUCK_POS_1,
        BUCK_POS_2,
        BUCK_EXIST,
        BUCK_NOEXIST,
        BUCK_ERROR
    };

    class SpinLock {
    public:
        SpinLock() = default;
        SpinLock(const SpinLock&) = delete;
        SpinLock& operator=(const SpinLock) = delete;

        void lock()
        {
            while (f.test_and_set(std::memory_order_acquire)) {};
        }

        void unlock()
        {
            f.clear(std::memory_order_release);
        }

    private:
        std::atomic_flag f;
    };

    template<class K, class V>
    struct InnerBuck {
        std::pair<std::atomic<K>, V> datas_[BUCKCAPACITY];
        InnerBuck* next_ = nullptr;
        SpinLock spin;

        BuckStatus Insert(K key, V& value, std::function<bool()> ValueSet)
        {
            for (size_t i = 0; i < BUCKCAPACITY; ++i) {
                K old_key = 0;
                if (datas_[i].first.load(std::memory_order_relaxed) == 0 &&
                    datas_[i].first.compare_exchange_strong(old_key, key)) {
                    bool ret = ValueSet();
                    if (!ret) {
                        datas_[i].first.store(0);
                        return BuckStatus::BUCK_ERROR;
                    }
                    datas_[i].second = value;
                    if (i == 0) {
                        return BuckStatus::BUCK_POS_0;
                    } else if (i == 1) {
                        return BuckStatus::BUCK_POS_1;
                    } else {
                        return BuckStatus::BUCK_POS_2;
                    };
                }
            }
            return BuckStatus::BUCK_ERROR;
        }

        BuckStatus Find(K key)
        {
            for (size_t i = 0; i < BUCKCAPACITY; ++i) {
                if (datas_[i].first.load(std::memory_order_relaxed) == key) {
                    if (i == 0) {
                        return BuckStatus::BUCK_POS_0;
                    } else if (i == 1) {
                        return BuckStatus::BUCK_POS_1;
                    } else {
                        return BuckStatus::BUCK_POS_2;
                    };
                }
            }
            return BuckStatus::BUCK_NOEXIST;
        }

        BuckStatus Remove(K key)
        {
            for (size_t i = 0; i < BUCKCAPACITY; ++i) {
                K oldkey = key;
                if (datas_[i].first.load(std::memory_order_relaxed) == key) {
                    if (datas_[i].first.compare_exchange_strong(oldkey, 0)) {
                        datas_[i].second = 0;
                        return BuckStatus::BUCK_EXIST;
                    }
                }
            }
            return BUCK_ERROR;
        }
    };

    template<class K, class T>
    class FasterMapper;

    template<class K, class V, class T, class Ref, class Ptr>
    class Iterator {
        template<class k, class v>
        friend class FasterMapper;
    public:
        typedef Iterator<K, V, T, Ref, Ptr> Self;

        Iterator(InnerBuck<K, V>* node, size_t pos, FasterMapper<K, V>* map): node_(node), pos_(pos), map_(map) {}
        Iterator(InnerBuck<K, V>* node, size_t pos, const FasterMapper<K, V>* map):node_(node), pos_(pos), map_(map) {}
        Iterator(const Iterator& other):node_(other.node_), pos_(other.pos_), map_(other.map_) {}

        Ref operator*()
        {
            return node_->datas_[pos_];
        }

        Ptr operator->()
        {
            return &node_->datas_[pos_];
        }

        Self operator++()
        {
            if (map_->use_spec_buck && IsSpecBuck()) {
                InnerBuck<K, V>* temp_buck = nullptr;
                size_t temp_pos = 0;
                if (FindNextBuck(0, 0, temp_buck, temp_pos)) {
                    node_ = temp_buck;
                    pos_ = temp_pos;
                    return *this;
                }
                node_ = nullptr;
                pos_ = 0;
                return *this;
            }

            size_t temp_pos = pos_ + 1;
            while (temp_pos < BUCKCAPACITY) {
                if (node_->datas_[temp_pos].first.load(std::memory_order_relaxed) != 0) {
                    pos_ = temp_pos;
                    return  *this;
                }
                temp_pos++;
            }
            while (node_->next_ != nullptr) {
                for (size_t i = 0; i < BUCKCAPACITY; ++i) {
                    if (node_->next_->datas_[i].first.load(std::memory_order_relaxed) != 0) {
                        pos_ = i;
                        node_ = node_->next_;
                        return *this;
                    }
                }
            }
            size_t map_index = node_->datas_[pos_].first.load(std::memory_order_relaxed) % map_->sub_map_count_;
            size_t buck_index = node_->datas_[pos_].first.load(std::memory_order_relaxed) % map_->buck_count_;
            GetNextIndex(map_index, buck_index);
            InnerBuck<K, V>* temp_buck = nullptr;
            temp_pos = 0;
            for (; map_index < map_->sub_map_count_; map_index++) {
                if (map_->buck_maps_[map_index] == nullptr) {
                    continue;
                }
                if (FindNextBuck(map_index, buck_index, temp_buck, temp_pos)) {
                    node_ = temp_buck;
                    pos_ = temp_pos;
                    return *this;
                }
                buck_index = 0;
            }
            node_ = nullptr;
            pos_ = 0;
            return *this;
        }

        bool operator==(const Self& other)
        {
            return node_ == other.node_;
        }

        bool operator!=(const Self& other)
        {
            return node_ != other.node_;
        }

    private:
        InnerBuck<K, V>* node_;
        size_t pos_;
        const FasterMapper<K, V>* map_;
        bool enter_spec = false;

        void GetNextIndex(size_t& map_index, size_t& buck_index)
        {
            map_index = (buck_index == (map_->buck_count_ - 1) ? ++map_index : map_index);
            buck_index = buck_index % (map_->buck_count_ - 1) + (buck_index ==  (map_->buck_count_ - 1) ? 0 : 1);
        }

        bool IsSpecBuck()
        {
            return node_ == map_->spec_buck;
        }

        bool FindNextBuck(size_t map_index, size_t buck_index, InnerBuck<K, V>*& buck, size_t& pos)
        {
            for (; buck_index < map_->buck_count_; buck_index++) {
                InnerBuck<K, V>* temp_buck = &map_->buck_maps_[map_index][buck_index];
                for (size_t i = 0; i < BUCKCAPACITY; ++i) {
                    if (temp_buck->datas_[i].first.load(std::memory_order_relaxed) != 0) {
                        pos = i;
                        buck = temp_buck;
                        return true;
                    }
                }
            }
            return false;
        }
    };

    template<class K, class V>
    class FasterMapper {
    public:
        typedef std::pair<std::atomic<K>, V> T;

        typedef Iterator<K, V, T, T&, T*> iterator;
        typedef Iterator<K, V, T, const T&, const T*> const_iterator;

        FasterMapper(size_t cap, size_t res) : capacity_(cap), reserve_(res) {};

        ~FasterMapper() = default;

        bool InitializeBuck()
        {
            size_t i = 0;

            while (i <= prime_max_) {
                if (pow(pow_base_, i) < reserve_) {
                    i++;
                    continue;
                }
                break;
            }
            buck_count_ = std::max(min_buck_num_, static_cast<int>(pow(pow_base_, i)));
            if (buck_count_ == 0 || buck_count_ > pow(pow_base_, prime_max_)) {
                return false;
            }
            for (auto &buck_map : buck_maps_) {
                InnerBuck<K, V>* buck_map_temp = new (std::nothrow) InnerBuck<K, V>[buck_count_];
                if (buck_map_temp == nullptr) {
                    FreeBuckMaps();
                    return false;
                }
                memset_s(buck_map_temp, sizeof(InnerBuck<K, V>) * buck_count_, 0,
                         sizeof(InnerBuck<K, V>) * buck_count_);
                buck_map = buck_map_temp;
            }
            return true;
        }

        void UnInitializeBuck()
        {
            FreeBuckExpend();
            FreeBuckMaps();
        }

        std::pair<iterator, bool> PutNotFind(InnerBuck<K, V>* buck, const K& key, V& value)
        {
            for (int i = 0; i < loop_max_; ++i) {
                // insert exist buck
                while (buck != nullptr) {
                    buck->spin.lock();
                    auto value_func = [&]() ->bool {
                        value = offset_.fetch_add(1);
                        return true;
                    };
                    BuckStatus ret = buck->Insert(key, value, value_func);
                    buck->spin.unlock();

                    if (ret == BuckStatus::BUCK_ERROR) {
                        return std::pair(iterator(nullptr, 0, this), false);
                    } else if (ret != BuckStatus::BUCK_ERROR) {
                        size_.fetch_add(1);
                        return std::pair(iterator(buck, static_cast<size_t>(ret), this), true);
                    }

                    if (buck->next_ != nullptr) {
                        buck = buck->next_;
                    } else {
                        break;
                    }
                }

                // insert not exist buck
                auto& old_spin = buck->spin;
                old_spin.lock();
                if (buck->next_ != nullptr) {
                    buck = buck->next_;
                    old_spin.unlock();
                    continue;
                }

                InnerBuck<K, V>* new_buck =  new (std::nothrow) InnerBuck<K, V>;
                memset_s(new_buck, sizeof(InnerBuck<K, V>), 0, sizeof(InnerBuck<K, V>));
                buck->next_ = new_buck;
                buck = new_buck;
                old_spin.unlock();
            }
            return std::pair(iterator(nullptr, 0, this), false);
        }

        std::pair<iterator, bool> PutFind(InnerBuck<K, V>* buck, const K& key, V& value)
        {
            while (buck != nullptr) {
                buck->spin.lock();
                auto status = buck->Find(key);
                value = buck->datas_[static_cast<size_t>(status)].second;
                if (status != BuckStatus::BUCK_NOEXIST) {
                    buck->spin.unlock();
                    return std::pair(iterator(buck, static_cast<size_t>(status), this), true);
                }
                buck->spin.unlock();

                if (buck->next_ != nullptr) {
                    buck = buck->next_;
                } else {
                    break;
                }
            }
            return std::pair(iterator(nullptr, 0, this), false);
        };

        std::pair<iterator, bool> Put(const K& key, V& value)
        {
            if (size_.load() == capacity_) {
                return std::pair(iterator(nullptr, 0, this), false);
            }

            if (key == 0) {
                if (spec_buck != nullptr) {
                    return std::pair(iterator(spec_buck, 0, this), true);
                }
                spec_buck =  new (std::nothrow) InnerBuck<K, V>;
                memset_s(spec_buck, sizeof(InnerBuck<K, V>), 0, sizeof(InnerBuck<K, V>));

                spec_buck->spin.lock();
                spec_buck->datas_[0].first.store(key);
                value = offset_.fetch_add(1);
                spec_buck->datas_[0].second = value;
                size_.fetch_add(1);
                spec_buck->spin.unlock();

                use_spec_buck = true;
                return std::pair(iterator(spec_buck, 0, this), true);
            }
            InnerBuck<K, V>* temp_buck = &(buck_maps_[key % sub_map_count_][key % buck_count_]);
            // first，find key if exist in buck
            std::pair<iterator, bool> put_ret = PutFind(temp_buck, key, value);
            if (put_ret.second == true) {
                return put_ret;
            }

            // if not find,
            put_ret = PutNotFind(temp_buck, key, value);
            if (put_ret.second == true) {
                return put_ret;
            }
            return std::pair(iterator(nullptr, 0, this), false);
        }

        iterator Find(const K& key)
        {
            if (key == 0) {
                if (spec_buck != nullptr) {
                    return iterator(spec_buck, 0, this);
                }
                return iterator(nullptr, 0, this);
            }
            InnerBuck<K, V>* temp_buck = &(buck_maps_[key % sub_map_count_][key % buck_count_]);
            if (temp_buck == nullptr) {
                return iterator(nullptr, 0, this);
            }
            auto status = temp_buck->Find(key);
            if (status == BuckStatus::BUCK_NOEXIST) {
                return  iterator(nullptr, 0, this);
            } else {
                return iterator(temp_buck, static_cast<size_t>(status), this);
            }
        }

        bool Remove(const K& key)
        {
            if (key == 0) {
                if (spec_buck != nullptr) {
                    delete spec_buck;
                    spec_buck = nullptr;
                    size_.fetch_sub(1);
                    use_spec_buck = false;
                    return true;
                }
                return false;
            }
            InnerBuck<K, V>* temp_buck = &(buck_maps_[key % sub_map_count_][key % buck_count_]);
            while (temp_buck != nullptr) {
                if (temp_buck->Find(key) == BuckStatus::BUCK_NOEXIST) {
                    return false;
                }
                temp_buck->spin.lock();
                if (temp_buck->Remove(key) == BuckStatus::BUCK_EXIST) {
                    temp_buck->spin.unlock();
                    size_.fetch_sub(1);
                    return true;
                }
                temp_buck->spin.unlock();
                temp_buck = temp_buck->next_;
            }
            return false;
        }

        bool ToVector(std::vector<std::pair<K, V>>& vec)
        {
            if (spec_buck != nullptr) {
                vec.push_back(std::make_pair(spec_buck->datas_[0].first.load(), spec_buck->datas_[0].second));
            }
            for (auto& sub_map : buck_maps_) {
                if (sub_map == nullptr) {
                    continue;
                }
                for (size_t i = 0; i < buck_count_; ++i) {
                    InnerBuck<K, V>* temp_buck = &sub_map[i];
                    while (temp_buck) {
                        for (size_t j = 0; j < BUCKCAPACITY && temp_buck->datas_[j].first != 0; ++j) {
                            vec.push_back(std::make_pair(temp_buck->datas_[j].first.load(),
                                                         temp_buck->datas_[j].second));
                        }
                        temp_buck = temp_buck->next_;
                    }
                }
            }
            return true;
        }

        iterator begin()
        {
            if (spec_buck != nullptr) {
                return iterator(spec_buck, 0, this);
            }
            for (auto &sub_map: buck_maps_) {
                if (sub_map == nullptr) {
                    continue;
                }
                for (size_t i = 0; i < buck_count_; ++i) {
                    InnerBuck<K, V> *buck = &sub_map[i];
                    while (buck) {
                        for (size_t j = 0; j < BUCKCAPACITY && buck->datas_[j].first != 0; ++j) {
                            return iterator(buck, j, this);
                        }
                        buck = buck->next_;
                    }
                }
            }
            return iterator(nullptr, 0, this);
        }

        iterator end()
        {
            return iterator(nullptr, 0, this);
        }

        const_iterator begin() const
        {
            if (spec_buck != nullptr) {
                return iterator(spec_buck, 0, this);
            }
            for (auto &sub_map: buck_maps_) {
                if (sub_map == nullptr) {
                    continue;
                }
                for (size_t i = 0; i < buck_count_; ++i) {
                    InnerBuck<K, V> *buck = &sub_map[i];
                    while (buck) {
                        for (size_t j = 0; j < BUCKCAPACITY && buck->datas_[j].first != 0; ++j) {
                            return iterator(buck, j, this);
                        }
                        buck = buck->next_;
                    }
                }
            }
            return iterator(nullptr, 0, this);
        }

        const_iterator  end() const
        {
            return iterator(nullptr, 0, this);
        }

        size_t Size()
        {
            return size_.load();
        }

        size_t Capacity()
        {
            return capacity_;
        }

        void FreeBuckMaps()
        {
            for (auto &buck_map : buck_maps_) {
                if (buck_map != nullptr) {
                    delete[] buck_map;
                    buck_map = nullptr;
                }
            }
            if (spec_buck != nullptr) {
                delete spec_buck;
                spec_buck = nullptr;
            }
            offset_.store(0);
            size_.store(0);
            reserve_ = 0;
            buck_count_ = 0;
            capacity_ = 0;
        }

        void FreeBuckExpend()
        {
            for (auto &buck_map : buck_maps_) {
                if (buck_map == nullptr) {
                    continue;
                }
                for (size_t i = 0; i < buck_count_; ++i) {
                    InnerBuck<K, V>* buck_attch = buck_map[i].next_;
                    while (buck_attch != nullptr) {
                        InnerBuck<K, V>* buck_attch_temp = buck_attch->next_;
                        delete buck_attch;
                        buck_attch = buck_attch_temp;
                    }
                }
            }
        }

        std::atomic<size_t> offset_{ 0 };
        std::atomic<size_t> size_{ 0 };

        size_t reserve_;
        size_t buck_count_;
        size_t  capacity_;

        static constexpr size_t sub_map_count_ = 5;
        static constexpr size_t prime_max_ = 32;
        static constexpr int pow_base_ = 2;
        static constexpr int min_buck_num_ = 128;
        static constexpr int loop_max_ = 8192;

        InnerBuck<K, V>* buck_maps_[sub_map_count_] {};
        InnerBuck<K, V>* spec_buck = nullptr;

        bool use_spec_buck = false;
    };
}

#endif // FAST_MAPPERFAST_H
