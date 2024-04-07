/* Copyright (c) Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
 ==============================================================================*/

#ifndef MXREC_COMMON_H
#define MXREC_COMMON_H

#include <cstdint>
#include <vector>
#include <atomic>

#ifndef HM_UNLIKELY
#define HM_UNLIKELY(x) __builtin_expect(!!(x), 0)
#endif

#ifndef HM_LIKELY
#define HM_LIKELY(x) __builtin_expect(!!(x), 1)
#endif

namespace EmbCache {

class LimitedSet {
public:
    struct Node {
        uint64_t value;
        Node *prev, *next;
        Node(uint64_t val = -1) : value(val), prev(nullptr), next(nullptr) {}
    };

    LimitedSet(uint64_t maxRange) : head(new Node(-1)), tail(new Node(-1))
    {
        nodes.resize(maxRange);
        for (auto &node : nodes) {
            node = new Node(-1);
        }
        head->next = tail;
        tail->prev = head;
    }

    ~LimitedSet()
    {
        for (auto &node : nodes) {
            delete node;
        }
        delete head;
        delete tail;
    }

    void insert(uint64_t value)
    {
        if (nodes[value]->value == value) {
            return;
        }
        Node *node = nodes[value];
        node->value = value;
        Node *next = head->next;
        node->next = next;
        node->prev = head;
        head->next = node;
        next->prev = node;
    }

    void remove(uint64_t value)
    {
        if (nodes[value]->value != value) {
            return;
        }
        Node *node = nodes[value];
        node->prev->next = node->next;
        node->next->prev = node->prev;
        node->value = -1;
    }

    bool find(uint64_t value)
    {
        return nodes[value]->value == value;
    }

    class Iterator {
    public:
        Iterator(Node *node) : current(node) {}
        bool operator != (const Iterator &other) const
        {
            return current != other.current;
        }
        const uint64_t &operator*() const
        {
            return current->value;
        }
        Iterator &operator ++ ()
        {
            current = current->next;
            return *this;
        }

    private:
        Node *current;
    };

    Iterator begin()
    {
        return { head->next };
    }

    Iterator end()
    {
        return { tail };
    }

private:
    Node *head;
    Node *tail;
    std::vector<Node *> nodes;
};

enum class FkvState {
    FKV_EXIST,
    FKV_NOT_EXIST,
    FKV_KEY_CONFLICT,
    FKV_BEFORE_PUT_FUNC_FAIL,
    FKV_BEFORE_REMOVE_FUNC_FAIL,
    FKV_NO_SPACE,
    FKV_FAIL,
};

enum class BeforePutFuncState {
    BEFORE_SUCCESS,
    BEFORE_NO_SPACE,
    BEFORE_FAIL,
};

enum class BeforeRemoveFuncState {
    BEFORE_SUCCESS,
    BEFORE_FAIL,
};

extern int64_t INVALID_KEY;
constexpr uint64_t TABLE_NAME_MAX_SIZE = 1024;
const uint32_t VOCAB_CACHE_RATIO = 15;
constexpr float NORMAL_MEAN_MAX = 1e9;
constexpr float NORMAL_MEAN_MIN = -1e9;
constexpr float NORMAL_STDDEV_MAX = 100;
constexpr float NORMAL_STDDEV_MIN = 0;
constexpr float CONSTANT_VALUE_MAX = 1e9;
constexpr float CONSTANT_VALUE_MIN = -1e9;
constexpr float INIT_K_MAX = 10000;
constexpr float INIT_K_MIN = -10000;
}
#endif // MXREC_COMMON_H
