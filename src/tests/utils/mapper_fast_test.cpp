/* Copyright 2024. Huawei Technologies Co.,Ltd. All rights reserved.

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

#include <gtest/gtest.h>
#include "utils/mapper_fast.h"

#include <vector>
#include <thread>
#include <random>
#include <iostream>
using namespace std;
using namespace testing;
using namespace RecMapper;

namespace TempVal {
    constexpr int THREAD_NUM = 4;
    constexpr int MAPPER_CAP = 180;
    constexpr int RESERVE_SPLIT = 15;
    constexpr int MAPPER_INSERT_CAP = 1800;
    constexpr int MAPPER_INSERT_COUNT = 200;
    constexpr int MAPPER_INSERT_NEW = 100;
}

TEST(InnerBuck, BuckCase)
{
    InnerBuck<::uint64_t, ::uint64_t>* buck = new (std::nothrow) InnerBuck<::uint64_t, ::uint64_t>;
    memset(buck, 0, sizeof(InnerBuck<::uint64_t, ::uint64_t>));
    buck->spin.lock();
    std::vector<uint64_t> keys = {1, 2, 3, 1, 4};
    std::vector<uint64_t> values = {1, 2, 3, 4};
    auto func = []() ->bool {return true;};
    BuckStatus ret_status = buck->Insert(keys[0], values[0], func);
    ASSERT_EQ(ret_status, BuckStatus::BUCK_POS_0);
    ret_status = buck->Insert(keys[1], values[1], func);
    ASSERT_EQ(ret_status, BuckStatus::BUCK_POS_1);
    ret_status = buck->Insert(keys[2], values[2], func);
    ASSERT_EQ(ret_status, BuckStatus::BUCK_POS_2);
    ret_status = buck->Insert(keys[3], values[3], func);
    ASSERT_EQ(ret_status, BuckStatus::BUCK_ERROR);
    buck->spin.unlock();

    buck->spin.lock();
    ret_status = buck->Find(keys[0]);
    ASSERT_EQ(ret_status, BuckStatus::BUCK_POS_0);
    ret_status = buck->Find(keys[4]);
    ASSERT_EQ(ret_status, BuckStatus::BUCK_NOEXIST);
    buck->spin.unlock();

    buck->spin.lock();
    ret_status = buck->Remove(keys[0]);
    ASSERT_EQ(ret_status, BuckStatus::BUCK_EXIST);
    ret_status = buck->Remove(keys[4]);
    ASSERT_EQ(ret_status, BuckStatus::BUCK_ERROR);
    buck->spin.unlock();
}

vector<int64_t> getRandom(size_t total, size_t use)
{
    if (total <= 0) {
        return {};
    }
    std::default_random_engine e;
    std::vector<int64_t> input;
    for (size_t i = 0; i < total; i++) {
        input.push_back(i);
    }
    vector<int64_t> output;
    int end = total;
    for (size_t i = 0; i < total && output.size() < use; i++) {
        vector<int64_t>::iterator iter = input.begin();
        int64_t num = e() % end;
        iter = iter + num;
        output.push_back(*iter);
        input.erase(iter);
        end--;
    }
    return output;
}

void GenerateKeys(vector<int64_t>& keys, int64_t& key_start, std::vector<int>& total_ids_num,
                  std::vector<int>& look_ids_num, int i)
{
    std::random_device rd;
    std::mt19937 g(rd());
    int new_ids_num =  (i > 0) ? (total_ids_num[i] - total_ids_num[i - 1]) : total_ids_num[i];
    int old_ids_num = look_ids_num[i] - new_ids_num;
    // old
    keys = getRandom(key_start, old_ids_num);
    // new
    for (int j = 0; j < new_ids_num; ++j) {
        keys.push_back(j + key_start);
    }
    std::shuffle(keys.begin(), keys.end(), g);
    key_start += new_ids_num;
}

void parallelGetInsert(FasterMapper<::uint64_t, ::uint64_t>* map, const std::vector<uint64_t>& ids)
{
    uint64_t failed_num = 0;
    for (uint64_t i = 0; i < ids.size(); i++) {
        uint64_t value;
        auto ret = map->Put(ids[i], value);
        if (ret.second == false) {
            failed_num++;
        }
    }
    ASSERT_EQ(failed_num, 0);
}

TEST(FasterMapper, PutCase)
{
    FasterMapper<::uint64_t, ::uint64_t> tMap(TempVal::MAPPER_INSERT_CAP,
                                              TempVal::MAPPER_INSERT_CAP / TempVal::RESERVE_SPLIT);
    bool ret = tMap.InitializeBuck();
    ASSERT_EQ(ret, true);

    int64_t key_start = 0;
    std::vector<int> look_ids_num(9, TempVal::MAPPER_INSERT_COUNT);
    std::vector<int> total_ids_num(9, TempVal::MAPPER_INSERT_COUNT);
    for (int i = 0; i < 9; ++i) {
        total_ids_num[i] += TempVal::MAPPER_INSERT_NEW * i;
    }
    for (size_t i = 0; i < look_ids_num.size(); ++i) {
        vector<int64_t> keys;
        GenerateKeys(keys, key_start, total_ids_num, look_ids_num, i);
        std::vector<std::vector<uint64_t>> ids_vec;
        size_t signum = keys.size() / TempVal::THREAD_NUM;
        for (size_t j = 0; j < (TempVal::THREAD_NUM - 1); j++) {
            ids_vec.push_back(vector<uint64_t>(keys.begin() + j * signum, keys.begin() + (j+1) * signum));
        }
        ids_vec.push_back(vector<uint64_t>(keys.begin() + (TempVal::THREAD_NUM - 1) * signum, keys.end()));
        std::vector<std::thread> workers(TempVal::THREAD_NUM);
        for (size_t j = 0; j < TempVal::THREAD_NUM; j++) {
            if (j == 0) {
                workers[j] = std::thread(parallelGetInsert, &tMap, ids_vec[j]);
            } else if (j == 1) {
                workers[j] = std::thread(parallelGetInsert, &tMap, ids_vec[j]);
            } else if (j == 2) {
                workers[j] = std::thread(parallelGetInsert, &tMap, ids_vec[j]);
            } else {
                workers[j] = std::thread(parallelGetInsert, &tMap, ids_vec[j]);
            }
        }
        for (size_t j = 0; j < TempVal::THREAD_NUM; j++) {
            workers[j].join();
        }
        ASSERT_EQ(tMap.Size(), total_ids_num[i]);
    }
}

TEST(FasterMapper, FindAndRemoveCase)
{
    FasterMapper<::uint64_t, ::uint64_t> tMap(TempVal::MAPPER_CAP, TempVal::MAPPER_CAP / TempVal::RESERVE_SPLIT);
    bool ret = tMap.InitializeBuck();
    ASSERT_EQ(ret, true);
    vector<uint64_t> val_insert;
    for (int i = 0; i < TempVal::MAPPER_CAP; i++) {
        uint64_t value;
        auto put_ret = tMap.Put(i, value);
        ASSERT_EQ(put_ret.second, true);
        val_insert.push_back(value);
    }
    for (int i = 0; i < TempVal::MAPPER_CAP; i++) {
        auto find_ret = tMap.Find(i);
        ASSERT_EQ(find_ret->first.load(), i);
        ASSERT_EQ(find_ret->second, val_insert[i]);
    }
    uint64_t error_value;
    auto put_ret = tMap.Put(TempVal::MAPPER_CAP + 1, error_value);
    ASSERT_EQ(put_ret.second, false);
    
    for (auto it = tMap.begin(); it != tMap.end(); ++it) {
        ::uint64_t val = it->second;
        auto iter = std::find(val_insert.begin(), val_insert.end(), val);
        ASSERT_NE(iter, val_insert.end());
    }

    std::vector<std::pair<uint64_t, uint64_t>> vec;
    tMap.ToVector(vec);
    ASSERT_EQ(vec.size(), TempVal::MAPPER_CAP);

    auto remove_ret = tMap.Remove(0);
    ASSERT_EQ(remove_ret, true);
    ASSERT_EQ(tMap.Size(), TempVal::MAPPER_CAP - 1);

    for (size_t i = 1; i < val_insert.size(); ++i) {
        auto find_ret = tMap.Remove(i);
        ASSERT_EQ(find_ret, true);
    }
    ASSERT_EQ(tMap.Size(), 0);
    tMap.UnInitializeBuck();
    ASSERT_EQ(tMap.Capacity(), 0);
}
