/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <algorithm>
#include <chrono>
#include <vector>
#include <thread>

#include "gtest/gtest.h"

#define private public
#include "common/types.h"
#include "feature/time_evictor.h"

namespace rec_sdk {
namespace feature {

using std::vector;
using std::chrono::seconds;
using std::this_thread::sleep_for;

using common::emb_key_t;

class TimeEvictorTest : public ::testing::Test {
protected:
    void SetUp() override
    {
        testFilePath = "./test_time_evictor.bin";
    }
    void TearDown() override {}

private:
    std::string testFilePath;
};

TEST_F(TimeEvictorTest, EvictNoneKeys)
{
    auto timeEvictor = TimeEvictor(0, 2, nullptr);

    auto keys = vector<emb_key_t>{1, 2, 3};
    const auto expected = vector<emb_key_t>{};

    timeEvictor.Update(keys);

    sleep_for(seconds(1));
    timeEvictor.Update(keys);

    auto res = timeEvictor.Evict();
    EXPECT_EQ(res, expected);
}

TEST_F(TimeEvictorTest, EvictSomeKeys)
{
    auto timeEvictor = TimeEvictor(0, 2, nullptr);

    auto keys = vector<emb_key_t>{1, 2, 3};
    const auto expected = vector<emb_key_t>{3};

    timeEvictor.Update(keys);

    sleep_for(seconds(1));
    keys = vector<emb_key_t>{1, 2};
    timeEvictor.Update(keys);

    sleep_for(seconds(2));
    keys = vector<emb_key_t>{};
    timeEvictor.Update(keys);

    auto res = timeEvictor.Evict();
    std::sort(res.begin(), res.end());
    EXPECT_EQ(res, expected);
}

TEST_F(TimeEvictorTest, EvictAllKeys)
{
    auto timeEvictor = TimeEvictor(0, 2, nullptr);

    auto keys = vector<emb_key_t>{1, 2, 3};
    const auto expected = vector<emb_key_t>{1, 2, 3};

    timeEvictor.Update(keys);

    sleep_for(seconds(3));
    keys = vector<emb_key_t>{};
    timeEvictor.Update(keys);

    auto res = timeEvictor.Evict();
    std::sort(res.begin(), res.end());
    EXPECT_EQ(res, expected);
}

TEST_F(TimeEvictorTest, SaveAndLoadOK)
{
    auto timeEvictor = TimeEvictor(0, 2, nullptr);

    auto keys = vector<emb_key_t>{1, 2, 3, 1, 2, 4};
    timeEvictor.Update(keys);

    timeEvictor.Save(testFilePath);
    timeEvictor.lastVisitedTimes_.clear();
    
    timeEvictor.Load(testFilePath);
    EXPECT_EQ(timeEvictor.lastVisitedTimes_.size(), 4);
}

}  // namespace feat
}  // namespace rec_sdk
