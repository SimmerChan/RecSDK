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

#include <vector>

#include "gtest/gtest.h"

#define private public
#include "common/types.h"
#include "feature/count_filter.h"

namespace rec_sdk {
namespace feature {

using std::vector;

using common::emb_key_t;
using common::i32;

class CountFilterTest : public ::testing::Test {
protected:
    void SetUp() override
    {
        testFilePath = "./test_count_filter.bin";
    }
    void TearDown() override {}

private:
    std::string testFilePath;
};

TEST_F(CountFilterTest, FilterOutAllKeys)
{
    auto countFilter = CountFilter(0, 2, nullptr, nullptr);

    auto keys = vector<emb_key_t>{0, 1, 2, 3, 4, 5};
    auto cnts = vector<i32>{1, 1, 1, 1, 1, 1};
    const auto expected = vector<emb_key_t>{-1, -1, -1, -1, -1, -1};

    countFilter.Filter(keys, cnts);
    EXPECT_EQ(keys, expected);
}

TEST_F(CountFilterTest, FilterOutSomeKeys)
{
    auto countFilter = CountFilter(0, 2, nullptr, nullptr);

    auto keys = vector<emb_key_t>{1, 2, 3, 1, 2, 4};
    auto cnts = vector<i32>{1, 1, 1, 1, 1, 1};
    const auto expected = vector<emb_key_t>{1, 2, -1, 1, 2, -1};

    countFilter.Filter(keys, cnts);
    EXPECT_EQ(keys, expected);
}

TEST_F(CountFilterTest, FilterOutNoKey)
{
    auto countFilter = CountFilter(0, 2, nullptr, nullptr);

    auto keys = vector<emb_key_t>{1, 1, 2, 2, 3, 3};
    auto cnts = vector<i32>{1, 1, 1, 1, 1, 1};
    const auto expected = vector<emb_key_t>{1, 1, 2, 2, 3, 3};

    countFilter.Filter(keys, cnts);
    EXPECT_EQ(keys, expected);
}

TEST_F(CountFilterTest, SaveAndLoadOK)
{
    auto countFilter = CountFilter(0, 2, nullptr, nullptr);

    auto keys = vector<emb_key_t>{1, 2, 3, 1, 2, 4};
    auto cnts = vector<i32>{1, 1, 1, 1, 1, 1};
    countFilter.Filter(keys, cnts);

    countFilter.Save(testFilePath);
    countFilter.histVisitedCnts_.clear();
    
    countFilter.Load(testFilePath);
    EXPECT_EQ(countFilter.histVisitedCnts_.size(), 4);
}

}  // namespace feat
}  // namespace rec_sdk
