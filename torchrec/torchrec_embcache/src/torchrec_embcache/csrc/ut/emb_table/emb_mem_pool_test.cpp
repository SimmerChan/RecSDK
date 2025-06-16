/*
* Copyright (c) Meta Platforms, Inc. and affiliates.
* Copyright (c) huawei Platforms, Inc. and affiliates.
* All rights reserved.
*
* This source code is licensed under the BSD-style license found in the
* LICENSE file in the root directory of this source tree.
 */
#include "emb_table/emb_mem_pool.h"

#include "../common_main.h"

using namespace Embcache;

class EmbMemPoolTest : public ::testing::Test {
protected:
    void SetUp() override
    {
        EmbConfig embConfig;
        embConfig.tableName = "user_table";
        embConfig.embDim = 8;
        embConfig.optimNum = 2;
        embConfig.weightInitMin = -0.5;
        embConfig.weightInitMax = 0.5;

        uint64_t bufferSize = 1024;
        uint64_t hostVocabSize = 1024 * 1024 * 1024;

        memPoolPtr = std::make_shared<EmbMemoryPool>(embConfig, bufferSize, hostVocabSize);
    }

    void TearDown() override
    {
        memPoolPtr->Stop();
    }

    std::shared_ptr<EmbMemoryPool> memPoolPtr;
};

TEST_F(EmbMemPoolTest, GetNewValueToBeInserted)
{
    uint64_t value;
    BeforePutFuncState state = memPoolPtr->GetNewValueToBeInserted(value);
    LOG(INFO) << "BeforePutFuncState, value:" << value;
    ASSERT_EQ(1, 1);
}

int main(int argc, char *argv[])
{
    return common_main(argc, argv);
}