/* Copyright 2025. Huawei Technologies Co.,Ltd. All rights reserved.

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
#include <emock/emock.hpp>

#include "pcie_through/rma_shm_svm.h"

using namespace MxRec;

namespace MxRec {
    void InitShmHeader(RmaShmHeader* header, int64_t memSize, int32_t capacity);
}

TEST(TestGetShmAddr, Basic)
{
    std::string name = "test";
    int deviceId = 1;
    int capacity = 51;

    EMOCK(GetChipName).stubs().will(returnValue(std::string("testDeviceName")));
    try {
        int64_t res = GetShmAddr(name, deviceId, capacity);
        EXPECT_GE(res, 0);
    }
    catch  (const std::runtime_error& e) {
        std::cerr << "Caught expected runtime_error: " << e.what() << std::endl;
    }
}

TEST(TestGetShmAddr, InitShmHeader_NormalCase)
{
    const int64_t memSize = 1024 * 1024; // 1MB
    const int32_t capacity = 100;
    RmaShmHeader header;

    InitShmHeader(&header, memSize, capacity);
    
    EXPECT_EQ(header.totalMemSize, memSize - RMA_SHM_HEAD_LEN);
    EXPECT_EQ(header.queueCapacity, capacity);
    EXPECT_EQ(header.seqIn, 0);
    EXPECT_EQ(header.seqOut, 0);
    EXPECT_EQ(header.frontOffset, RMA_SHM_HEAD_LEN);
    EXPECT_EQ(header.tailOffset, RMA_SHM_HEAD_LEN);
    EXPECT_EQ(header.buffLimit, 0);
    EXPECT_EQ(header.seqOutPre, 0);
    EXPECT_EQ(header.frontOffsetPre, RMA_SHM_HEAD_LEN);
}
