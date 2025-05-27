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
#include <gmock/gmock.h>
#include "tensorflow/c/experimental/stream_executor/stream_executor.h"
#include "adaptor/device_interface_mock.h"

using ::testing::_;
using ::testing::Return;

#ifdef GOOGLE_TEST

namespace tensorflow {
namespace npu_xla {

// 全局 mock 指针定义
DeviceInterfaceMock* mock_adaptor = nullptr;

// 替换 DeviceInterface::Create 方法
DeviceInterface& DeviceInterface::Create(int32_t deviceId) {
    return *mock_adaptor;
}

}  // namespace npu_xla
}  // namespace tensorflow

#endif  // GOOGLE_TEST

TEST(CFunctionTest, SE_InitPlugin_Allocate_ReturnsMock) {

    // 创建 mock 实例
    tensorflow::npu_xla::DeviceInterfaceMock mock;
    tensorflow::npu_xla::mock_adaptor = &mock;

    // 构造 SP_Device
    SP_Device device{};
    device.struct_size = sizeof(SP_Device);
    device.ordinal = 0;  // deviceId

    // 构造 SP_DeviceMemoryBase
    SP_DeviceMemoryBase mem{};
    mem.struct_size = SP_DEVICE_MEMORY_BASE_STRUCT_SIZE;
    mem.size = 0;

    // 构造 SE_PlatformRegistrationParams
    SE_PlatformRegistrationParams params{};
    SP_Platform platform{};
    SP_PlatformFns platform_fns{};
    params.struct_size = sizeof(SE_PlatformRegistrationParams);
    params.ext = nullptr;
    params.major_version = 2;
    params.minor_version = 18;
    params.patch_version = 0;
    params.platform = &platform;
    params.platform_fns = &platform_fns;

    TF_Status* status = TF_NewStatus();

    // 调用 NPU 插件入口函数
    SE_InitPlugin(&params, status);
    EXPECT_EQ(TF_GetCode(status), TF_OK);

    // 构造 SE_CreateStreamExecutorParams
    SE_CreateStreamExecutorParams create_params{};
    SP_StreamExecutor se{};
    create_params.struct_size = sizeof(SE_CreateStreamExecutorParams);
    create_params.ext = nullptr;
    create_params.stream_executor = &se; 

    // 调用 create_stream_executor
    platform_fns.create_stream_executor(&platform, &create_params, status);
    EXPECT_EQ(TF_GetCode(status), TF_OK);
    ASSERT_NE(&se, nullptr);

    // -------------------------------
    // 测试 Allocate
    // -------------------------------
    void* fake_ptr = reinterpret_cast<void*>(0x12345678);
    EXPECT_CALL(mock, Allocate(1024)).WillOnce(Return(fake_ptr));

    se.allocate(&device, 1024, 0, &mem);

    EXPECT_EQ(mem.opaque, fake_ptr);
    EXPECT_EQ(mem.size, 1024);

    // -------------------------------
    // 测试 SyncMemcpyDToH (成功)
    // -------------------------------
    void* hostDst = reinterpret_cast<void*>(0x87654321);

    EXPECT_CALL(mock, MemcpyDToH(hostDst, 1024, fake_ptr, 1024))
        .WillOnce(Return(true));

    se.sync_memcpy_dtoh(&device, hostDst, &mem, 1024, status);
    EXPECT_EQ(TF_GetCode(status), TF_OK);

    // -------------------------------
    // 测试 SyncMemcpyHToD (失败)
    // -------------------------------
    void* hostSrc = reinterpret_cast<void*>(0x88888888);

    EXPECT_CALL(mock, MemcpyHToD(fake_ptr, 1024, hostSrc, 1024))
        .WillOnce(Return(false));

    se.sync_memcpy_htod(&device, &mem, hostSrc, 1024, status);
    EXPECT_EQ(TF_GetCode(status), TF_INTERNAL);
    EXPECT_STREQ(TF_Message(status), "MemcpyHToD failed");

    // -------------------------------
    // 测试 Deallocate
    // -------------------------------
    EXPECT_CALL(mock, Deallocate(fake_ptr)).Times(1);

    se.deallocate(&device, &mem);

    EXPECT_EQ(mem.opaque, nullptr);
    EXPECT_EQ(mem.size, 0);
}