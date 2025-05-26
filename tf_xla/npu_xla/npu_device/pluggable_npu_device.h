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

#ifndef PLUGGABLE_NPU_DEVICE_H
#define PLUGGABLE_NPU_DEVICE_H

#include <cstdint>
#include <cstddef>
#include "tensorflow/c/experimental/stream_executor/stream_executor.h"

namespace tensorflow {
namespace npu_xla {

void Allocate(const struct SP_Device* const device, uint64_t size, int64_t memorySpace, struct SP_DeviceMemoryBase* const mem);

void Deallocate(const struct SP_Device* const device, struct SP_DeviceMemoryBase* const mem);

void MemcpyDToH(const struct SP_Device* const device, void* stream, void* hostDst,
                const struct SP_DeviceMemoryBase* const device_src, uint64_t size, void* status);
                
void MemcpyHToD(const struct SP_Device* const device, void* stream, struct SP_DeviceMemoryBase* const device_dst,
                const void* hostSrc, uint64_t size, void* status);

void PopulateNpuPlatformRegistrationParams(SE_PlatformRegistrationParams* const params);

void CreateStreamExecutor(const SP_Platform* platform, SE_CreateStreamExecutorParams* params, TF_Status* status);
                
}  // namespace npu_xla
}  // namespace tensorflow

void SE_InitPlugin(SE_PlatformRegistrationParams* const params, TF_Status* const status);

#endif  // PLUGGABLE_NPU_DEVICE_H