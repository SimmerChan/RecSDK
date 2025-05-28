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

#ifndef DEVICE_INTERFACE_H
#define DEVICE_INTERFACE_H

#include <cstdint>
#include <cstddef>

namespace tensorflow {
namespace npu_xla {

class DeviceInterface {
public:
    virtual ~DeviceInterface() = default;

    virtual void* Allocate(size_t size) = 0;

    virtual void Deallocate(void* ptr) = 0;

    virtual bool MemcpyHToD(void* dst, size_t dstSize, const void* src, size_t srcSize) = 0;

    virtual bool MemcpyDToH(void* dst, size_t dstSize, const void* src, size_t srcSize) = 0;

    static DeviceInterface& Create(int32_t deviceId);
};

} // namespace npu_xla
} // namespace tensorflow

#endif // DEVICE_INTERFACE_H