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

#ifndef DEVICE_INTERFACE_MOCK_H
#define DEVICE_INTERFACE_MOCK_H

#include <gmock/gmock.h>
#include "device_interface.h"

namespace tensorflow {
namespace npu_xla {

class DeviceInterfaceMock : public DeviceInterface {
public:
    MOCK_METHOD(void*, Allocate, (size_t), (override));

    MOCK_METHOD(void, Deallocate, (void*), (override));

    MOCK_METHOD(bool, MemcpyHToD, (void*, size_t, const void*, size_t), (override));
    
    MOCK_METHOD(bool, MemcpyDToH, (void*, size_t, const void*, size_t), (override));
};

}  // namespace npu_xla
}  // namespace tensorflow

#endif // DEVICE_INTERFACE_MOCK_H