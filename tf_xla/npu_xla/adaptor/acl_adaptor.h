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

#ifndef ACL_ADAPTOR_H
#define ACL_ADAPTOR_H

#include <cstdint>
#include <cstddef>
#include "device_interface.h"

namespace tensorflow {
namespace npu_xla {

class AclAdaptor : public DeviceInterface {
public:
    static AclAdaptor &GetInstance(int32_t deviceId);

    ~AclAdaptor() override;

    void *Allocate(size_t size) override;

    void Deallocate(void *ptr) override;

    bool MemcpyHToD(void *dst, size_t dstSize, const void *src, size_t srcSize) override;

    bool MemcpyDToH(void *dst, size_t dstSize, const void *src, size_t srcSize) override;

private:
    AclAdaptor();
    void SetDevice(int32_t deviceID);
};

} // namespace npu_xla
} // namespace tensorflow

#endif // ACL_ADAPTOR_H