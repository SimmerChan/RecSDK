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

#include <cstdint>
#include <cstddef>

namespace tensorflow {
namespace npu_xla {

class AclAdaptor {
public:
    static AclAdaptor &GetInstance(int32_t deviceId);

    ~AclAdaptor();

    void *Allocate(size_t size);

    void Deallocate(void *ptr);

    bool MemcpyHToD(void *dst, size_t dstSize, const void *src, size_t srcSize);

    bool MemcpyDToH(void *dst, size_t dstSize, const void *src, size_t srcSize);

private:
    AclAdaptor();

    void SetDevice(int32_t deviceID);

};

} // namespace npu_xla
} // namesapce tensorflow