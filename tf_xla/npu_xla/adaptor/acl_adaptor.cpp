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

#include "acl_adaptor.h"

#include <acl/acl.h>
#include "tsl/platform/logging.h"

using namespace tensorflow::npu_xla;

AclAdaptor &AclAdaptor::GetInstance(int32_t deviceId) {
    static AclAdaptor instance;
    instance.SetDevice(deviceId);
    return instance;
}

AclAdaptor::AclAdaptor() {
    auto ret = aclInit(nullptr);
    VLOG(2) << "aclInit, ret: " << ret;
}

AclAdaptor::~AclAdaptor() {
    aclFinalize();
}

void *AclAdaptor::Allocate(size_t size) {
    void *ptr = nullptr;
    auto ret = aclrtMalloc(&ptr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
        VLOG(3) << "aclrtMalloc " << "size " << size << " failed, ret: " << ret;
        return nullptr;
    }
    return ptr;
}

void AclAdaptor::Deallocate(void *ptr) {
    auto ret = aclrtFree(ptr);
    if (ret != ACL_SUCCESS) {
        VLOG(3) << "aclrtFree " << "ptr " << ptr << " failed, ret: " << ret;
    }
}

void AclAdaptor::SetDevice(int32_t deviceId) {
    auto ret = aclrtSetDevice(deviceId);
    VLOG(2) << "aclrtSetDevice " << deviceId << " ret: " << ret;
}

bool AclAdaptor::MemcpyHToD(void *dst, size_t dstSize, const void *src, size_t srcSize) {
    auto ret = aclrtMemcpy(dst, dstSize, src, srcSize, ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) {
        VLOG(3) << "aclrtMemcpy " << srcSize << " bytes from host to device failed, ret: " << ret;
        return false;
    }
    return true;
}

bool AclAdaptor::MemcpyDToH(void *dst, size_t dstSize, const void *src, size_t srcSize) {
    auto ret = aclrtMemcpy(dst, dstSize, src, srcSize, ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_SUCCESS) {
        VLOG(3) << "aclrtMemcpy " << srcSize << " bytes from device to host failed, ret: " << ret;
        return false;
    }
    return true;
}