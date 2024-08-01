/* Copyright 2024. Huawei Technologies Co.,Ltd. All rights reserved.

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

#ifndef MXREC_CORE_ACL_RAII_H_
#define MXREC_CORE_ACL_RAII_H_

#include "acl/acl_tdt.h"

namespace MxRec {

namespace Inner {
aclError DestroyAclChannel(acltdtChannelHandle*);
}

template <typename T, aclError (*Fn)(T*)>
class AclResourceRAII {
public:
    AclResourceRAII() : inner_(nullptr) {}
    explicit AclResourceRAII(T* inner) : inner_(inner) {}
    ~AclResourceRAII()
    {
        if (this->inner_ != nullptr) {
            Fn(this->inner_);
            this->inner_ = nullptr;
        }
    }
    AclResourceRAII(const AclResourceRAII& rhs) = delete;
    AclResourceRAII& operator=(const AclResourceRAII& rhs) = delete;
    AclResourceRAII(AclResourceRAII&& rhs) noexcept : inner_(rhs.inner_)
    {
        rhs.inner_ = nullptr;
    }
    AclResourceRAII& operator=(AclResourceRAII&& rhs) noexcept
    {
        if (this != &rhs) {
            if (this->inner_ != nullptr) {
                Fn(this->inner_);
            }
            this->inner_ = rhs.inner_;
            rhs.inner_ = nullptr;
        }
        return *this;
    }

    T* ptr() const
    {
        return this->inner_;
    }

private:
    T* inner_;
};

/// RAII wrapper for `acltdtDataset*`
using AcltdtDataset = AclResourceRAII<acltdtDataset, acltdtDestroyDataset>;

/// RAII wrapper for `acltdtChannelHandle*`
using AcltdtChannelHandle = AclResourceRAII<acltdtChannelHandle, Inner::DestroyAclChannel>;

}  // namespace MxRec

#endif
