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

#include "error.h"

#include <cstdint>
#include <string>

#include "absl/strings/str_format.h"

namespace MxRec {
std::string Error::ToString() const
{
    std::string modName = this->ModAsString();
    std::string errType = this->TypeAsString();

    return absl::StrFormat("Module: %s, Error Type: %s, Error Message: %s", modName, errType, this->msg_);
}

uint32_t Error::ErrorCode() const
{
    return static_cast<uint32_t>(this->mod_) * 100 + static_cast<uint32_t>(this->e_);
}

std::string Error::TypeAsString() const noexcept
{
    switch (this->e_) {
        case ErrorType::kNotFound:
            return "NotFound";
        case ErrorType::kFileNotExist:
            return "FileNotExist";
        case ErrorType::kNotSupported:
            return "NotSupported";
        case ErrorType::kInvalidArgument:
            return "InvalidArgument";
        case ErrorType::kIOError:
            return "IOError";
        case ErrorType::kAscendCLError:
            return "AscendCError";
        default:
            return "UnknownError";
    }
}

std::string Error::ModAsString() const noexcept
{
    switch (this->mod_) {
        case ModuleName::checkPoint:
            return "CheckPoint";
        case ModuleName::embTable:
            return "EmbTable";
        case ModuleName::fileSystem:
            return "FileSystem";
        case ModuleName::hdTransfer:
            return "HdTransfer";
        case ModuleName::hybridMgmt:
            return "HybridMgmt";
        case ModuleName::keyProcess:
            return "KeyProcess";
        case ModuleName::l3Storage:
            return "L3Storge";
        case ModuleName::ssdEngine:
            return "SsdEngine";
        case ModuleName::utils:
            return "Utils";
        default:
            return "UnknownModule";
    }
}
}  // namespace MxRec
