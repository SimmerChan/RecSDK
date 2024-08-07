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

#ifndef MXREC_CORE_UTILS_ERROR_H
#define MXREC_CORE_UTILS_ERROR_H

#include <cstdint>
#include <string>

namespace MxRec {

enum class ErrorType {
    kNotFound = 1,
    kFileNotExist = 2,
    kNotSupported = 3,
    kInvalidArgument = 4,
    kIOError = 5,
    kAscendCLError = 6,
    kUnknown,
};

enum class ModuleName {
    checkPoint = 1,
    embTable = 2,
    fileSystem = 3,
    hdTransfer = 4,
    hybridMgmt = 5,
    keyProcess = 6,
    l3Storage = 7,
    ssdEngine = 8,
    utils = 9,
};

class Error {
public:
    Error() = delete;
    Error(ModuleName mod, ErrorType e, const std::string& msg) : mod_(mod), e_(e), msg_(msg) {}

    std::string ToString() const;
    uint32_t ErrorCode() const;

private:
    std::string TypeAsString() const noexcept;
    std::string ModAsString() const noexcept;

    ModuleName mod_;
    ErrorType e_;
    std::string msg_;
};

}  // namespace MxRec

#endif
