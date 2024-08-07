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
    NOT_FOUND = 1,
    FILE_NOT_EXIST = 2,
    NOT_SUPPORTED = 3,
    INVALID_ARGUMENT = 4,
    IO_ERROR = 5,
    ACL_ERROR = 6,
    UNKNOWN,
};

enum class ModuleName {
    CHECK_POINT = 1,
    EMB_TABLE = 2,
    FILE_SYSTEM = 3,
    HD_TRANSFER = 4,
    HYBRID_MGMT = 5,
    KEY_PROCESS = 6,
    L3_STORAGE = 7,
    SSD_ENGINE = 8,
    UTILS = 9,
};

class Error {
public:
    Error() = delete;
    Error(ModuleName mod, ErrorType e, const std::string& msg) : mod_(mod), e_(e), msg_(msg) {}

    std::string ToString() const;

private:
    std::string TypeAsString() const noexcept;
    std::string ModAsString() const noexcept;

    ModuleName mod_;
    ErrorType e_;
    std::string msg_;
};

}  // namespace MxRec

#endif
