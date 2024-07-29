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
#include <string>

#include "expected.h"

namespace MxRec {

enum class ErrorCode {
    // TODO add real error kinds
    kNotFound = 1,
    kFileNotExist = 2,
    kUnknown,
};

class Error {
public:
    Error() = delete;
    explicit Error(ErrorCode e) : e_(e), msg_() {}
    Error(ErrorCode e, const std::string& msg) : e_(e), msg_(msg) {}

    std::string ToString()
    {
        std::string res = this->CodeAsString();
        if (!this->msg_.empty()) {
            res += ": " + this->msg_;
        }
        return res;
    }

private:
    std::string CodeAsString()
    {
        switch (this->e_) {
            case ErrorCode::kNotFound:
                return "NotFound";
            case ErrorCode::kFileNotExist:
                return "FileNotExist";
            default:
                return "Unknown";
        }
    }

    ErrorCode e_;
    std::string msg_;
};

template <typename T>
using Expected = tl::expected<T, Error>;

using UnExpected = tl::unexpected<Error>;

template <typename... Args, typename std::enable_if<std::is_constructible<Error, Args&&...>::value>::type* = nullptr>
UnExpected make_unexpected(Args&&... args)
{
    return tl::unexpected<Error>(std::forward<Args>(args)...);
}
}  // namespace MxRec
