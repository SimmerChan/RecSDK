/* Copyright (c) Huawei Technologies Co., Ltd. 2022-2024. All rights reserved.
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

#include "external_logger.h"

#include <cstring>

namespace ock {
void ExternalLogger::SetExternalLogFunction(ExternalLog func)
{
    if (mLogFunc == nullptr) {
        mLogFunc = func;
    }
}

void ExternalLogger::Log(const int level, const std::ostringstream &oss, const char* file, int line) const
{
    if (mLogFunc != nullptr) {
        std::cout << "[OCK][" << (strrchr(file, '/') ? strrchr(file, '/') + 1 : file) << ":" << line << "] " << oss.str().c_str() << std::endl;
    }
}

void ExternalLogger::PrintLog(LogLevel level, const std::string &message, const char* file, int line)
{
    std::ostringstream oss;
    oss << message;
    auto logger = ExternalLogger::Instance();
    if (logger != nullptr) {
        logger->Log(static_cast<int>(level), oss, file, line);
    }
}

void ExternalLogger::PrintLog(LogLevel level, const std::string &message, bool flag, const char* file, int line)
{
    if (flag) {
        PrintLog(level, message, file, line);
    }
}
}
