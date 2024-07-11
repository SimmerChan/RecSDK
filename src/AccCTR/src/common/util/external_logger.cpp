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

#include <ctime>
#include <sys/time.h>
#include <cstring>

namespace ock {

int ExternalLogger::rank = 0;

void ExternalLogger::SetExternalLogFunction(ExternalLog func)
{
    if (mLogFunc == nullptr) {
        mLogFunc = func;
    }
}

void ExternalLogger::Log(const int level, const std::string &message, const char* file, int line) const
{
    if (mLogFunc != nullptr) {
        std::stringstream ss;
        struct tm t;
        struct timeval tv;
        gettimeofday(&tv, nullptr);
        localtime_r(&tv.tv_sec, &t);

        ss << "[OCK][" << YEAR_BASE + t.tm_year << "/" << 1 + t.tm_mon << "/" << t.tm_mday<< " "
           << t.tm_hour << ":" << t.tm_min << ":" << t.tm_sec << "." << tv.tv_usec << "] ["
           << ExternalLogger::rank << "] ["<< ExternalLogger::LevelToStr(level) << "] ["
           << (strrchr(file, '/') ? strrchr(file, '/') + 1 : file) << ":" << line << "] " << message << std::endl;

        std::cout << ss.str();
    }
}

void ExternalLogger::PrintLog(LogLevel level, const std::string &message, const char* file, int line)
{
    auto logger = ExternalLogger::Instance();
    if (logger != nullptr) {
        logger->Log(static_cast<int>(level), message, file, line);
    }
}

void ExternalLogger::PrintLog(LogLevel level, const std::string &message, bool flag, const char* file, int line)
{
    if (flag) {
        PrintLog(level, message, file, line);
    }
}

const char* ExternalLogger::LevelToStr(int logLevel) const
{
    static const char* msg[] = {
            "DEBUG",
            "INFO",
            "WARN",
            "ERROR",
    };
    return msg[logLevel];
}

void ExternalLogger::SetRank(const int logRank) {
    ExternalLogger::rank = logRank;
}
}
