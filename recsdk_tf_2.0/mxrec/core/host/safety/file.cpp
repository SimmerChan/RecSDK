/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "file.h"

#include <string>

#include <sys/stat.h>

namespace rec_sdk {
namespace safety {

bool CheckFilePermission(const std::string& filePath)
{
    constexpr mode_t mask = 0777;
    constexpr mode_t reqPerm = 0644;

    struct stat fileStat;
    if (stat(filePath.c_str(), &fileStat) != 0) {
        return false;
    }

    auto perm = fileStat.st_mode & mask;
    if (perm != reqPerm) {
        return false;
    }

    return true;
}

}  // namespace safety
}  // namespace rec_sdk
