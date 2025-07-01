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

#pragma once

#include <chrono>
#include <fstream>
#include <iostream>
#include <string>

#include <sys/stat.h>

#include "absl/container/flat_hash_map.h"
#include "spdlog/spdlog.h"

#include "safety/file.h"

namespace rec_sdk {
namespace feature {

template <typename K, typename V>
bool SaveFlatHashMapToBinaryFile(const absl::flat_hash_map<K, V>& map, const std::string& filePath)
{
    auto outputFile = std::ofstream(filePath, std::ios::binary);
    if (!outputFile.is_open()) {
        return false;
    }

    auto isPermValid = safety::CheckFilePermission(filePath);
    if (!isPermValid) {
        spdlog::error("The permission of file {} is invalid.", filePath);
        return false;
    }

    size_t len = map.size();
    outputFile.write(reinterpret_cast<const char*>(&len), sizeof(len));
    if (outputFile.fail()) {
        spdlog::error("Failed to write length ==> {} <== to file ==> {} <==.", len, filePath);
        return false;
    }

    for (const auto& [key, value] : map) {
        outputFile.write(reinterpret_cast<const char*>(&key), sizeof(K));
        if (outputFile.fail()) {
            spdlog::error("Failed to write key ==> {} <== to file ==> {} <==.", key, filePath);
            return false;
        }

        outputFile.write(reinterpret_cast<const char*>(&value), sizeof(V));
        if (outputFile.fail()) {
            spdlog::error("Failed to write value ==> {} <== to file ==> {} <==.", value, filePath);
            return false;
        }
    }

    outputFile.close();
    return true;
}

template <typename K, typename V>
bool LoadFlatHashMapFromBinaryFile(absl::flat_hash_map<K, V>& map, const std::string& filePath)
{
    auto inputFile = std::ifstream(filePath, std::ios::binary);
    if (!inputFile.is_open()) {
        return false;
    }

    auto isPermValid = safety::CheckFilePermission(filePath);
    if (!isPermValid) {
        spdlog::error("The permission of file {} is invalid.", filePath);
        return false;
    }

    size_t len = 0;
    inputFile.read(reinterpret_cast<char*>(&len), sizeof(len));
    if (inputFile.fail()) {
        spdlog::error("Failed to read length from file ==> {} <==.", filePath);
        return false;
    }

    map.clear();
    for (size_t i = 0; i < len; ++i) {
        auto key = K();
        auto value = V();

        inputFile.read(reinterpret_cast<char*>(&key), sizeof(K));
        if (inputFile.fail()) {
            spdlog::error("Failed to read key from file ==> {} <==.", filePath);
            return false;
        }

        inputFile.read(reinterpret_cast<char*>(&value), sizeof(V));
        if (inputFile.fail()) {
            spdlog::error("Failed to read value from file ==> {} <==.", filePath);
            return false;
        }

        map.emplace(key, value);
    }

    inputFile.close();
    return true;
}

template <typename K>
bool SaveFlatHashMapToBinaryFile(const absl::flat_hash_map<K, std::chrono::time_point<std::chrono::system_clock>>& map,
                                 const std::string& filePath)
{
    auto outputFile = std::ofstream(filePath, std::ios::binary);
    if (!outputFile.is_open()) {
        return false;
    }

    auto isPermValid = safety::CheckFilePermission(filePath);
    if (!isPermValid) {
        spdlog::error("The permission of file {} is invalid.", filePath);
        return false;
    }

    size_t len = map.size();
    outputFile.write(reinterpret_cast<const char*>(&len), sizeof(len));
    if (outputFile.fail()) {
        spdlog::error("Failed to write length ==> {} <== to file ==> {} <==.", len, filePath);
        return false;
    }

    for (const auto& [key, value] : map) {
        outputFile.write(reinterpret_cast<const char*>(&key), sizeof(K));
        if (outputFile.fail()) {
            spdlog::error("Failed to write key ==> {} <== to file ==> {} <==.", key, filePath);
            return false;
        }

        auto duration_since_epoch = value.time_since_epoch();
        auto count = std::chrono::duration_cast<std::chrono::milliseconds>(duration_since_epoch).count();
        outputFile.write(reinterpret_cast<const char*>(&count), sizeof(count));
        if (outputFile.fail()) {
            spdlog::error("Failed to write value ==> {} <== to file ==> {} <==.", count, filePath);
            return false;
        }
    }

    outputFile.close();
    return true;
}

template <typename K>
bool LoadFlatHashMapFromBinaryFile(absl::flat_hash_map<K, std::chrono::time_point<std::chrono::system_clock>>& map,
                                   const std::string& filePath)
{
    auto inputFile = std::ifstream(filePath, std::ios::binary);
    if (!inputFile.is_open()) {
        return false;
    }

    auto isPermValid = safety::CheckFilePermission(filePath);
    if (!isPermValid) {
        spdlog::error("The permission of file {} is invalid.", filePath);
        return false;
    }

    size_t len = 0;
    inputFile.read(reinterpret_cast<char*>(&len), sizeof(len));
    if (inputFile.fail()) {
        spdlog::error("Failed to read length from file ==> {} <==.", filePath);
        return false;
    }

    map.clear();
    for (size_t i = 0; i < len; ++i) {
        auto key = K();
        auto count = 0;

        inputFile.read(reinterpret_cast<char*>(&key), sizeof(K));
        if (inputFile.fail()) {
            spdlog::error("Failed to read key from file ==> {} <==.", filePath);
            return false;
        }

        inputFile.read(reinterpret_cast<char*>(&count), sizeof(count));
        if (inputFile.fail()) {
            spdlog::error("Failed to read value from file ==> {} <==.", filePath);
            return false;
        }

        auto value = std::chrono::time_point<std::chrono::system_clock>(std::chrono::milliseconds(count));
        map.emplace(key, value);
    }

    inputFile.close();
    return true;
}

}  // namespace util
}  // namespace rec_sdk
