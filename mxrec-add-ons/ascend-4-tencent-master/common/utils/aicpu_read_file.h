/**
 * Copyright (c) 2023-2023 Huawei Technologies Co., Ltd.  All rights reserved.
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

#ifndef OPS_TEST_UTILS_AICPU_READ_FILE_H_
#define OPS_TEST_UTILS_AICPU_READ_FILE_H_
#include <iostream>
#include <string>
#include <fstream>
#include <exception>
#include <vector>
#include "Eigen/Core"

const std::string ktestcaseFilePath =
    "../community/tests/";

bool ReadFile(std::string file_name, Eigen::half output[], uint64_t size);

template<typename T>
bool ReadFile(std::string file_name, std::vector<T> &output) {
  try {
    std::ifstream in_file{file_name};
    if (!in_file.is_open()) {
      std::cout << "open file: " << file_name << " failed." << std::endl;
      return false;
    }
    T tmp;
    while (in_file >> tmp) {
      output.push_back(tmp);
    }
    in_file.close();
  } catch (std::exception &e) {
    std::cout << "read file " << file_name << " failed, "
              << e.what() << std::endl;
    return false;
  }
  return true;
}

template<typename T>
bool ReadFile(std::string file_name, T output[], uint64_t size) {
  try {
    std::ifstream in_file{file_name};
    if (!in_file.is_open()) {
      std::cout << "open file: " << file_name << " failed." << std::endl;
      return false;
    }
    T tmp;
    uint64_t index = 0;
    while (in_file >> tmp) {
      if (index >= size) {
        break;
      }
      output[index] = tmp;
      index++;
    }
    in_file.close();
  } catch (std::exception &e) {
    std::cout << "read file " << file_name << " failed, "
              << e.what() << std::endl;
    return false;
  }
  return true;
}

#endif