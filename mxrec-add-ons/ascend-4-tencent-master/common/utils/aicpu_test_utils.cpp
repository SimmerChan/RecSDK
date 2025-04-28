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

#include "aicpu_test_utils.h"

using namespace std;

uint64_t CalTotalElements(std::vector<std::vector<int64_t>> &shapes,
                          uint32_t index) {
  if(index < 0) {
    return 0;
  }
  uint64_t nums = 1;
  for(auto shape : shapes[index]) {
    nums = nums * shape;
  }
  return nums;
}

template <>
bool CompareResult(float output[], float expect_output[], uint64_t num) {
  bool result = true;
  for (uint64_t i = 0; i < num; ++i) {
    double absolute_error = std::fabs(output[i] - expect_output[i]);
    double relative_error = 0;
    if (expect_output[i] == 0) {
      relative_error = 2e-6;
    } else {
      relative_error = absolute_error / std::fabs(expect_output[i]);
    }
    if ((absolute_error > 1e-6) && (relative_error > 1e-6)) {
      std::cout << "output[" << i << "] = ";
      std::cout << output[i];
      std::cout << ", expect_output[" << i << "] = ";
      std::cout << expect_output[i] << std::endl;
      result = false;
    }
  }
  return result;
}

template <>
bool CompareResult(double output[], double expect_output[], uint64_t num) {
  bool result = true;
  for (uint64_t i = 0; i < num; ++i) {
    double absolute_error = std::fabs(output[i] - expect_output[i]);
    double relative_error = 0;
    if (expect_output[i] == 0) {
      relative_error = 2e-10;
    } else {
      relative_error = absolute_error / std::fabs(expect_output[i]);
    }
    if ((absolute_error > 1e-12) && (relative_error > 1e-10)) {
      std::cout << "output[" << i << "] = ";
      std::cout << output[i];
      std::cout << ", expect_output[" << i << "] = ";
      std::cout << expect_output[i] << std::endl;
      result = false;
    }
  }
  return result;
}

template <>
bool CompareResult(complex<float> output[], complex<float> expect_output[], uint64_t num) {
  bool result = true;
  for (uint64_t i = 0; i < num; ++i) {
    double absolute_real_error = std::fabs(output[i].real() - expect_output[i].real());
    double relative_real_error = 0;
    if (expect_output[i].real() == 0) {
      relative_real_error = 2e-6;
    } else {
      relative_real_error = absolute_real_error / std::fabs(expect_output[i].real());
    }

    double absolute_imag_error = std::fabs(output[i].imag() - expect_output[i].imag());
    double relative_imag_error = 0;
    if (expect_output[i].imag() == 0) {
      relative_imag_error = 2e-6;
    } else {
      relative_imag_error = absolute_imag_error / std::fabs(expect_output[i].imag());
    }

    if (((absolute_real_error > 1e-6) && (relative_real_error > 1e-6)) ||
        ((absolute_imag_error > 1e-6) && (relative_imag_error > 1e-6))) {
      std::cout << "output[" << i << "] = ";
      std::cout << output[i];
      std::cout << ", expect_output[" << i << "] = ";
      std::cout << expect_output[i] << std::endl;
      result = false;
    }
  }
  return result;
}

template <>
bool CompareResult(complex<double> output[], complex<double> expect_output[], uint64_t num) {
  bool result = true;
  for (uint64_t i = 0; i < num; ++i) {
    double absolute_real_error = std::fabs(output[i].real() - expect_output[i].real());
    double relative_real_error = 0;
    if (expect_output[i].real() == 0) {
      relative_real_error = 2e-10;
    } else {
      relative_real_error = absolute_real_error / std::fabs(expect_output[i].real());
    }

    double absolute_imag_error = std::fabs(output[i].imag() - expect_output[i].imag());
    double relative_imag_error = 0;
    if (expect_output[i].imag() == 0) {
      relative_imag_error = 2e-10;
    } else {
      relative_imag_error = absolute_imag_error / std::fabs(expect_output[i].imag());
    }

    if (((absolute_real_error > 10e-8) && (relative_real_error > 10e-8)) ||
        ((absolute_imag_error > 10e-8) && (relative_imag_error > 10e-8))) {
      std::cout << "output[" << i << "] = ";
      std::cout << output[i];
      std::cout << ", expect_output[" << i << "] = ";
      std::cout << expect_output[i] << std::endl;
      result = false;
    }
  }
  return result;
}

bool CompareResult(Eigen::half output[], Eigen::half expect_output[],
    uint64_t num) {
  bool result = true;
  for (uint64_t i = 0; i < num; ++i) {
    Eigen::half absolute_error = (output[i] - expect_output[i]);
    absolute_error =
        absolute_error >= Eigen::half(0) ? absolute_error : -absolute_error;
    Eigen::half relative_error(0);
    if (expect_output[i] == Eigen::half(0)) {
      relative_error = Eigen::half(2e-3);
    } else {
      relative_error = absolute_error / expect_output[i];
      relative_error =
          relative_error >= Eigen::half(0) ? relative_error : -relative_error;
    }
    if ((absolute_error > Eigen::half(1e-3)) &&
        (relative_error > Eigen::half(1e-3))) {
      std::cout << "output[" << i << "] = ";
      std::cout << output[i];
      std::cout << ", expect_output[" << i << "] = ";
      std::cout << expect_output[i] << std::endl;
      result = false;
    }
  }
  return result;
}