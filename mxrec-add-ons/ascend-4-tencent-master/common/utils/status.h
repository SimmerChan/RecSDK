/**
 * Copyright (c) 2022-2022 Huawei Technologies Co., Ltd.  All rights reserved.
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

#ifndef KERNEL_STATUS_H
#define KERNEL_STATUS_H

#include <cstdint>

namespace aicpu {
using KernelStatus = uint32_t;
/*
 * status code
 */
// 0-3 is fixed error code, runtime need interprete 0-3 error codes
constexpr uint32_t KERNEL_STATUS_OK = 0U;
constexpr uint32_t KERNEL_STATUS_PARAM_INVALID = 1U;
constexpr uint32_t KERNEL_STATUS_INNER_ERROR = 2U;
constexpr uint32_t KERNEL_STATUS_TIMEOUT = 3U;
constexpr uint32_t KERNEL_STATUS_PROTOBUF_ERROR = 4U;
constexpr uint32_t KERNEL_STATUS_SHARDER_ERROR = 5U;
constexpr uint32_t KERNEL_STATUS_END_OF_SEQUENCE = 201U;
}  // namespace aicpu
#endif  // KERNEL_STATUS_H
