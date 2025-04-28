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

/*!
 * \file tiling_util.cc
 * \brief
 */
#include "tiling_rt2_util.h"

namespace optiling {
bool AddWorkspace(gert::TilingContext* context, const size_t workspace) {
  size_t* workspace_size = context->GetWorkspaceSizes(1);
  OPS_CHECK_NULL_WITH_CONTEXT_RET(context, workspace_size, false);
  *workspace_size = workspace;
  return true;
}

int64_t GetPartShapeSize(const gert::Shape& shape, size_t begin, size_t end) {
  int64_t size = 1;
  for (size_t i = begin; i < end; i++) {
    size *= shape[i];
  }
  return size;
}

int64_t CeilAlign(int64_t u_value, int64_t d_value) {
  int64_t res_value = 0;
  if (d_value == 0) {
    return u_value;
  }
  res_value = (u_value + d_value - 1) / d_value * d_value;

  return res_value;
}

int64_t GetRemainder(int64_t u_value, int64_t d_value) {
  int64_t res_value = 0;
  if (d_value == 0) {
    return u_value;
  }
  res_value = u_value % d_value;

  return res_value;
}
}  // namespace optiling
