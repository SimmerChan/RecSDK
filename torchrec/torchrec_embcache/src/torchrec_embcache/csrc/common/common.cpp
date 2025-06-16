/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * Copyright (c) huawei Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include "common/common.h"

namespace Embcache {

const char* FVK_STATE_STR[] = {
    "FKV_EXIST",    "FKV_NOT_EXIST", "FKV_KEY_CONFLICT", "FKV_BEFORE_PUT_FUNC_FAIL", "FKV_BEFORE_REMOVE_FUNC_FAIL",
    "FKV_NO_SPACE", "FKV_FAIL"};

}
