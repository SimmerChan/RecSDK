/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * Copyright (c) huawei Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef BUCKETIZE_H
#define BUCKETIZE_H

#include <atomic>
#include <functional>
#include <torch/extension.h>

#include "common/common.h"

namespace Embcache {
std::tuple<at::Tensor, at::Tensor> ModBucketize(const at::Tensor& lengths, const at::Tensor& values,
                                                int64_t numBuckets);
}  // namespace Embcache
#endif  // BUCKETIZE_H
