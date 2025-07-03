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

#include <array>
#include <cstdint>

#include "gtest/gtest.h"
#include "acl/acl_base.h"
#include "tensorflow/core/framework/tensor_shape.h"

#include "hdc/acl_channel.h"
#include "common/types.h"

namespace rec_sdk {
namespace hdc {

using std::array;
using common::i64;

TEST(Convert1DTest, DataItemToTensorInt64)
{
    constexpr auto dims = array<int64_t, 1>{7};
    auto nums = array<i64, 7>{1, 2, 3, 4, 5, 6, 7};
    auto dataItem =
        acltdtCreateDataItem(ACL_TENSOR_DATA_TENSOR, dims.data(), dims.size(), ACL_INT64, nums.data(), nums.size());

    auto tensor = AssembleDataItemToTensor(dataItem);
    auto tensorData = tensor.flat<i64>();

    ASSERT_EQ(tensor.dtype(), tensorflow::DT_INT64);
    ASSERT_EQ(tensor.dim_size(0), nums.size());

    for (size_t i = 0; i < nums.size(); ++i) {
        EXPECT_EQ(tensorData(i), nums[i]);
    }

    acltdtDestroyDataItem(dataItem);
}

TEST(Convert1DTest, TensorToDataItemInt64)
{
    auto tensorShape = tensorflow::TensorShape({7});
    auto tensor = tensorflow::Tensor(tensorflow::DT_INT64, tensorShape);
    auto tensorData = tensor.flat<i64>();

    auto nums = array<i64, 7>{1, 2, 3, 4, 5, 6, 7};
    for (size_t i = 0; i < nums.size(); ++i) {
        tensorData(i) = nums[i];
    }

    auto dataItem = AssembleTensorToDataItem(std::move(tensor));

    ASSERT_EQ(acltdtGetDataTypeFromItem(dataItem), ACL_INT64);
    ASSERT_EQ(acltdtGetDimNumFromItem(dataItem), tensor.dims());

    auto data = reinterpret_cast<i64*>(acltdtGetDataAddrFromItem(dataItem));
    for (size_t i = 0; i < nums.size(); ++i) {
        EXPECT_EQ(data[i], nums[i]);
    }

    acltdtDestroyDataItem(dataItem);
}

}  // namespace trans
}  // namespace rec_sdk
