/**
 * Copyright (C)  2020-2021. Huawei Technologies Co., Ltd. All rights reserved.

 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0.You may not use this file except in compliance with the License.

 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * @brief
 *
 * @version 1.0
 *
 */

#ifndef GE_OP_SPARSE_FW_FFM_H
#define GE_OP_SPARSE_FW_FFM_H
#include "graph/operator_reg.h"
namespace ge {

REG_OP(SparseFwFFM)
    .INPUT(weight, TensorType({DT_FLOAT16,DT_FLOAT,DT_DOUBLE}))
    .INPUT(fw_weight, TensorType({DT_FLOAT16,DT_FLOAT,DT_DOUBLE}))
    .INPUT(field, TensorType({DT_INT32}))
    .INPUT(index, TensorType({DT_INT32}))
    .OUTPUT(output, TensorType({DT_FLOAT16,DT_FLOAT,DT_DOUBLE}))
    .OUTPUT(cross_mean_sum, TensorType({DT_FLOAT16,DT_FLOAT,DT_DOUBLE}))
    .OUTPUT(cross_mean_square_sum, TensorType({DT_FLOAT16,DT_FLOAT,DT_DOUBLE}))
    .OUTPUT(fw_field_map, TensorType({DT_INT32}))
    .OP_END_FACTORY_REG(SparseFwFFM)
}
#endif //GE_OP_SPARSE_FW_FFM_H
