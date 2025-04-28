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

#ifndef GE_OP_SPARSE_FW_FFM_GRAD_H
#define GE_OP_SPARSE_FW_FFM_GRAD_H
#include "graph/operator_reg.h"
namespace ge {

REG_OP(SparseFwFFMGrad)
    .INPUT(weight, TensorType({DT_FLOAT,DT_DOUBLE}))
    .INPUT(fw_weight, TensorType({DT_FLOAT,DT_DOUBLE}))
    .INPUT(field, TensorType({DT_INT32}))
    .INPUT(index, TensorType({DT_INT32}))
    .INPUT(cross_mean_sum, TensorType({DT_FLOAT,DT_DOUBLE}))
    .INPUT(cross_mean_square_sum, TensorType({DT_FLOAT,DT_DOUBLE}))
    .INPUT(fw_field_map, TensorType({DT_INT32}))
    .INPUT(grad, TensorType({DT_FLOAT,DT_DOUBLE}))
    .OUTPUT(output, TensorType({DT_FLOAT,DT_DOUBLE}))
    .OUTPUT(fw_output, TensorType({DT_FLOAT,DT_DOUBLE}))
    .OP_END_FACTORY_REG(SparseFwFFMGrad)
}
#endif //GE_OP_SPARSE_FW_FFM_GRAD_H
