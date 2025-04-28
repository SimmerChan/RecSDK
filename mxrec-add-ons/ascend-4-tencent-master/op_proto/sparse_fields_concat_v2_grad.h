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

#ifndef GE_OP_SPARSE_FIELDS_CONCAT_V2_GRAD_H
#define GE_OP_SPARSE_FIELDS_CONCAT_V2_GRAD_H
#include "graph/operator_reg.h"
namespace ge {

REG_OP(SparseFieldsConcatV2Grad)
    .INPUT(weight, TensorType({DT_FLOAT16,DT_FLOAT,DT_DOUBLE}))
    .INPUT(field, TensorType({DT_INT32}))
    .INPUT(index, TensorType({DT_INT32}))
    .INPUT(grad_part1, TensorType({DT_FLOAT16,DT_FLOAT,DT_DOUBLE}))
    .INPUT(grad_part2, TensorType({DT_FLOAT16,DT_FLOAT,DT_DOUBLE}))
    .INPUT(keys_per_field, TensorType({DT_INT32}))
    .OUTPUT(output, TensorType({DT_FLOAT16,DT_FLOAT,DT_DOUBLE}))
    .ATTR(fw_field_num, Int, 1)
    .REQUIRED_ATTR(part1_fields, ListInt)
    .REQUIRED_ATTR(part2_fields, ListInt)
    .OP_END_FACTORY_REG(SparseFieldsConcatV2Grad)
}
#endif //GE_OP_SPARSE_FIELDS_CONCAT_V2_GRAD_H
