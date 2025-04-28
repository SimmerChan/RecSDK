/**
 * Copyright (C)  2022. Huawei Technologies Co., Ltd. All rights reserved.

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

#ifndef GE_OP_SPARSE_FW_FFM_PART2_H
#define GE_OP_SPARSE_FW_FFM_PART2_H
#include "graph/operator_reg.h"
namespace ge {

REG_OP(SparseFwFFMPart2)
    .INPUT(fw_weight, T)
    .INPUT(cross_mean_sum, T)
    .INPUT(cross_mean_square_sum, T)
    .INPUT(fw_field_map, TIndex)
    .DATATYPE(T, TensorType({DT_FLOAT}))
    .DATATYPE(TIndex, TensorType({DT_INT32}))
    .OUTPUT(output, T)
    .OP_END_FACTORY_REG(SparseFwFFMPart2)
}
#endif //GE_OP_SPARSE_FW_FFM_PART2_H
