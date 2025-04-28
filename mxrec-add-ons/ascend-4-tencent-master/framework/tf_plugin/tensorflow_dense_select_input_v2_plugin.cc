/* Copyright (C) 2020-2021. Huawei Technologies Co., Ltd. All
rights reserved.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0.
 * You may not use this file except in compliance with the License.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#include "register/register.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/operator.h"

namespace domi {
using namespace ge;

Status AutoMappingDenseSet(const Message* op_src, ge::Operator& op) {
    map<string, pair<string, string>> value;
    value["in"] = pair<string, string>("embeddings", "field_num");
    AutoMappingFnDynamic(op_src, op, value);
    return SUCCESS;
}

// register op info to GE
REGISTER_CUSTOM_OP("DenseSelectInputV2")
    .FrameworkType(TENSORFLOW)   // type: CAFFE, TENSORFLOW
    .OriginOpType("DenseSelectInputV2")      // name in tf module
    .ParseParamsFn(AutoMappingDenseSet)
    .ImplyType(ImplyType::AI_CPU);
}  // namespace domi
