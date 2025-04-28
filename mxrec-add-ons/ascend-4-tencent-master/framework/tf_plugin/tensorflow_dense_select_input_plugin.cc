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

namespace domi {
Status AutoMappingFnDenseSelectInput(const google::protobuf::Message* op_src, ge::Operator& op) {
  map<string, pair<string, string>> value;
  value["in"] = pair<string, string>("embeddings", "field_num");
  AutoMappingFnDynamic(op_src, op, value);
  return SUCCESS;
}

// register op info to GE
REGISTER_CUSTOM_OP("DenseSelectInput")
    .FrameworkType(TENSORFLOW)   // type: CAFFE, TENSORFLOW
    .OriginOpType("DenseSelectInput")      // name in tf module
    .ParseParamsFn(AutoMappingFnDenseSelectInput)
    .ImplyType(ImplyType::AI_CPU);
}  // namespace domi