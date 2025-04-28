/*
 * @Author: q00475944 qianzehong@huawei.com
 * @Date: 2022-12-24 15:29:56
 * @LastEditors: q00475944 qianzehong@huawei.com
 * @LastEditTime: 2023-01-05 09:13:11
 * @FilePath: /ascend-4-tencent/framework/tf_plugin/tensorflow_sparse_fw_ffm_grad_part2_plugin.cc
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 */
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
// register op info to GE
REGISTER_CUSTOM_OP("SparseFwFFMPart2Grad")
    .FrameworkType(TENSORFLOW)   // type: CAFFE, TENSORFLOW
    .OriginOpType("SparseFwFFMPart2Grad")      // name in tf module
    .ParseParamsByOperatorFn(AutoMappingByOpFn);
}  // namespace domi
