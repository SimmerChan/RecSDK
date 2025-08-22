/**
 * @file expand_into_jagged_permute.cpp
 *
 * Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>

#include "../common/pytorch_npu_helper.hpp"
using torch::autograd::AutogradContext;
using torch::autograd::Function;
using torch::autograd::Variable;
using tensor_list = std::vector<at::Tensor>;
using namespace at;

