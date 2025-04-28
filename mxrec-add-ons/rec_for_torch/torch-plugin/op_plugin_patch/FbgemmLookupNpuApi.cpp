/**
 * @file FbgemmLookupNpuApi.cpp
 *
 * Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 */

#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"
#include "torch/extension.h"

#define DISPATCH_TO_NPU(name, function) m.impl(name, torch::dispatch(c10::DispatchKey::PrivateUse1, TORCH_FN(function)))
// optimize type
constexpr int ADAGRAD = 1;
constexpr int ADAM = 2;
constexpr int SGD = 3;

namespace fbgemm_npu_lookups {

at::Tensor split_embedding_backward_codegen_adagrad_unweighted_exact_cuda(
    const at::Tensor& grad_output, const at::Tensor& dev_weights, const at::Tensor& uvm_weights,
    const at::Tensor& lxu_cache_weights, const at::Tensor& weights_placements, const at::Tensor& weights_offsets,
    const at::Tensor& D_offsets, const int64_t max_D, const at::Tensor& hash_size_cumsum,
    const int64_t total_hash_size_bits, const at::Tensor& indices, const at::Tensor& offsets,
    const int64_t pooling_mode, const at::Tensor& lxu_cache_locations, const int64_t BT_block_size,
    const int64_t max_segment_length_per_warp, const bool stochastic_rounding, const int64_t info_B_num_bits,
    const int64_t info_B_mask_int64, const bool use_uniq_cache_locations, const bool use_homogeneous_placements,
    at::Tensor momentum1_dev, at::Tensor momentum1_uvm, at::Tensor momentum1_placements, at::Tensor momentum1_offsets,
    const at::Tensor& hash_indices, const at::Tensor& unique_ids, const at::Tensor& unique_offsets,
    const at::Tensor& unique_inverse, double eps = 0, double learning_rate = 0);

at::Tensor split_embedding_backward_codegen_adam_unweighted_exact_cuda(
    const at::Tensor& grad_output, const at::Tensor& dev_weights, const at::Tensor& uvm_weights,
    const at::Tensor& lxu_cache_weights, const at::Tensor& weights_placements, const at::Tensor& weights_offsets,
    const at::Tensor& D_offsets, const int64_t max_D, const at::Tensor& hash_size_cumsum,
    const int64_t total_hash_size_bits, const at::Tensor& indices, const at::Tensor& offsets,
    const int64_t pooling_mode, const at::Tensor& lxu_cache_locations, const int64_t BT_block_size,
    const int64_t max_segment_length_per_warp, const bool stochastic_rounding, const int64_t info_B_num_bits,
    const int64_t info_B_mask_int64, const bool use_uniq_cache_locations, const bool use_homogeneous_placements,
    at::Tensor momentum1_dev, at::Tensor momentum1_uvm, at::Tensor momentum1_placements, at::Tensor momentum1_offsets,
    at::Tensor momentum2_dev, at::Tensor momentum2_uvm, at::Tensor momentum2_placements, at::Tensor momentum2_offsets,
    const at::Tensor& hash_indices, const at::Tensor& unique_ids, const at::Tensor& unique_offsets,
    const at::Tensor& unique_inverse, double eps = 0, double learning_rate = 0, double beta1 = 0.9,
    double beta2 = 0.999, int64_t iter = 0);

at::Tensor split_embedding_backward_codegen_sgd_unweighted_exact_cuda(
    const at::Tensor& grad_output, const at::Tensor& dev_weights, const at::Tensor& uvm_weights,
    const at::Tensor& lxu_cache_weights, const at::Tensor& weights_placements, const at::Tensor& weights_offsets,
    const at::Tensor& D_offsets, const int64_t max_D, const at::Tensor& hash_size_cumsum,
    const int64_t total_hash_size_bits, const at::Tensor& indices, const at::Tensor& offsets,
    const int64_t pooling_mode, const at::Tensor& lxu_cache_locations, const int64_t BT_block_size,
    const int64_t max_segment_length_per_warp, const bool stochastic_rounding, const int64_t info_B_num_bits,
    const int64_t info_B_mask_int64, const bool use_uniq_cache_locations, const bool use_homogeneous_placements,
    const at::Tensor& hash_indices, const at::Tensor& unique_ids, const at::Tensor& unique_offsets,
    const at::Tensor& unique_inverse, double learning_rate = 0);

at::Tensor split_embedding_codegen_forward_unweighted_cuda(
    const at::Tensor& dev_weights, const at::Tensor& uvm_weights, const at::Tensor& lxu_cache_weights,
    const at::Tensor& weights_placements, const at::Tensor& weights_offsets, const at::Tensor& D_offsets,
    const int64_t total_D, const int64_t max_D, const at::Tensor& indices, const at::Tensor& offsets,
    const int64_t pooling_mode, const at::Tensor& lxu_cache_locations, const int64_t output_dtype,
    const bool is_experimental, const at::Tensor& hash_indices);

enum class SparseType : uint8_t {
    FP32 = 0,
    FP16 = 1,
    INT8 = 2,
    INT4 = 3,
    INT2 = 4,
    BF16 = 5,
    FP8 = 6,
    INVALID = 7,
};

enum class PoolingMode : uint8_t {
    SUM = 0,
    MEAN = 1,
    NONE = 2
};

class SplitLookupFunction_adagrad_Op : public torch::autograd::Function<SplitLookupFunction_adagrad_Op> {
public:
    static torch::autograd::variable_list forward(
        torch::autograd::AutogradContext* ctx, const at::Tensor& placeholder_autograd_tensor,
        const int64_t output_dtype, const at::Tensor& dev_weights, const at::Tensor& uvm_weights,
        const at::Tensor& lxu_cache_weights, const at::Tensor& weights_placements, const at::Tensor& weights_offsets,
        const at::Tensor& D_offsets, const int64_t total_D, const int64_t max_D, const at::Tensor& hash_size_cumsum,
        const int64_t total_hash_size_bits, const at::Tensor& indices, const c10::optional<at::Tensor>& hash_indices,
        const c10::optional<at::Tensor>& unique_ids, const c10::optional<at::Tensor>& unique_offsets,
        const c10::optional<at::Tensor>& unique_inverse, const at::Tensor& offsets, const int64_t pooling_mode,
        const c10::optional<at::Tensor>& indice_weights, const c10::optional<at::Tensor>& feature_requires_grad,
        const at::Tensor& lxu_cache_locations, const bool gradient_clipping, const double max_gradient,
        const bool stochastic_rounding, const bool is_experimental, const bool use_uniq_cache_locations_bwd,
        const bool use_homogeneous_placements, at::Tensor momentum1_dev, at::Tensor momentum1_uvm,
        at::Tensor momentum1_placements, at::Tensor momentum1_offsets, double eps = 0, double learning_rate = 0)
    {
        const auto T = weights_offsets.size(0);
        if (T == 0) {
            return {at::Tensor()};
        }
        const auto max_B_ = offsets.size(0) / T;

        auto info_B_num_bits = max_B_;
        auto info_B_mask = T;

        ctx->save_for_backward({dev_weights,
                                uvm_weights,
                                lxu_cache_weights,
                                weights_placements,
                                weights_offsets,
                                D_offsets,
                                hash_size_cumsum,
                                indices,
                                offsets,
                                indice_weights.value_or(at::Tensor()),
                                feature_requires_grad.value_or(at::Tensor()),
                                lxu_cache_locations,
                                momentum1_dev,
                                momentum1_uvm,
                                momentum1_placements,
                                momentum1_offsets,
                                hash_indices.value_or(at::Tensor()),
                                unique_ids.value_or(at::Tensor()),
                                unique_offsets.value_or(at::Tensor()),
                                unique_inverse.value_or(at::Tensor())});
        ctx->saved_data["max_D"] = max_D;
        ctx->saved_data["pooling_mode"] = pooling_mode;
        ctx->saved_data["total_hash_size_bits"] = total_hash_size_bits;
        ctx->saved_data["gradient_clipping"] = gradient_clipping;
        ctx->saved_data["max_gradient"] = max_gradient;
        ctx->saved_data["stochastic_rounding"] = stochastic_rounding;
        ctx->saved_data["info_B_num_bits"] = info_B_num_bits;
        const auto info_B_mask_int64 = static_cast<int64_t>(info_B_mask);
        ctx->saved_data["info_B_mask"] = info_B_mask_int64;
        ctx->saved_data["use_uniq_cache_locations_bwd"] = use_uniq_cache_locations_bwd;
        ctx->saved_data["use_homogeneous_placements"] = use_homogeneous_placements;
        ctx->saved_data["eps"] = eps;
        ctx->saved_data["learning_rate"] = learning_rate;
        const auto& flatten_dev_weights = dev_weights;
        if (!indice_weights) {
            static auto split_embedding_codegen_forward_op =
                torch::Dispatcher::singleton()
                    .findSchemaOrThrow("fbgemm::split_embedding_codegen_forward_unweighted_cuda", "")
                    .typed<decltype(split_embedding_codegen_forward_unweighted_cuda)>();
            return {split_embedding_codegen_forward_op.call(
                flatten_dev_weights, uvm_weights, lxu_cache_weights, weights_placements, weights_offsets, D_offsets,
                total_D, max_D, indices, offsets, pooling_mode, lxu_cache_locations, output_dtype, is_experimental,
                hash_indices.value_or(at::Tensor()))};
        }
        return {at::Tensor()};
    }

    static torch::autograd::variable_list backward(torch::autograd::AutogradContext* ctx,
                                                   torch::autograd::variable_list grad_outputs)
    {
        const auto saved = ctx->get_saved_variables();
        auto savedItr = std::begin(saved);
        auto dev_weights = *savedItr++;
        auto uvm_weights = *savedItr++;
        auto lxu_cache_weights = *savedItr++;
        auto weights_placements = *savedItr++;
        auto weights_offsets = *savedItr++;
        auto D_offsets = *savedItr++;
        auto hash_size_cumsum = *savedItr++;
        auto indices = *savedItr++;
        auto offsets = *savedItr++;
        auto indice_weights = *savedItr++;
        auto feature_requires_grad = *savedItr++;
        auto lxu_cache_locations = *savedItr++;
        auto momentum1_dev = *savedItr++;
        auto momentum1_uvm = *savedItr++;
        auto momentum1_placements = *savedItr++;
        auto momentum1_offsets = *savedItr++;
        auto hash_indices = *savedItr++;
        auto unique_ids = *savedItr++;
        auto unique_offsets = *savedItr++;
        auto unique_inverse = *savedItr++;
        auto max_D = ctx->saved_data["max_D"].toInt();
        auto pooling_mode = ctx->saved_data["pooling_mode"].toInt();
        auto total_hash_size_bits = ctx->saved_data["total_hash_size_bits"].toInt();
        auto gradient_clipping = ctx->saved_data["gradient_clipping"].toBool();
        auto max_gradient = ctx->saved_data["max_gradient"].toDouble();
        auto stochastic_rounding = ctx->saved_data["stochastic_rounding"].toBool();
        const int32_t info_B_num_bits = ctx->saved_data["info_B_num_bits"].toInt();
        const int64_t info_B_mask_int64 = ctx->saved_data["info_B_mask"].toInt();
        const auto use_uniq_cache_locations_bwd = ctx->saved_data["use_uniq_cache_locations_bwd"].toBool();
        const auto use_homogeneous_placements = ctx->saved_data["use_homogeneous_placements"].toBool();
        auto eps = ctx->saved_data["eps"].toDouble();
        auto learning_rate = ctx->saved_data["learning_rate"].toDouble();

        TORCH_CHECK_EQ(grad_outputs.size(), 1);

        constexpr int32_t BT_block_size = 32;
        constexpr int32_t max_segment_length_per_warp = 32;

        using torch::autograd::Variable;
        auto grad_output = gradient_clipping ? clamp(grad_outputs[0], -max_gradient, max_gradient) : grad_outputs[0];

        static auto split_embedding_codegen_unweighted_backward_op =
            torch::Dispatcher::singleton()
                .findSchemaOrThrow("fbgemm::split_embedding_backward_codegen_adagrad_unweighted_exact_cuda", "")
                .typed<decltype(split_embedding_backward_codegen_adagrad_unweighted_exact_cuda)>();

        const auto grad_dev_weights = split_embedding_codegen_unweighted_backward_op.call(
            grad_output, dev_weights, uvm_weights, lxu_cache_weights, weights_placements, weights_offsets, D_offsets,
            max_D, hash_size_cumsum, total_hash_size_bits, indices, offsets, pooling_mode, lxu_cache_locations,
            BT_block_size, max_segment_length_per_warp, stochastic_rounding, info_B_num_bits, info_B_mask_int64,
            use_uniq_cache_locations_bwd, use_homogeneous_placements, momentum1_dev, momentum1_uvm,
            momentum1_placements, momentum1_offsets, hash_indices, unique_ids, unique_offsets, unique_inverse, eps,
            learning_rate);
        return {at::Tensor(),      // placeholder autograd tensor
                Variable(),        // output_dtype
                grad_dev_weights,  // dev_weights
                Variable(),        // uvm_weights
                Variable(),        // lxu_cache_weights
                Variable(),        // weights_placements
                Variable(),        // weights_offsets
                Variable(),        // D_offsets
                Variable(),        // total_D
                Variable(),        // max_D
                Variable(),        // hash_size_cumsum
                Variable(),        // total_hash_size_bits
                Variable(),        // indices
                Variable(),        // offsets
                Variable(),        // pooling_mode
                Variable(),        // indice_weights
                Variable(),        // feature_requires_grad
                Variable(),        // lxu_cache_locations
                Variable(),        // gradient_clipping
                Variable(),        // max_gradient
                Variable(),        // stochastic_rounding
                Variable(),        // is_experimental
                Variable(),        // use_uniq_cache_locations_bwd
                Variable(),        // use_homogeneous_placements
                Variable(),        // momentum1_dev
                Variable(),        // momentum1_uvm
                Variable(),        // momentum1_placements
                Variable(),        // momentum1_offsets
                Variable(),        // hash_indices
                Variable(),        // unique_ids
                Variable(),        // unique_offsets
                Variable(),        // unique_inverse
                Variable(),        // eps
                Variable()};       // learning_rate
    }
};

class SplitLookupFunction_adam_Op : public torch::autograd::Function<SplitLookupFunction_adam_Op> {
public:
    static torch::autograd::variable_list forward(
        torch::autograd::AutogradContext* ctx, const at::Tensor& placeholder_autograd_tensor,
        const int64_t output_dtype, const at::Tensor& dev_weights, const at::Tensor& uvm_weights,
        const at::Tensor& lxu_cache_weights, const at::Tensor& weights_placements, const at::Tensor& weights_offsets,
        const at::Tensor& D_offsets, const int64_t total_D, const int64_t max_D, const at::Tensor& hash_size_cumsum,
        const int64_t total_hash_size_bits, const at::Tensor& indices, const c10::optional<at::Tensor>& hash_indices,
        const c10::optional<at::Tensor>& unique_ids, const c10::optional<at::Tensor>& unique_offsets,
        const c10::optional<at::Tensor>& unique_inverse, const at::Tensor& offsets, const int64_t pooling_mode,
        const c10::optional<at::Tensor>& indice_weights, const c10::optional<at::Tensor>& feature_requires_grad,
        const at::Tensor& lxu_cache_locations, const bool gradient_clipping, const double max_gradient,
        const bool stochastic_rounding, const bool is_experimental, const bool use_uniq_cache_locations_bwd,
        const bool use_homogeneous_placements, at::Tensor momentum1_dev, at::Tensor momentum1_uvm,
        at::Tensor momentum1_placements, at::Tensor momentum1_offsets, at::Tensor momentum2_dev,
        at::Tensor momentum2_uvm, at::Tensor momentum2_placements, at::Tensor momentum2_offsets, double eps = 0,
        double learning_rate = 0, double beta1 = 0.9, double beta2 = 0.999, int64_t iter = 0)
    {
        const auto T = weights_offsets.size(0);
        if (T == 0) {
            return {at::Tensor()};
        }

        const auto max_B_ = offsets.size(0) / T;

        auto info_B_num_bits = max_B_;
        auto info_B_mask = T;

        ctx->save_for_backward({dev_weights,
                                uvm_weights,
                                lxu_cache_weights,
                                weights_placements,
                                weights_offsets,
                                D_offsets,
                                hash_size_cumsum,
                                indices,
                                offsets,
                                indice_weights.value_or(at::Tensor()),
                                feature_requires_grad.value_or(at::Tensor()),
                                lxu_cache_locations,
                                momentum1_dev,
                                momentum1_uvm,
                                momentum1_placements,
                                momentum1_offsets,
                                momentum2_dev,
                                momentum2_uvm,
                                momentum2_placements,
                                momentum2_offsets,
                                hash_indices.value_or(at::Tensor()),
                                unique_ids.value_or(at::Tensor()),
                                unique_offsets.value_or(at::Tensor()),
                                unique_inverse.value_or(at::Tensor())});
        ctx->saved_data["max_D"] = max_D;
        ctx->saved_data["pooling_mode"] = pooling_mode;
        ctx->saved_data["total_hash_size_bits"] = total_hash_size_bits;
        ctx->saved_data["gradient_clipping"] = gradient_clipping;
        ctx->saved_data["max_gradient"] = max_gradient;
        ctx->saved_data["stochastic_rounding"] = stochastic_rounding;
        ctx->saved_data["info_B_num_bits"] = info_B_num_bits;
        const auto info_B_mask_int64 = static_cast<int64_t>(info_B_mask);
        ctx->saved_data["info_B_mask"] = info_B_mask_int64;
        ctx->saved_data["use_uniq_cache_locations_bwd"] = use_uniq_cache_locations_bwd;
        ctx->saved_data["use_homogeneous_placements"] = use_homogeneous_placements;
        ctx->saved_data["eps"] = eps;
        ctx->saved_data["learning_rate"] = learning_rate;
        ctx->saved_data["beta1"] = beta1;
        ctx->saved_data["beta2"] = beta2;
        ctx->saved_data["iter"] = iter;

        const auto& flatten_dev_weights = dev_weights;
        if (!indice_weights) {
            static auto split_embedding_codegen_forward_op =
                torch::Dispatcher::singleton()
                    .findSchemaOrThrow("fbgemm::split_embedding_codegen_forward_unweighted_cuda", "")
                    .typed<decltype(split_embedding_codegen_forward_unweighted_cuda)>();

            return {split_embedding_codegen_forward_op.call(
                flatten_dev_weights, uvm_weights, lxu_cache_weights, weights_placements, weights_offsets, D_offsets,
                total_D, max_D, indices, offsets, pooling_mode, lxu_cache_locations, output_dtype, is_experimental,
                hash_indices.value_or(at::Tensor()))};
        }
        return {at::Tensor()};
    }

    static torch::autograd::variable_list backward(torch::autograd::AutogradContext* ctx,
                                                   torch::autograd::variable_list grad_outputs)
    {
        const auto saved = ctx->get_saved_variables();
        auto savedItr = std::begin(saved);
        auto dev_weights = *savedItr++;
        auto uvm_weights = *savedItr++;
        auto lxu_cache_weights = *savedItr++;
        auto weights_placements = *savedItr++;
        auto weights_offsets = *savedItr++;
        auto D_offsets = *savedItr++;
        auto hash_size_cumsum = *savedItr++;
        auto indices = *savedItr++;
        auto offsets = *savedItr++;
        auto indice_weights = *savedItr++;
        auto feature_requires_grad = *savedItr++;
        auto lxu_cache_locations = *savedItr++;
        auto momentum1_dev = *savedItr++;
        auto momentum1_uvm = *savedItr++;
        auto momentum1_placements = *savedItr++;
        auto momentum1_offsets = *savedItr++;
        auto momentum2_dev = *savedItr++;
        auto momentum2_uvm = *savedItr++;
        auto momentum2_placements = *savedItr++;
        auto momentum2_offsets = *savedItr++;
        auto hash_indices = *savedItr++;
        auto unique_ids = *savedItr++;
        auto unique_offsets = *savedItr++;
        auto unique_inverse = *savedItr++;
        auto max_D = ctx->saved_data["max_D"].toInt();
        auto pooling_mode = ctx->saved_data["pooling_mode"].toInt();
        auto total_hash_size_bits = ctx->saved_data["total_hash_size_bits"].toInt();
        auto gradient_clipping = ctx->saved_data["gradient_clipping"].toBool();
        auto max_gradient = ctx->saved_data["max_gradient"].toDouble();
        auto stochastic_rounding = ctx->saved_data["stochastic_rounding"].toBool();
        const int32_t info_B_num_bits = ctx->saved_data["info_B_num_bits"].toInt();
        const int64_t info_B_mask_int64 = ctx->saved_data["info_B_mask"].toInt();
        const auto use_uniq_cache_locations_bwd = ctx->saved_data["use_uniq_cache_locations_bwd"].toBool();
        const auto use_homogeneous_placements = ctx->saved_data["use_homogeneous_placements"].toBool();
        auto eps = ctx->saved_data["eps"].toDouble();
        auto learning_rate = ctx->saved_data["learning_rate"].toDouble();
        auto beta1 = ctx->saved_data["beta1"].toDouble();
        auto beta2 = ctx->saved_data["beta2"].toDouble();
        auto iter = ctx->saved_data["iter"].toInt();

        TORCH_CHECK_EQ(grad_outputs.size(), 1);

        constexpr int32_t BT_block_size = 32;
        constexpr int32_t max_segment_length_per_warp = 32;

        using torch::autograd::Variable;
        auto grad_output = gradient_clipping ? clamp(grad_outputs[0], -max_gradient, max_gradient) : grad_outputs[0];

        static auto split_embedding_codegen_unweighted_backward_op =
            torch::Dispatcher::singleton()
                .findSchemaOrThrow("fbgemm::split_embedding_backward_codegen_adam_unweighted_exact_cuda", "")
                .typed<decltype(split_embedding_backward_codegen_adam_unweighted_exact_cuda)>();

        const auto grad_dev_weights = split_embedding_codegen_unweighted_backward_op.call(
            grad_output, dev_weights, uvm_weights, lxu_cache_weights, weights_placements, weights_offsets, D_offsets,
            max_D, hash_size_cumsum, total_hash_size_bits, indices, offsets, pooling_mode, lxu_cache_locations,
            BT_block_size, max_segment_length_per_warp, stochastic_rounding, info_B_num_bits, info_B_mask_int64,
            use_uniq_cache_locations_bwd, use_homogeneous_placements, momentum1_dev, momentum1_uvm,
            momentum1_placements, momentum1_offsets, momentum2_dev, momentum2_uvm, momentum2_placements,
            momentum2_offsets, hash_indices, unique_ids, unique_offsets, unique_inverse, eps, learning_rate, beta1,
            beta2, iter);
        return {
            at::Tensor(),      // placeholder autograd tensor
            Variable(),        // output_dtype
            grad_dev_weights,  // dev_weights
            Variable(),        // uvm_weights
            Variable(),        // lxu_cache_weights
            Variable(),        // weights_placements
            Variable(),        // weights_offsets
            Variable(),        // D_offsets
            Variable(),        // total_D
            Variable(),        // max_D
            Variable(),        // hash_size_cumsum
            Variable(),        // total_hash_size_bits
            Variable(),        // indices
            Variable(),        // offsets
            Variable(),        // pooling_mode
            Variable(),        // indice_weights
            Variable(),        // feature_requires_grad
            Variable(),        // lxu_cache_locations
            Variable(),        // gradient_clipping
            Variable(),        // max_gradient
            Variable(),        // stochastic_rounding
            Variable(),        // is_experimental
            Variable(),        // use_uniq_cache_locations_bwd
            Variable(),        // use_homogeneous_placements
            Variable(),        // momentum1_dev
            Variable(),        // momentum1_uvm
            Variable(),        // momentum1_placements
            Variable(),        // momentum1_offsets
            Variable(),        // momentum2_dev
            Variable(),        // momentum2_uvm
            Variable(),        // momentum2_placements
            Variable(),        // momentum2_offsets
            Variable(),        // hash_indices
            Variable(),        // unique_ids
            Variable(),        // unique_offsets
            Variable(),        // unique_inverse
            Variable(),        // eps
            Variable(),        // learning_rate
            Variable(),        // beta1
            Variable(),        // beta2
            Variable(),        // iter
        };
    }
};

class SplitLookupFunction_sgd_Op : public torch::autograd::Function<SplitLookupFunction_sgd_Op> {
public:
    static torch::autograd::variable_list forward(
        torch::autograd::AutogradContext* ctx, const at::Tensor& placeholder_autograd_tensor,
        const int64_t output_dtype, const at::Tensor& dev_weights, const at::Tensor& uvm_weights,
        const at::Tensor& lxu_cache_weights, const at::Tensor& weights_placements, const at::Tensor& weights_offsets,
        const at::Tensor& D_offsets, const int64_t total_D, const int64_t max_D, const at::Tensor& hash_size_cumsum,
        const int64_t total_hash_size_bits, const at::Tensor& indices, const c10::optional<at::Tensor>& hash_indices,
        const c10::optional<at::Tensor>& unique_ids, const c10::optional<at::Tensor>& unique_offsets,
        const c10::optional<at::Tensor>& unique_inverse, const at::Tensor& offsets, const int64_t pooling_mode,
        const c10::optional<at::Tensor>& indice_weights, const c10::optional<at::Tensor>& feature_requires_grad,
        const at::Tensor& lxu_cache_locations, const bool gradient_clipping, const double max_gradient,
        const bool stochastic_rounding, const bool is_experimental, const bool use_uniq_cache_locations_bwd,
        const bool use_homogeneous_placements, double learning_rate = 0)
    {
        const auto T = weights_offsets.size(0);
        if (T == 0) {
            return {at::Tensor()};
        }
        const auto max_B_ = offsets.size(0) / T;

        auto info_B_num_bits = max_B_;
        auto info_B_mask = T;

        ctx->save_for_backward({dev_weights, uvm_weights, lxu_cache_weights, weights_placements, weights_offsets,
                                D_offsets, hash_size_cumsum, indices, offsets, indice_weights.value_or(at::Tensor()),
                                feature_requires_grad.value_or(at::Tensor()), lxu_cache_locations,
                                hash_indices.value_or(at::Tensor()), unique_ids.value_or(at::Tensor()),
                                unique_offsets.value_or(at::Tensor()), unique_inverse.value_or(at::Tensor())});
        ctx->saved_data["max_D"] = max_D;
        ctx->saved_data["pooling_mode"] = pooling_mode;
        ctx->saved_data["total_hash_size_bits"] = total_hash_size_bits;
        ctx->saved_data["gradient_clipping"] = gradient_clipping;
        ctx->saved_data["max_gradient"] = max_gradient;
        ctx->saved_data["stochastic_rounding"] = stochastic_rounding;
        ctx->saved_data["info_B_num_bits"] = info_B_num_bits;
        const auto info_B_mask_int64 = static_cast<int64_t>(info_B_mask);
        ctx->saved_data["info_B_mask"] = info_B_mask_int64;
        ctx->saved_data["use_uniq_cache_locations_bwd"] = use_uniq_cache_locations_bwd;
        ctx->saved_data["use_homogeneous_placements"] = use_homogeneous_placements;
        ctx->saved_data["learning_rate"] = learning_rate;
        const auto& flatten_dev_weights = dev_weights;
        if (!indice_weights) {
            static auto split_embedding_codegen_forward_op =
                torch::Dispatcher::singleton()
                    .findSchemaOrThrow("fbgemm::split_embedding_codegen_forward_unweighted_cuda", "")
                    .typed<decltype(split_embedding_codegen_forward_unweighted_cuda)>();
            return {split_embedding_codegen_forward_op.call(
                flatten_dev_weights, uvm_weights, lxu_cache_weights, weights_placements, weights_offsets, D_offsets,
                total_D, max_D, indices, offsets, pooling_mode, lxu_cache_locations, output_dtype, is_experimental,
                hash_indices.value_or(at::Tensor()))};
        }
        return {at::Tensor()};
    }

    static torch::autograd::variable_list backward(torch::autograd::AutogradContext* ctx,
                                                   torch::autograd::variable_list grad_outputs)
    {
        const auto saved = ctx->get_saved_variables();
        auto savedItr = std::begin(saved);
        auto dev_weights = *savedItr++;
        auto uvm_weights = *savedItr++;
        auto lxu_cache_weights = *savedItr++;
        auto weights_placements = *savedItr++;
        auto weights_offsets = *savedItr++;
        auto D_offsets = *savedItr++;
        auto hash_size_cumsum = *savedItr++;
        auto indices = *savedItr++;
        auto offsets = *savedItr++;
        auto indice_weights = *savedItr++;
        auto feature_requires_grad = *savedItr++;
        auto lxu_cache_locations = *savedItr++;
        auto hash_indices = *savedItr++;
        auto unique_ids = *savedItr++;
        auto unique_offsets = *savedItr++;
        auto unique_inverse = *savedItr++;
        auto max_D = ctx->saved_data["max_D"].toInt();
        auto pooling_mode = ctx->saved_data["pooling_mode"].toInt();
        auto total_hash_size_bits = ctx->saved_data["total_hash_size_bits"].toInt();
        auto gradient_clipping = ctx->saved_data["gradient_clipping"].toBool();
        auto max_gradient = ctx->saved_data["max_gradient"].toDouble();
        auto stochastic_rounding = ctx->saved_data["stochastic_rounding"].toBool();
        const int32_t info_B_num_bits = ctx->saved_data["info_B_num_bits"].toInt();
        const int64_t info_B_mask_int64 = ctx->saved_data["info_B_mask"].toInt();
        const auto use_uniq_cache_locations_bwd = ctx->saved_data["use_uniq_cache_locations_bwd"].toBool();
        const auto use_homogeneous_placements = ctx->saved_data["use_homogeneous_placements"].toBool();
        auto learning_rate = ctx->saved_data["learning_rate"].toDouble();

        TORCH_CHECK_EQ(grad_outputs.size(), 1);

        constexpr int32_t BT_block_size = 32;
        constexpr int32_t max_segment_length_per_warp = 32;

        using torch::autograd::Variable;
        auto grad_output = gradient_clipping ? clamp(grad_outputs[0], -max_gradient, max_gradient) : grad_outputs[0];

        static auto split_embedding_codegen_unweighted_backward_op =
            torch::Dispatcher::singleton()
                .findSchemaOrThrow("fbgemm::split_embedding_backward_codegen_sgd_unweighted_exact_cuda", "")
                .typed<decltype(split_embedding_backward_codegen_sgd_unweighted_exact_cuda)>();

        const auto grad_dev_weights = split_embedding_codegen_unweighted_backward_op.call(
            grad_output, dev_weights, uvm_weights, lxu_cache_weights, weights_placements, weights_offsets, D_offsets,
            max_D, hash_size_cumsum, total_hash_size_bits, indices, offsets, pooling_mode, lxu_cache_locations,
            BT_block_size, max_segment_length_per_warp, stochastic_rounding, info_B_num_bits, info_B_mask_int64,
            use_uniq_cache_locations_bwd, use_homogeneous_placements, hash_indices, unique_ids, unique_offsets,
            unique_inverse, learning_rate);
        return {at::Tensor(),      // placeholder autograd tensor
                Variable(),        // output_dtype
                grad_dev_weights,  // dev_weights
                Variable(),        // uvm_weights
                Variable(),        // lxu_cache_weights
                Variable(),        // weights_placements
                Variable(),        // weights_offsets
                Variable(),        // D_offsets
                Variable(),        // total_D
                Variable(),        // max_D
                Variable(),        // hash_size_cumsum
                Variable(),        // total_hash_size_bits
                Variable(),        // indices
                Variable(),        // offsets
                Variable(),        // pooling_mode
                Variable(),        // indice_weights
                Variable(),        // feature_requires_grad
                Variable(),        // lxu_cache_locations
                Variable(),        // gradient_clipping
                Variable(),        // max_gradient
                Variable(),        // stochastic_rounding
                Variable(),        // is_experimental
                Variable(),        // use_uniq_cache_locations_bwd
                Variable(),        // use_homogeneous_placements
                Variable(),        // hash_indices
                Variable(),        // unique_ids
                Variable(),        // unique_offsets
                Variable(),        // unique_inverse
                Variable()};       // learning_rate
    }
};

at::Tensor split_embedding_codegen_lookup_adagrad_function(
    const at::Tensor& placeholder_autograd_tensor, const at::Tensor& dev_weights, const at::Tensor& uvm_weights,
    const at::Tensor& lxu_cache_weights, const at::Tensor& weights_placements, const at::Tensor& weights_offsets,
    const at::Tensor& D_offsets, const int64_t total_D, const int64_t max_D, const at::Tensor& hash_size_cumsum,
    const int64_t total_hash_size_bits, const at::Tensor& indices, const at::Tensor& offsets,
    const int64_t pooling_mode, const c10::optional<at::Tensor>& indice_weights,
    const c10::optional<at::Tensor>& feature_requires_grad, const at::Tensor& lxu_cache_locations,
    const bool gradient_clipping, const double max_gradient, const bool stochastic_rounding, at::Tensor momentum1_dev,
    at::Tensor momentum1_uvm, at::Tensor momentum1_placements, at::Tensor momentum1_offsets,
    const c10::optional<at::Tensor>& hash_indices = c10::optional<at::Tensor>(),
    const c10::optional<at::Tensor>& unique_ids = c10::optional<at::Tensor>(),
    const c10::optional<at::Tensor>& unique_offsets = c10::optional<at::Tensor>(),
    const c10::optional<at::Tensor>& unique_inverse = c10::optional<at::Tensor>(), double eps = 0,
    double learning_rate = 0, const int64_t output_dtype = static_cast<int64_t>(SparseType::FP32),
    const c10::optional<at::Tensor>& B_offsets = c10::optional<at::Tensor>(),
    const c10::optional<at::Tensor>& vbe_output_offsets_feature_rank = c10::optional<at::Tensor>(),
    const c10::optional<at::Tensor>& vbe_B_offsets_rank_per_feature = c10::optional<at::Tensor>(),
    const int64_t max_B = -1, const int64_t max_B_feature_rank = -1, const int64_t vbe_output_size = -1,
    const bool is_experimental = false, const bool use_uniq_cache_locations_bwd = false,
    const bool use_homogeneous_placements = false)
{
    return SplitLookupFunction_adagrad_Op::apply(
        placeholder_autograd_tensor, output_dtype, dev_weights, uvm_weights, lxu_cache_weights, weights_placements,
        weights_offsets, D_offsets, total_D, max_D, hash_size_cumsum, total_hash_size_bits, indices, hash_indices,
        unique_ids, unique_offsets, unique_inverse, offsets, pooling_mode, indice_weights, feature_requires_grad,
        lxu_cache_locations, gradient_clipping, max_gradient, stochastic_rounding, is_experimental,
        use_uniq_cache_locations_bwd, use_homogeneous_placements, momentum1_dev, momentum1_uvm, momentum1_placements,
        momentum1_offsets, eps, learning_rate)[0];
}

at::Tensor split_embedding_codegen_lookup_adam_function(
    const at::Tensor& placeholder_autograd_tensor, const at::Tensor& dev_weights, const at::Tensor& uvm_weights,
    const at::Tensor& lxu_cache_weights, const at::Tensor& weights_placements, const at::Tensor& weights_offsets,
    const at::Tensor& D_offsets, const int64_t total_D, const int64_t max_D, const at::Tensor& hash_size_cumsum,
    const int64_t total_hash_size_bits, const at::Tensor& indices, const at::Tensor& offsets,
    const int64_t pooling_mode, const c10::optional<at::Tensor>& indice_weights,
    const c10::optional<at::Tensor>& feature_requires_grad, const at::Tensor& lxu_cache_locations,
    const bool gradient_clipping, const double max_gradient, const bool stochastic_rounding, at::Tensor momentum1_dev,
    at::Tensor momentum1_uvm, at::Tensor momentum1_placements, at::Tensor momentum1_offsets, at::Tensor momentum2_dev,
    at::Tensor momentum2_uvm, at::Tensor momentum2_placements, at::Tensor momentum2_offsets,
    const c10::optional<at::Tensor>& hash_indices = c10::optional<at::Tensor>(),
    const c10::optional<at::Tensor>& unique_ids = c10::optional<at::Tensor>(),
    const c10::optional<at::Tensor>& unique_offsets = c10::optional<at::Tensor>(),
    const c10::optional<at::Tensor>& unique_inverse = c10::optional<at::Tensor>(), double eps = 0,
    double learning_rate = 0, double beta1 = 0.9, double beta2 = 0.999, int64_t iter = 0, double weight_decay = 0,
    const int64_t output_dtype = static_cast<int64_t>(SparseType::FP32),
    const c10::optional<at::Tensor>& B_offsets = c10::optional<at::Tensor>(),
    const c10::optional<at::Tensor>& vbe_output_offsets_feature_rank = c10::optional<at::Tensor>(),
    const c10::optional<at::Tensor>& vbe_B_offsets_rank_per_feature = c10::optional<at::Tensor>(),
    const int64_t max_B = -1, const int64_t max_B_feature_rank = -1, const int64_t vbe_output_size = -1,
    const bool is_experimental = false,  // formerly named is_experimental
    const bool use_uniq_cache_locations_bwd = false, const bool use_homogeneous_placements = false)
{
    return SplitLookupFunction_adam_Op::apply(
        placeholder_autograd_tensor, output_dtype, dev_weights, uvm_weights, lxu_cache_weights, weights_placements,
        weights_offsets, D_offsets, total_D, max_D, hash_size_cumsum, total_hash_size_bits, indices, hash_indices,
        unique_ids, unique_offsets, unique_inverse, offsets, pooling_mode, indice_weights, feature_requires_grad,
        lxu_cache_locations, gradient_clipping, max_gradient, stochastic_rounding, is_experimental,
        use_uniq_cache_locations_bwd, use_homogeneous_placements, momentum1_dev, momentum1_uvm, momentum1_placements,
        momentum1_offsets, momentum2_dev, momentum2_uvm, momentum2_placements, momentum2_offsets, eps, learning_rate,
        beta1, beta2, iter)[0];
}

at::Tensor split_embedding_codegen_lookup_sgd_function(
    const at::Tensor& placeholder_autograd_tensor, const at::Tensor& dev_weights, const at::Tensor& uvm_weights,
    const at::Tensor& lxu_cache_weights, const at::Tensor& weights_placements, const at::Tensor& weights_offsets,
    const at::Tensor& D_offsets, const int64_t total_D, const int64_t max_D, const at::Tensor& hash_size_cumsum,
    const int64_t total_hash_size_bits, const at::Tensor& indices, const at::Tensor& offsets,
    const int64_t pooling_mode, const c10::optional<at::Tensor>& indice_weights,
    const c10::optional<at::Tensor>& feature_requires_grad, const at::Tensor& lxu_cache_locations,
    const bool gradient_clipping, const double max_gradient, const bool stochastic_rounding,
    const c10::optional<at::Tensor>& hash_indices = c10::optional<at::Tensor>(),
    const c10::optional<at::Tensor>& unique_ids = c10::optional<at::Tensor>(),
    const c10::optional<at::Tensor>& unique_offsets = c10::optional<at::Tensor>(),
    const c10::optional<at::Tensor>& unique_inverse = c10::optional<at::Tensor>(), double learning_rate = 0,
    const int64_t output_dtype = static_cast<int64_t>(SparseType::FP32),
    const c10::optional<at::Tensor>& B_offsets = c10::optional<at::Tensor>(),
    const c10::optional<at::Tensor>& vbe_output_offsets_feature_rank = c10::optional<at::Tensor>(),
    const c10::optional<at::Tensor>& vbe_B_offsets_rank_per_feature = c10::optional<at::Tensor>(),
    const int64_t max_B = -1, const int64_t max_B_feature_rank = -1, const int64_t vbe_output_size = -1,
    const bool is_experimental = false, const bool use_uniq_cache_locations_bwd = false,
    const bool use_homogeneous_placements = false)
{
    return SplitLookupFunction_sgd_Op::apply(
        placeholder_autograd_tensor, output_dtype, dev_weights, uvm_weights, lxu_cache_weights, weights_placements,
        weights_offsets, D_offsets, total_D, max_D, hash_size_cumsum, total_hash_size_bits, indices, hash_indices,
        unique_ids, unique_offsets, unique_inverse, offsets, pooling_mode, indice_weights, feature_requires_grad,
        lxu_cache_locations, gradient_clipping, max_gradient, stochastic_rounding, is_experimental,
        use_uniq_cache_locations_bwd, use_homogeneous_placements, learning_rate)[0];
}

at::Tensor split_embedding_codegen_lookup_adagrad_function_npu(
    const at::Tensor& placeholder_autograd_tensor, const at::Tensor& dev_weights, const at::Tensor& uvm_weights,
    const at::Tensor& lxu_cache_weights, const at::Tensor& weights_placements, const at::Tensor& weights_offsets,
    const at::Tensor& D_offsets, const int64_t total_D, const int64_t max_D, const at::Tensor& hash_size_cumsum,
    const int64_t total_hash_size_bits, const at::Tensor& indices, const at::Tensor& offsets,
    const int64_t pooling_mode, const c10::optional<at::Tensor>& indice_weights,
    const c10::optional<at::Tensor>& feature_requires_grad, const at::Tensor& lxu_cache_locations,
    const bool gradient_clipping, const double max_gradient, const bool stochastic_rounding, at::Tensor momentum1_dev,
    at::Tensor momentum1_uvm, at::Tensor momentum1_placements, at::Tensor momentum1_offsets, double eps = 0,
    double learning_rate = 0, const int64_t output_dtype = 0,
    const c10::optional<at::Tensor>& B_offsets = c10::optional<at::Tensor>(),
    const c10::optional<at::Tensor>& vbe_output_offsets_feature_rank = c10::optional<at::Tensor>(),
    const c10::optional<at::Tensor>& vbe_B_offsets_rank_per_feature = c10::optional<at::Tensor>(),
    const int64_t max_B = -1, const int64_t max_B_feature_rank = -1, const int64_t vbe_output_size = -1,
    const bool is_experimental = false, const bool use_uniq_cache_locations_bwd = false,
    const bool use_homogeneous_placements = false)
{
    const at::OptionalDeviceGuard guard(device_of(dev_weights));
    int64_t featCnt = weights_placements.size(0);
    if (featCnt == 0) {
        return at::Tensor();
    }
    int64_t batchSize = (offsets.size(0) - 1) / featCnt;

    auto output = at::full({batchSize, total_D}, 1.0, dev_weights.options());
    return output;
}

at::Tensor split_embedding_codegen_forward_unweighted_npu(
    const at::Tensor& dev_weights, const at::Tensor& uvm_weights, const at::Tensor& lxu_cache_weights,
    const at::Tensor& weights_placements, const at::Tensor& weights_offsets, const at::Tensor& D_offsets,
    const int64_t total_D, const int64_t max_D, const at::Tensor& indices, const at::Tensor& offsets,
    const int64_t pooling_mode, const at::Tensor& lxu_cache_locations, const int64_t output_dtype,
    const bool is_experimental, const at::Tensor& hash_indices)
{
    const at::OptionalDeviceGuard guard(device_of(dev_weights));
    int64_t featCnt = weights_placements.size(0);
    int32_t totalLen = hash_indices.numel() == 0 ? indices.numel() : hash_indices.numel();
    if (featCnt == 0) {
        return at::Tensor();
    }

    if (totalLen == 0) {
        return at::Tensor();
    }

    int64_t batchSize = (offsets.size(0) - 1) / featCnt;
    auto output = at::full({batchSize, total_D}, 0.0, dev_weights.options());

    if (static_cast<PoolingMode>(pooling_mode) == PoolingMode::NONE) {
        output = at::full({totalLen, max_D}, 0.0, dev_weights.options());
    }

    EXEC_NPU_CMD(aclnnSplitEmbeddingCodegenForwardUnweighted, dev_weights, uvm_weights, lxu_cache_weights,
                 weights_placements, weights_offsets, D_offsets, indices, offsets, lxu_cache_locations, hash_indices,
                 total_D, max_D, pooling_mode, output_dtype, is_experimental, output);
    return output;
}

at::Tensor split_embedding_backward_codegen_adagrad_unweighted_exact_npu(
    const at::Tensor& grad_output, const at::Tensor& dev_weights, const at::Tensor& uvm_weights,
    const at::Tensor& lxu_cache_weights, const at::Tensor& weights_placements, const at::Tensor& weights_offsets,
    const at::Tensor& D_offsets, const int64_t max_D, const at::Tensor& hash_size_cumsum,
    const int64_t total_hash_size_bits, const at::Tensor& indices, const at::Tensor& offsets,
    const int64_t pooling_mode, const at::Tensor& lxu_cache_locations, const int64_t BT_block_size,
    const int64_t max_segment_length_per_warp, const bool stochastic_rounding, const int64_t info_B_num_bits,
    const int64_t info_B_mask_int64, const bool use_uniq_cache_locations, const bool use_homogeneous_placements,
    at::Tensor momentum1_dev, at::Tensor momentum1_uvm, at::Tensor momentum1_placements, at::Tensor momentum1_offsets,
    const at::Tensor& hash_indices, const at::Tensor& unique_ids, const at::Tensor& unique_offsets,
    const at::Tensor& unique_inverse, double eps = 0, double learning_rate = 0)
{
    const at::OptionalDeviceGuard guard(device_of(dev_weights));
    int64_t totalEmbed = unique_ids.numel() == 0 ? dev_weights.size(0) : unique_ids.numel() * max_D;
    auto output = at::empty({totalEmbed}, dev_weights.options());
    int optim_type = ADAGRAD;
    float beta = 0;
    int iter = 0;

    EXEC_NPU_CMD(aclnnBackwardCodegenAdagradUnweightedExact, grad_output, dev_weights, uvm_weights, lxu_cache_weights,
                 weights_placements, weights_offsets, D_offsets, hash_size_cumsum, indices, offsets,
                 lxu_cache_locations, momentum1_dev, momentum1_uvm, momentum1_placements, momentum1_offsets,
                 momentum1_dev, momentum1_uvm, momentum1_placements, momentum1_offsets, hash_indices, unique_ids,
                 unique_offsets, unique_inverse, max_D, total_hash_size_bits, pooling_mode, BT_block_size,
                 max_segment_length_per_warp, stochastic_rounding, info_B_num_bits, info_B_mask_int64,
                 use_uniq_cache_locations, use_homogeneous_placements, optim_type, eps, learning_rate, beta, beta, iter,
                 output, momentum1_dev, momentum1_dev, dev_weights);
    return at::Tensor();
}

at::Tensor split_embedding_backward_codegen_adam_unweighted_exact_npu(
    const at::Tensor& grad_output, const at::Tensor& dev_weights, const at::Tensor& uvm_weights,
    const at::Tensor& lxu_cache_weights, const at::Tensor& weights_placements, const at::Tensor& weights_offsets,
    const at::Tensor& D_offsets, const int64_t max_D, const at::Tensor& hash_size_cumsum,
    const int64_t total_hash_size_bits, const at::Tensor& indices, const at::Tensor& offsets,
    const int64_t pooling_mode, const at::Tensor& lxu_cache_locations, const int64_t BT_block_size,
    const int64_t max_segment_length_per_warp, const bool stochastic_rounding, const int64_t info_B_num_bits,
    const int64_t info_B_mask_int64, const bool use_uniq_cache_locations, const bool use_homogeneous_placements,
    at::Tensor momentum1_dev, at::Tensor momentum1_uvm, at::Tensor momentum1_placements, at::Tensor momentum1_offsets,
    at::Tensor momentum2_dev, at::Tensor momentum2_uvm, at::Tensor momentum2_placements, at::Tensor momentum2_offsets,
    const at::Tensor& hash_indices, const at::Tensor& unique_ids, const at::Tensor& unique_offsets,
    const at::Tensor& unique_inverse, double eps = 0, double learning_rate = 0, double beta1 = 0, double beta2 = 0,
    int64_t iter = 0)
{
    const at::OptionalDeviceGuard guard(device_of(dev_weights));
    int64_t totalEmbed = dev_weights.size(0);
    auto output = at::empty({totalEmbed}, dev_weights.options());
    int optim_type = ADAM;
    EXEC_NPU_CMD(aclnnBackwardCodegenAdagradUnweightedExact, grad_output, dev_weights, uvm_weights, lxu_cache_weights,
                 weights_placements, weights_offsets, D_offsets, hash_size_cumsum, indices, offsets,
                 lxu_cache_locations, momentum1_dev, momentum1_uvm, momentum1_placements, momentum1_offsets,
                 momentum2_dev, momentum2_uvm, momentum2_placements, momentum2_offsets, hash_indices, unique_ids,
                 unique_offsets, unique_inverse, max_D, total_hash_size_bits, pooling_mode, BT_block_size,
                 max_segment_length_per_warp, stochastic_rounding, info_B_num_bits, info_B_mask_int64,
                 use_uniq_cache_locations, use_homogeneous_placements, optim_type, eps, learning_rate, beta1, beta2,
                 iter, output, momentum1_dev, momentum2_dev, dev_weights);
    return at::Tensor();
}

at::Tensor split_embedding_backward_codegen_sgd_unweighted_exact_npu(
    const at::Tensor& grad_output, const at::Tensor& dev_weights, const at::Tensor& uvm_weights,
    const at::Tensor& lxu_cache_weights, const at::Tensor& weights_placements, const at::Tensor& weights_offsets,
    const at::Tensor& D_offsets, const int64_t max_D, const at::Tensor& hash_size_cumsum,
    const int64_t total_hash_size_bits, const at::Tensor& indices, const at::Tensor& offsets,
    const int64_t pooling_mode, const at::Tensor& lxu_cache_locations, const int64_t BT_block_size,
    const int64_t max_segment_length_per_warp, const bool stochastic_rounding, const int64_t info_B_num_bits,
    const int64_t info_B_mask_int64, const bool use_uniq_cache_locations, const bool use_homogeneous_placements,
    const at::Tensor& hash_indices, const at::Tensor& unique_ids, const at::Tensor& unique_offsets,
    const at::Tensor& unique_inverse, double learning_rate = 0)
{
    const at::OptionalDeviceGuard guard(device_of(dev_weights));
    int64_t totalEmbed = unique_ids.numel() == 0 ? dev_weights.size(0) : unique_ids.numel() * max_D;
    auto output = at::empty({totalEmbed}, dev_weights.options());

    int optim_type = SGD;
    const auto _unused = at::Tensor();
    const int iter = 0;
    const float beta = 0;

    EXEC_NPU_CMD(aclnnBackwardCodegenAdagradUnweightedExact, grad_output, dev_weights, uvm_weights, lxu_cache_weights,
                 weights_placements, weights_offsets, D_offsets, hash_size_cumsum, indices, offsets,
                 lxu_cache_locations, _unused, _unused, _unused, _unused, _unused, _unused, _unused, _unused,
                 hash_indices, unique_ids, unique_offsets, unique_inverse, max_D, total_hash_size_bits, pooling_mode,
                 BT_block_size, max_segment_length_per_warp, stochastic_rounding, info_B_num_bits, info_B_mask_int64,
                 use_uniq_cache_locations, use_homogeneous_placements, optim_type, beta, learning_rate, beta, beta,
                 iter, output, _unused, _unused, dev_weights);
    return at::Tensor();
}
};  // namespace fbgemm_npu_lookups

TORCH_LIBRARY_FRAGMENT(fbgemm, m)
{
    m.def("split_embedding_codegen_lookup_adagrad_function("
          "    Tensor placeholder_autograd_tensor, "
          "    Tensor dev_weights, Tensor uvm_weights, "
          "    Tensor lxu_cache_weights, "
          "    Tensor weights_placements, "
          "    Tensor weights_offsets, "
          "    Tensor D_offsets, "
          "    int total_D, "
          "    int max_D, "
          "    Tensor hash_size_cumsum, "
          "    int total_hash_size_bits, "
          "    Tensor indices, "
          "    Tensor offsets, "
          "    int pooling_mode, "
          "    Tensor? indice_weights, "
          "    Tensor? feature_requires_grad, "
          "    Tensor lxu_cache_locations, "
          "    bool gradient_clipping, "
          "    float max_gradient, "
          "    bool stochastic_rounding, "
          "    Tensor momentum1_dev, Tensor momentum1_uvm, Tensor momentum1_placements, Tensor momentum1_offsets, "
          "    Tensor? hash_indices = None, "
          "    Tensor? unique_ids = None, "
          "    Tensor? unique_offsets = None, "
          "    Tensor? unique_inverse = None, "
          "    float eps = 0, float learning_rate = 0, "
          "    int output_dtype=0, "
          "    Tensor? B_offsets=None, "
          "    Tensor? vbe_output_offsets_feature_rank=None, "
          "    Tensor? vbe_B_offsets_rank_per_feature=None, "
          "    int max_B=-1, "
          "    int max_B_feature_rank=-1, "
          "    int vbe_output_size=-1, "
          "    bool is_experimental=False, "
          "    bool use_uniq_cache_locations_bwd=False, "
          "    bool use_homogeneous_placements=False) -> Tensor");
    m.impl("split_embedding_codegen_lookup_adagrad_function",
           torch::dispatch(c10::DispatchKey::Autograd,
                           TORCH_FN(fbgemm_npu_lookups::split_embedding_codegen_lookup_adagrad_function)));
    DISPATCH_TO_NPU("split_embedding_codegen_lookup_adagrad_function",
                    fbgemm_npu_lookups::split_embedding_codegen_lookup_adagrad_function);
}

TORCH_LIBRARY_FRAGMENT(fbgemm, m)
{
    m.def("split_embedding_codegen_lookup_adam_function("
          "    Tensor placeholder_autograd_tensor, "
          "    Tensor dev_weights, Tensor uvm_weights, "
          "    Tensor lxu_cache_weights, "
          "    Tensor weights_placements, "
          "    Tensor weights_offsets, "
          "    Tensor D_offsets, "
          "    int total_D, "
          "    int max_D, "
          "    Tensor hash_size_cumsum, "
          "    int total_hash_size_bits, "
          "    Tensor indices, "
          "    Tensor offsets, "
          "    int pooling_mode, "
          "    Tensor? indice_weights, "
          "    Tensor? feature_requires_grad, "
          "    Tensor lxu_cache_locations, "
          "    bool gradient_clipping, "
          "    float max_gradient, "
          "    bool stochastic_rounding, "
          "    Tensor momentum1_dev, Tensor momentum1_uvm, Tensor momentum1_placements, Tensor momentum1_offsets, "
          "    Tensor momentum2_dev, Tensor momentum2_uvm, Tensor momentum2_placements, Tensor momentum2_offsets, "
          "    Tensor? hash_indices = None, "
          "    Tensor? unique_ids = None, "
          "    Tensor? unique_offsets = None, "
          "    Tensor? unique_inverse = None, "
          "    float eps = 0, "
          "    float learning_rate = 0, "
          "    float beta1 = 0.9, "
          "    float beta2 = 0.999, "
          "    int iter = 0, "
          "    float weight_decay = 0, "
          "    int output_dtype = 0,"
          "    Tensor? B_offsets = None,"
          "    Tensor? vbe_output_offsets_feature_rank = None, "
          "    Tensor? vbe_B_offsets_rank_per_feature = None, "
          "    int max_B = -1, "
          "    int max_B_feature_rank = -1, "
          "    int vbe_output_size = -1, "
          "    bool is_experimental = False, "
          "    bool use_uniq_cache_locations_bwd = False, "
          "    bool use_homogeneous_placements = False"
          ") -> Tensor");
    m.impl("split_embedding_codegen_lookup_adam_function",
           torch::dispatch(c10::DispatchKey::Autograd,
                           TORCH_FN(fbgemm_npu_lookups::split_embedding_codegen_lookup_adam_function)));
    DISPATCH_TO_NPU("split_embedding_codegen_lookup_adam_function",
                    fbgemm_npu_lookups::split_embedding_codegen_lookup_adam_function);
}

TORCH_LIBRARY_FRAGMENT(fbgemm, m)
{
    m.def("split_embedding_codegen_lookup_sgd_function("
          "    Tensor placeholder_autograd_tensor, "
          "    Tensor dev_weights, Tensor uvm_weights, "
          "    Tensor lxu_cache_weights, "
          "    Tensor weights_placements, "
          "    Tensor weights_offsets, "
          "    Tensor D_offsets, "
          "    int total_D, "
          "    int max_D, "
          "    Tensor hash_size_cumsum, "
          "    int total_hash_size_bits, "
          "    Tensor indices, "
          "    Tensor offsets, "
          "    int pooling_mode, "
          "    Tensor? indice_weights, "
          "    Tensor? feature_requires_grad, "
          "    Tensor lxu_cache_locations, "
          "    bool gradient_clipping, "
          "    float max_gradient, "
          "    bool stochastic_rounding, "
          "    Tensor? hash_indices = None, "
          "    Tensor? unique_ids = None, "
          "    Tensor? unique_offsets = None, "
          "    Tensor? unique_inverse = None, "
          "    float learning_rate = 0, "
          "    int output_dtype=0, "
          "    Tensor? B_offsets=None, "
          "    Tensor? vbe_output_offsets_feature_rank=None, "
          "    Tensor? vbe_B_offsets_rank_per_feature=None, "
          "    int max_B=-1, "
          "    int max_B_feature_rank=-1, "
          "    int vbe_output_size=-1, "
          "    bool is_experimental=False, "
          "    bool use_uniq_cache_locations_bwd=False, "
          "    bool use_homogeneous_placements=False) -> Tensor");
    m.impl("split_embedding_codegen_lookup_sgd_function",
           torch::dispatch(c10::DispatchKey::Autograd,
                           TORCH_FN(fbgemm_npu_lookups::split_embedding_codegen_lookup_sgd_function)));
    DISPATCH_TO_NPU("split_embedding_codegen_lookup_sgd_function",
                    fbgemm_npu_lookups::split_embedding_codegen_lookup_sgd_function);
}

TORCH_LIBRARY_FRAGMENT(fbgemm, m)
{
    m.def("split_embedding_codegen_forward_unweighted_cuda("
          "    Tensor dev_weights, "
          "    Tensor uvm_weights, "
          "    Tensor lxu_cache_weights, "
          "    Tensor weights_placements, "
          "    Tensor weights_offsets, "
          "    Tensor D_offsets, "
          "    int total_D, "
          "    int max_D, "
          "    Tensor indices, "
          "    Tensor offsets, "
          "    int pooling_mode, "
          "    Tensor lxu_cache_locations, "
          "    int output_dtype, "
          "    bool is_experimental, "
          "    Tensor hash_indices = None"
          ") -> Tensor");
    DISPATCH_TO_NPU("split_embedding_codegen_forward_unweighted_cuda",
                    fbgemm_npu_lookups::split_embedding_codegen_forward_unweighted_npu);
}

TORCH_LIBRARY_FRAGMENT(fbgemm, m)
{
    m.def("split_embedding_backward_codegen_adagrad_unweighted_exact_cuda("
          "    Tensor grad_output, "
          "    Tensor dev_weights, "
          "    Tensor uvm_weights, "
          "    Tensor lxu_cache_weights, "
          "    Tensor weights_placements, "
          "    Tensor weights_offsets, "
          "    Tensor D_offsets, "
          "    int max_D, "
          "    Tensor hash_size_cumsum, "
          "    int total_hash_size_bits, "
          "    Tensor indices, "
          "    Tensor offsets, "
          "    int pooling_mode, "
          "    Tensor lxu_cache_locations, "
          "    int unused_, "
          "    int max_segment_length_per_warp, "
          "    bool stochastic_rounding, "
          "    int info_B_num_bits, "
          "    int info_B_mask_int64, "
          "    bool use_uniq_cache_locations, "
          "    bool use_homogeneous_placements, "
          "    Tensor momentum1_dev, Tensor momentum1_uvm, Tensor momentum1_placements, Tensor momentum1_offsets, "
          "    Tensor hash_indices = None, "
          "    Tensor unique_ids = None, "
          "    Tensor unique_offsets = None, "
          "    Tensor unique_inverse = None, "
          "    float eps = 0, float learning_rate = 0"
          ") -> Tensor");
    DISPATCH_TO_NPU("split_embedding_backward_codegen_adagrad_unweighted_exact_cuda",
                    fbgemm_npu_lookups::split_embedding_backward_codegen_adagrad_unweighted_exact_npu);
}

TORCH_LIBRARY_FRAGMENT(fbgemm, m)
{
    m.def("split_embedding_backward_codegen_adam_unweighted_exact_cuda("
          "    Tensor placeholder_autograd_tensor, "
          "    Tensor dev_weights, "
          "    Tensor uvm_weights, "
          "    Tensor lxu_cache_weights, "
          "    Tensor weights_placements, "
          "    Tensor weights_offsets, "
          "    Tensor D_offsets, "
          "    int max_D, "
          "    Tensor hash_size_cumsum, "
          "    int total_hash_size_bits, "
          "    Tensor indices, "
          "    Tensor offsets, "
          "    int pooling_mode, "
          "    Tensor lxu_cache_locations, "
          "    int unused_, "
          "    int max_segment_length_per_warp, "
          "    bool stochastic_rounding, "
          "    int info_B_num_bits, "
          "    int info_B_mask_int64, "
          "    bool use_uniq_cache_locations, "
          "    bool use_homogeneous_placements, "
          "    Tensor momentum1_dev, "
          "    Tensor momentum1_uvm, "
          "    Tensor momentum1_placements, "
          "    Tensor momentum1_offsets, "
          "    Tensor momentum2_dev, "
          "    Tensor momentum2_uvm, "
          "    Tensor momentum2_placements, "
          "    Tensor momentum2_offsets, "
          "    Tensor hash_indices = None, "
          "    Tensor unique_ids = None, "
          "    Tensor unique_offsets = None, "
          "    Tensor unique_inverse = None, "
          "    float eps = 0, "
          "    float learning_rate = 0, "
          "    float beta1 = 0, "
          "    float beta2 = 0, "
          "    int iter = 0"
          ") -> Tensor");
    DISPATCH_TO_NPU("split_embedding_backward_codegen_adam_unweighted_exact_cuda",
                    fbgemm_npu_lookups::split_embedding_backward_codegen_adam_unweighted_exact_npu);
}

TORCH_LIBRARY_FRAGMENT(fbgemm, m)
{
    m.def("split_embedding_backward_codegen_sgd_unweighted_exact_cuda("
          "    Tensor grad_output, "
          "    Tensor dev_weights, "
          "    Tensor uvm_weights, "
          "    Tensor lxu_cache_weights, "
          "    Tensor weights_placements, "
          "    Tensor weights_offsets, "
          "    Tensor D_offsets, "
          "    int max_D, "
          "    Tensor hash_size_cumsum, "
          "    int total_hash_size_bits, "
          "    Tensor indices, "
          "    Tensor offsets, "
          "    int pooling_mode, "
          "    Tensor lxu_cache_locations, "
          "    int unused_, "
          "    int max_segment_length_per_warp, "
          "    bool stochastic_rounding, "
          "    int info_B_num_bits, "
          "    int info_B_mask_int64, "
          "    bool use_uniq_cache_locations, "
          "    bool use_homogeneous_placements, "
          "    Tensor hash_indices = None, "
          "    Tensor unique_ids = None, "
          "    Tensor unique_offsets = None, "
          "    Tensor unique_inverse = None, "
          "    float learning_rate = 0"
          ") -> Tensor");
    DISPATCH_TO_NPU("split_embedding_backward_codegen_sgd_unweighted_exact_cuda",
                    fbgemm_npu_lookups::split_embedding_backward_codegen_sgd_unweighted_exact_npu);
}