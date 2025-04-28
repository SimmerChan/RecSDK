
/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
 * Description: implement of SparseFwFFMGrad
 */
#include "sparse_fw_ffm_grad_kernels.h"

#include <cmath>
#include <algorithm>

namespace  {
const char *SPARSE_FW_FFM_GRAD = "SparseFwFFMGrad";
}

namespace aicpu  {

template <typename T>
std::uint32_t DoComputeSparseFwFFMGrad(const Tensor* weight_tensor, const Tensor* fw_weight_tensor,
        const Tensor* field_tensor, const Tensor* index_tensor,
        const Tensor* cross_mean_sum_tensor, const Tensor* cross_mean_square_sum_tensor,
        const Tensor* fw_field_map_tensor, const Tensor* grad_tensor,
        Tensor* output_tensor, Tensor* fw_output_tensor) {

    const int32_t sample_feature_size = weight_tensor->GetTensorShape()->GetDimSize(0);
    const int32_t field_num = weight_tensor->GetTensorShape()->GetDimSize(1);
    const int32_t embedding_size = weight_tensor->GetTensorShape()->GetDimSize(2);
    const int32_t batch_size = grad_tensor->GetTensorShape()->GetDimSize(0);
    const int32_t fw_field_num = cross_mean_sum_tensor->GetTensorShape()->GetDimSize(2);

    const T* weight_flat = reinterpret_cast<const T*>(weight_tensor->GetData());
    const T* fw_weight_ptr = reinterpret_cast<const T*>(fw_weight_tensor->GetData());
    const int32_t* field_flat = reinterpret_cast<const int32_t*>(field_tensor->GetData());
    const int32_t* index_flat = reinterpret_cast<const int32_t*>(index_tensor->GetData());
    
    T* output_ptr = reinterpret_cast<T*>(output_tensor->GetData());
    T* fw_output_ptr = reinterpret_cast<T*>(fw_output_tensor->GetData());
    std::fill_n(output_ptr, output_tensor->NumElements(), T(0.));
    std::fill_n(fw_output_ptr, fw_output_tensor->NumElements(), T(0.));

    const size_t buffer_size = field_num * fw_field_num * embedding_size;
    const T* fw_cross_mean_sum_ptr = reinterpret_cast<const T*>(cross_mean_sum_tensor->GetData());
    const T* fw_cross_mean_square_sum_ptr = reinterpret_cast<const T*>(cross_mean_square_sum_tensor->GetData());
    const int32_t* fw_field_map_ptr = reinterpret_cast<const int32_t*>(fw_field_map_tensor->GetData());
    const T* grad_ptr = reinterpret_cast<const T*>(grad_tensor->GetData());

    std::unique_ptr<T[]> fw_cross_mean_sum_grad(new T[batch_size * buffer_size]);
    std::unique_ptr<T[]> fw_cross_mean_square_sum_grad(new T[batch_size * buffer_size]);
    std::fill_n(fw_cross_mean_sum_grad.get(), batch_size * buffer_size, T(0.));
    std::fill_n(fw_cross_mean_square_sum_grad.get(), batch_size * buffer_size, T(0.));

    for (int n = 0; n < batch_size; n++) {
        auto fw_weight_data = fw_weight_tensor->GetTensorShape()->GetDims() > 1 ?
            fw_weight_ptr + n * fw_weight_tensor->GetTensorShape()->GetDimSize(1) : fw_weight_ptr;
        auto fw_output_data = fw_weight_tensor->GetTensorShape()->GetDims() > 1 ?
            fw_output_ptr + n * fw_output_tensor->GetTensorShape()->GetDimSize(1) : fw_output_ptr;

        int32_t fw_iter = 0;
        const T* cross_mean_sum_ptr = fw_cross_mean_sum_ptr + n * buffer_size;
        const T* cross_mean_square_sum_ptr = fw_cross_mean_square_sum_ptr + n * buffer_size;
        T* cross_mean_sum_grad_ptr = fw_cross_mean_sum_grad.get() + n * buffer_size;
        T* cross_mean_square_sum_grad_ptr = fw_cross_mean_square_sum_grad.get() + n * buffer_size;

        for (int32_t fw_field_1 = 0; fw_field_1 < fw_field_num; fw_field_1++) {
            bool multi_tag = false;
            int32_t field_1 = fw_field_map_ptr[n * fw_field_num + fw_field_1];
            if (field_1 >= 0) {
                if (field_1 >= field_num) {
                    multi_tag = true;
                    field_1 -= field_num;
                }
                for (int32_t fw_field_2 = 0; fw_field_2 < fw_field_1; fw_field_2++) {
                    int32_t field_2 = fw_field_map_ptr[n * fw_field_num + fw_field_2];
                    if (field_2 >= 0) {
                        if (field_2 >= field_num) {
                            field_2 -= field_num;
                        }
                        for (int32_t k = 0; k < embedding_size; k++) {
                            T grad_value = grad_ptr[n * embedding_size + k];
                            int32_t index_1 = (field_1 * fw_field_num + fw_field_2) * embedding_size + k;
                            int32_t index_2 = (field_2 * fw_field_num + fw_field_1) * embedding_size + k;

                            if (fabs(T(1.) + fw_weight_data[fw_iter]) > T(0.)) {
                                cross_mean_sum_grad_ptr[index_1] += grad_value * (T(1.) + fw_weight_data[fw_iter]) * cross_mean_sum_ptr[index_2];
                                cross_mean_sum_grad_ptr[index_2] += grad_value * (T(1.) + fw_weight_data[fw_iter]) * cross_mean_sum_ptr[index_1];
                            }

                            fw_output_data[fw_iter] += grad_value * cross_mean_sum_ptr[index_1] * cross_mean_sum_ptr[index_2];
                        }
                    }
                    fw_iter++;
                }
                
                for (int32_t k = 0; k < embedding_size; k++) {
                    T grad_value = grad_ptr[n * embedding_size + k];
                    int32_t index_1 = (field_1 * fw_field_num + fw_field_1) * embedding_size + k;
                    if (fabs(T(1.) + fw_weight_data[fw_iter]) > T(0.)) {
                        cross_mean_sum_grad_ptr[index_1] += grad_value * (T(1.) + fw_weight_data[fw_iter]) * cross_mean_sum_ptr[index_1];
                        cross_mean_square_sum_grad_ptr[index_1] -= T(0.5) * grad_value * (T(1.) + fw_weight_data[fw_iter]);
                    }
                    if (multi_tag) {
                        fw_output_data[fw_iter] +=
                            T(0.5) * grad_value * (cross_mean_sum_ptr[index_1] * cross_mean_sum_ptr[index_1] - cross_mean_square_sum_ptr[index_1]);
                    }
                }

                fw_iter++;
            } else {
                fw_iter += fw_field_1 + 1;
            }
        }
    }

    for (int32_t s = 0; s < sample_feature_size; s++) {
        int32_t sample_id = index_flat[s];
        int32_t field_1 = field_flat[s * 2] - 1;
        int32_t fw_field_1 = field_flat[s * 2 + 1] - 1;

        T* cross_mean_sum_grad_ptr = fw_cross_mean_sum_grad.get() + sample_id * buffer_size;
        T* cross_mean_square_sum_grad_ptr = fw_cross_mean_square_sum_grad.get() + sample_id * buffer_size;

        for (int32_t field_2 = 0; field_2 < field_num; field_2++) {
            for (int32_t k = 0; k < embedding_size; k++) {
                int32_t index_2 = (field_2 * fw_field_num + fw_field_1) * embedding_size + k;
                T weight_value = weight_flat[s * (field_num * embedding_size) + field_2 * embedding_size + k];

                if (field_1 == field_2) {
                    output_ptr[s * (field_num * embedding_size) + field_2 * embedding_size + k] +=
                        cross_mean_sum_grad_ptr[index_2] + T(2.) * cross_mean_square_sum_grad_ptr[index_2] * weight_value;
                } else {
                    output_ptr[s * (field_num * embedding_size) + field_2 * embedding_size + k] +=
                        cross_mean_sum_grad_ptr[index_2];
                }
            }
        }
    }

    return 0;
}

uint32_t SparseFwFFMGradCpuKernel::Compute(CpuKernelContext &ctx)
{
    const Tensor* weight_tensor = ctx.Input(0);
    const Tensor* fw_weight_tensor = ctx.Input(1);
    const Tensor* field_tensor = ctx.Input(2);
    const Tensor* index_tensor = ctx.Input(3);
    const Tensor* cross_mean_sum_tensor = ctx.Input(4);
    const Tensor* cross_mean_square_sum_tensor = ctx.Input(5);
    const Tensor* fw_field_map_tensor = ctx.Input(6);
    const Tensor* grad_tensor = ctx.Input(7);

    Tensor* output_tensor = ctx.Output(0);
    Tensor* fw_output_tensor = ctx.Output(1);

    switch (weight_tensor->GetDataType())
    {
        case DT_FLOAT:
            return DoComputeSparseFwFFMGrad<float>(weight_tensor, fw_weight_tensor, field_tensor, index_tensor,
                cross_mean_sum_tensor, cross_mean_square_sum_tensor, fw_field_map_tensor, grad_tensor,
                output_tensor, fw_output_tensor);
        case DT_DOUBLE:
            return DoComputeSparseFwFFMGrad<double>(weight_tensor, fw_weight_tensor, field_tensor, index_tensor,
                cross_mean_sum_tensor, cross_mean_square_sum_tensor, fw_field_map_tensor, grad_tensor,
                output_tensor, fw_output_tensor);
        default:
            return 1;
    }

    return 0;
}

REGISTER_CPU_KERNEL(SPARSE_FW_FFM_GRAD, SparseFwFFMGradCpuKernel);
} // namespace aicpu
