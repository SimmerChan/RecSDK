
/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
 * Description: implement of SparseFwFFM
 */
#include "sparse_fw_ffm_kernels.h"

#include "unsupported/Eigen/CXX11/Tensor"
#include <cmath>
#include <algorithm>

namespace  {
const char *SPARSE_FW_FFM = "SparseFwFFM";
}

namespace aicpu  {

template <typename T>
std::uint32_t DoComputeSparseFwFFM(const Tensor* weight_tensor, const Tensor* fw_weight_tensor,
        const Tensor* field_tensor, const Tensor* index_tensor, Tensor* output_tensor,
        Tensor* cross_mean_sum_tensor, Tensor* cross_mean_square_sum_tensor,
        Tensor* fw_field_map_tensor) {
    const int32_t sample_feature_size = weight_tensor->GetTensorShape()->GetDimSize(0);
    const int32_t field_num = weight_tensor->GetTensorShape()->GetDimSize(1);
    const int32_t embedding_size = weight_tensor->GetTensorShape()->GetDimSize(2);
    const int32_t batch_size = output_tensor->GetTensorShape()->GetDimSize(0);
    const int32_t fw_field_num = cross_mean_square_sum_tensor->GetTensorShape()->GetDimSize(2);

    const T* weight_flat = reinterpret_cast<T*>(weight_tensor->GetData());
    const T* fw_weight_ptr = reinterpret_cast<const T*>(fw_weight_tensor->GetData());
    const int32_t* field_flat = reinterpret_cast<const int32_t*>(field_tensor->GetData());
    const int32_t* index_flat = reinterpret_cast<const int32_t*>(index_tensor->GetData());

    const size_t buffer_size = field_num * fw_field_num * embedding_size;
    T* fw_cross_mean_sum_ptr = reinterpret_cast<T*>(cross_mean_sum_tensor->GetData());
    T* fw_cross_mean_square_sum_ptr = reinterpret_cast<T*>(cross_mean_square_sum_tensor->GetData());
    int32_t* fw_field_map_ptr = reinterpret_cast<int32_t*>(fw_field_map_tensor->GetData());
    std::fill_n(fw_field_map_ptr, fw_field_map_tensor->NumElements(), -1);

    T* output_ptr = reinterpret_cast<T*>(output_tensor->GetData());
    int32_t s = 0;
    for (int32_t n = 0; n < batch_size; n++) {
        const T* fw_weight_data = nullptr;
        if (fw_weight_tensor->GetTensorShape()->GetDims() > 1) {
            fw_weight_data = fw_weight_ptr + n * fw_weight_tensor->GetTensorShape()->GetDimSize(1);
        } else {
            fw_weight_data = fw_weight_ptr;
        }

        std::fill_n(fw_cross_mean_sum_ptr + n * buffer_size, buffer_size, T(0.));
        std::fill_n(fw_cross_mean_square_sum_ptr + n * buffer_size, buffer_size, T(0.));
        std::fill_n(output_ptr + n * embedding_size, embedding_size, T(0.));
        for (; (s < sample_feature_size && index_flat[s] == n); s++) {
            int32_t field_1 = field_flat[s * 2] - 1;
            if (field_1 < 0 || field_1 >= field_num) continue;
            int32_t fw_field_1 = field_flat[s * 2 + 1] - 1;
            if (fw_field_1 < 0 || fw_field_1 >= fw_field_num) continue;
            
            int32_t sample_id = index_flat[s];
            T* cross_mean_sum_ptr = fw_cross_mean_sum_ptr + sample_id * buffer_size;
            T* cross_mean_square_sum_ptr = fw_cross_mean_square_sum_ptr + sample_id * buffer_size;
            for (int32_t field_2 = 0; field_2 < field_num; field_2++) {
                for (int32_t k = 0; k < embedding_size; k++) {
                    int32_t index = (field_2 * fw_field_num + fw_field_1) * embedding_size + k;
                    T weight_value = weight_flat[s * (field_num * embedding_size) + field_2 * embedding_size + k];
                    cross_mean_sum_ptr[index] += weight_value;
                    cross_mean_square_sum_ptr[index] += weight_value * weight_value;
                }
            }

            int32_t offset = sample_id * fw_field_num + fw_field_1;
            if (fw_field_map_ptr[offset] != -1) {
                if (fw_field_map_ptr[offset] < field_num) {
                    fw_field_map_ptr[offset] += field_num;
                }
            } else {
                fw_field_map_ptr[offset] = field_1;
            }
        }

        int32_t fw_iter = 0;
        T* cross_mean_sum_ptr = fw_cross_mean_sum_ptr + n * buffer_size;
        T* cross_mean_square_sum_ptr = fw_cross_mean_square_sum_ptr + n * buffer_size;
        for (int32_t fw_field_1 = 0; fw_field_1 < fw_field_num; fw_field_1++) {
            int32_t field_1 = fw_field_map_ptr[n * fw_field_num + fw_field_1];
            if (field_1 >= 0) {
                bool multi_tag = false;
                if (field_1 >= field_num) {
                    field_1 -= field_num;
                    multi_tag = true;
                }
                for (int32_t fw_field_2 = 0; fw_field_2 < fw_field_1; fw_field_2++) {
                    int32_t field_2 = fw_field_map_ptr[n * fw_field_num + fw_field_2];
                    if (field_2 >= 0 && fabs(1.0 + static_cast<double>(fw_weight_data[fw_iter])) > 0.) {
                        if (field_2 >= field_num) {
                            field_2 -= field_num;
                        }
                        for (int32_t k = 0; k < embedding_size; k++) {
                            int32_t index_1 = (field_1 * fw_field_num + fw_field_2) * embedding_size + k;
                            int32_t index_2 = (field_2 * fw_field_num + fw_field_1) * embedding_size + k;
                            output_ptr[n * embedding_size + k] +=
                                cross_mean_sum_ptr[index_1] * cross_mean_sum_ptr[index_2] * (T(1.) + fw_weight_data[fw_iter]);
                        }
                    }
                    fw_iter++;
                }

                if (multi_tag && fabs(1.0 + static_cast<double>(fw_weight_data[fw_iter])) > 0.) {
                    for (int32_t k = 0; k < embedding_size; k++) {
                        int32_t index_1 = (field_1 * fw_field_num + fw_field_1) * embedding_size + k;
                        output_ptr[n * embedding_size + k] +=
                            (T(0.5) * (cross_mean_sum_ptr[index_1] * cross_mean_sum_ptr[index_1] - cross_mean_square_sum_ptr[index_1]))
                                * (T(1.) + fw_weight_data[fw_iter]);
                    }
                }
                fw_iter++;
            } else {
                fw_iter += fw_field_1 + 1;
            }
        }
    }

    return 0;
}


uint32_t SparseFwFFMCpuKernel::Compute(CpuKernelContext &ctx)
{
    const Tensor* weight_tensor = ctx.Input(0);
    const Tensor* fw_weight_tensor = ctx.Input(1);
    const Tensor* field_tensor = ctx.Input(2);
    const Tensor* index_tensor = ctx.Input(3);
    auto weight_shape = weight_tensor->GetTensorShape();
    auto fw_weight_shape = fw_weight_tensor->GetTensorShape();
    auto field_shape = field_tensor->GetTensorShape();
    auto index_shape = index_tensor->GetTensorShape();

    if (weight_shape->GetDims() != 3) {
        return 1;
    }
    if (field_shape->GetDims() < 2) {
        return 1;
    }
    if (fw_weight_shape->GetDims() != 1 && fw_weight_shape->GetDims() != 2) {
        return 1;
    }

    if (weight_shape->GetDimSize(0) != field_shape->GetDimSize(0)) {
        return 1;
    }
    if (weight_shape->GetDimSize(0) != index_shape->GetDimSize(0) - 1) {
        return 1;
    }
    if (field_shape->GetDimSize(1) != 2) {
        return 1;
    }
    
    for (int32_t i = 3; i < field_shape->GetDims(); i++) {
        if (field_shape->GetDimSize(i) != 1) {
            return 1;
        }
    }
    for (int32_t i = 2; i < index_shape->GetDims(); i++) {
        if (index_shape->GetDimSize(i) != 1) {
            return 1;
        }
    }

    const int32_t batch_size = reinterpret_cast<const int32_t*>(index_tensor->GetData())[index_shape->GetDimSize(0) - 1];

    if (fw_weight_shape->GetDims() > 1) {
        if (batch_size != fw_weight_shape->GetDimSize(0)) {
            return 1;
        }
    }

    const int32_t field_num = weight_shape->GetDimSize(1);
    const int32_t embedding_size = weight_shape->GetDimSize(2);
    const int32_t fw_field_num =
        floor(sqrt(2 * fw_weight_shape->GetDimSize(fw_weight_shape->GetDims() - 1)));

    Tensor* output_tensor = ctx.Output(0);
    auto output_shape = output_tensor->GetTensorShape();
    output_shape->SetDimSizes({batch_size, embedding_size});
    output_tensor->SetTensorShape(output_shape.get());

    Tensor* cross_mean_sum_tensor = ctx.Output(1);
    auto cross_mean_sum_shape = cross_mean_sum_tensor->GetTensorShape();
    cross_mean_sum_shape->SetDimSizes({batch_size, field_num, fw_field_num, embedding_size});
    cross_mean_sum_tensor->SetTensorShape(cross_mean_sum_shape.get());

    Tensor* cross_mean_square_sum_tensor = ctx.Output(2);
    auto cross_mean_square_sum_shape = cross_mean_square_sum_tensor->GetTensorShape();
    cross_mean_square_sum_shape->SetDimSizes({batch_size, field_num, fw_field_num, embedding_size});
    cross_mean_square_sum_tensor->SetTensorShape(cross_mean_square_sum_shape.get());

    Tensor* fw_field_map_tensor = ctx.Output(3);
    auto fw_field_map_shape = fw_field_map_tensor->GetTensorShape();
    fw_field_map_shape->SetDimSizes({batch_size, fw_field_num});
    fw_field_map_tensor->SetTensorShape(fw_field_map_shape.get());

    switch (weight_tensor->GetDataType())
    {
        case DT_FLOAT16:
            return DoComputeSparseFwFFM<Eigen::half>(weight_tensor, fw_weight_tensor, field_tensor, index_tensor,
                output_tensor, cross_mean_sum_tensor, cross_mean_square_sum_tensor, fw_field_map_tensor);
        case DT_FLOAT:
            return DoComputeSparseFwFFM<float>(weight_tensor, fw_weight_tensor, field_tensor, index_tensor,
                output_tensor, cross_mean_sum_tensor, cross_mean_square_sum_tensor, fw_field_map_tensor);
        case DT_DOUBLE:
            return DoComputeSparseFwFFM<double>(weight_tensor, fw_weight_tensor, field_tensor, index_tensor,
                output_tensor, cross_mean_sum_tensor, cross_mean_square_sum_tensor, fw_field_map_tensor);
        default:
            return 1;
    }

    return 0;
}

REGISTER_CPU_KERNEL(SPARSE_FW_FFM, SparseFwFFMCpuKernel);
} // namespace aicpu
