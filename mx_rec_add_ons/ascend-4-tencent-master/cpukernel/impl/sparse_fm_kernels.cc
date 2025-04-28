
/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
 * Description: implement of SparseFM
 */
#include "sparse_fm_kernels.h"

#include <unsupported/Eigen/CXX11/Tensor>
#include <algorithm>

namespace  {
const char *SPARSE_FM = "SparseFM";
}

namespace aicpu  {

template <typename T>
std::uint32_t DoComputeSparseFM(const Tensor* weight_tensor, const Tensor* index_tensor, int32_t bias_dim,
        Tensor* output_tensor, Tensor* cross_mean_sum_tensor, Tensor* cross_mean_square_sum_tensor) {
    const int32_t sample_feature_size = weight_tensor->GetTensorShape()->GetDimSize(0);
    const int32_t embedding_size = weight_tensor->GetTensorShape()->GetDimSize(1);
    const int32_t batch_size = output_tensor->GetTensorShape()->GetDimSize(0);

    T* output_flat = reinterpret_cast<T*>(output_tensor->GetData());
    T* cross_mean_sum_flat = reinterpret_cast<T*>(cross_mean_sum_tensor->GetData());
    T* cross_mean_square_sum_flat = reinterpret_cast<T*>(cross_mean_square_sum_tensor->GetData());
    std::fill_n(cross_mean_sum_flat, batch_size * embedding_size, T(0.));
    std::fill_n(cross_mean_square_sum_flat, batch_size * embedding_size, T(0.));

    const T* weight_flat = reinterpret_cast<T*>(weight_tensor->GetData());
    const int32_t* index_flat = reinterpret_cast<int32_t*>(index_tensor->GetData());
    for (int32_t s = 0; s < sample_feature_size; s++) {
        int32_t sample_id = index_flat[s];
        T* cross_mean_sum_ptr = cross_mean_sum_flat + sample_id * embedding_size;
        T* cross_mean_square_sum_ptr = cross_mean_square_sum_flat + sample_id * embedding_size;

        for (int32_t k = 0; k < embedding_size; k++) {
            T weight_value = weight_flat[s * embedding_size + k];
            cross_mean_sum_ptr[k] += weight_value;
            cross_mean_square_sum_ptr[k] += weight_value * weight_value;
        }
    }

    for (int32_t n = 0; n < batch_size; n++) {
        T* cross_mean_sum_ptr = cross_mean_sum_flat + n * embedding_size;
        T* cross_mean_square_sum_ptr = cross_mean_square_sum_flat + n * embedding_size;

        for (int32_t k = 0; k < bias_dim; k++) {
            output_flat[n * embedding_size + k] = cross_mean_sum_ptr[k];
        }
        for (int32_t k = bias_dim; k < embedding_size; k++) {
            output_flat[n * embedding_size + k] =
                static_cast<T>(0.5f) * (cross_mean_sum_ptr[k] * cross_mean_sum_ptr[k] -
                    cross_mean_square_sum_ptr[k]);
        }
    }

    return 0;
}

template <typename T>
std::uint32_t ComputeSparseFM(const CpuKernelContext &ctx) {
    int bias_dim = 1;
    const AttrValue* attr = ctx.GetAttr("bias_dim");
    if (attr != nullptr) {
        bias_dim = attr->GetInt();
    }

    const Tensor* weight_tensor = ctx.Input(0);
    const Tensor* index_tensor = ctx.Input(1);
    const int32_t* index_data = reinterpret_cast<const int32_t*>(index_tensor->GetData());
    const int32_t batch_size = index_data[index_tensor->NumElements() - 1];
    const int32_t embedding_size = weight_tensor->GetTensorShape()->GetDimSize(1);

    Tensor* output_tensor = ctx.Output(0);
    Tensor* cross_mean_sum_tensor = ctx.Output(1);
    Tensor* cross_mean_square_sum_tensor = ctx.Output(2);

    std::vector<Tensor*> output_tensors = {output_tensor, cross_mean_sum_tensor, cross_mean_square_sum_tensor};
    for (auto tensor: output_tensors) {
        auto output_shape = tensor->GetTensorShape();
        output_shape->SetDimSizes({batch_size, embedding_size});
        tensor->SetTensorShape(output_shape.get());
    }

    return DoComputeSparseFM<T>(weight_tensor, index_tensor, bias_dim, output_tensor,
        cross_mean_sum_tensor, cross_mean_square_sum_tensor);
}

uint32_t SparseFMCpuKernel::Compute(CpuKernelContext &ctx)
{
    DataType input_type = ctx.Input(0)->GetDataType();
    switch (input_type) {
        case DT_FLOAT16:
            return ComputeSparseFM<Eigen::half>(ctx);
        case DT_FLOAT:
            return ComputeSparseFM<float>(ctx);
        case DT_DOUBLE:
            return ComputeSparseFM<double>(ctx);
        default:
            return 1;
    }

    return 0;
}

REGISTER_CPU_KERNEL(SPARSE_FM, SparseFMCpuKernel);
} // namespace aicpu
