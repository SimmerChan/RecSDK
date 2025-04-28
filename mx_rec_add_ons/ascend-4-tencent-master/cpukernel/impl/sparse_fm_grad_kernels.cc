
/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
 * Description: implement of SparseFMGrad
 */
#include "sparse_fm_grad_kernels.h"

namespace  {
const char *BIAS_DIM = "bias_dim";
const char *SPARSE_FM_GRAD = "SparseFMGrad";
}

namespace aicpu  {

template <typename T>
std::uint32_t DoComputeSparseFMGrad(const Tensor* weight_tensor, const Tensor* index_tensor,
        const Tensor* cross_mean_sum_tensor, const Tensor* cross_mean_square_sum_tensor,
        const Tensor* grad_tensor, Tensor* output_tensor, int bias_dim) {
    const int32_t sample_feature_size = weight_tensor->GetTensorShape()->GetDimSize(0);
    const int32_t embedding_size = weight_tensor->GetTensorShape()->GetDimSize(1);

    T* output_flat = reinterpret_cast<T*>(output_tensor->GetData());
    const T* weight_flat = reinterpret_cast<const T*>(weight_tensor->GetData());
    const int32_t* index_flat = reinterpret_cast<const int32_t*>(index_tensor->GetData());
    const T* cross_mean_sum_flat = reinterpret_cast<const T*>(cross_mean_sum_tensor->GetData());
    const T* grad_flat = reinterpret_cast<const T*>(grad_tensor->GetData());

    for (int32_t s = 0; s < sample_feature_size; s++) {
        int32_t sample_id = index_flat[s];
        const T* cross_mean_sum_ptr = cross_mean_sum_flat + sample_id * embedding_size;
        const T* grad_ptr = grad_flat + sample_id * embedding_size;
        for (int32_t k = 0; k < bias_dim; k++) {
            output_flat[s * embedding_size + k] = grad_ptr[k];
        }
        for (int32_t k = bias_dim; k < embedding_size; k++) {
            T weight_value = weight_flat[s * embedding_size + k];
            output_flat[s * embedding_size + k] =
                (cross_mean_sum_ptr[k] - weight_value) * grad_ptr[k];
        }
    }

    return 0;
}

template <typename T>
std::uint32_t ComputeSparseFMGrad(const CpuKernelContext &ctx) {
    int bias_dim = 1;
    const AttrValue* attr = ctx.GetAttr(BIAS_DIM);
    if (attr != nullptr) {
        bias_dim = attr->GetInt();
    }

    const Tensor* weight_tensor = ctx.Input(0);
    const Tensor* index_tensor = ctx.Input(1);
    const Tensor* cross_mean_sum_tensor = ctx.Input(2);
    const Tensor* cross_mean_square_sum_tensor = ctx.Input(3);
    const Tensor* grad_tensor = ctx.Input(4);
    Tensor* output_tensor = ctx.Output(0);

    return DoComputeSparseFMGrad<T>(weight_tensor, index_tensor, cross_mean_sum_tensor,
        cross_mean_square_sum_tensor, grad_tensor, output_tensor, bias_dim);
}

uint32_t SparseFMGradCpuKernel::Compute(CpuKernelContext &ctx)
{
    DataType input_type = ctx.Input(0)->GetDataType();
    switch (input_type) {
        case DT_FLOAT:
            return ComputeSparseFMGrad<float>(ctx);
        case DT_DOUBLE:
            return ComputeSparseFMGrad<double>(ctx);
        default:
            return 1;
    }

    return 0;
}

REGISTER_CPU_KERNEL(SPARSE_FM_GRAD, SparseFMGradCpuKernel);
} // namespace aicpu
