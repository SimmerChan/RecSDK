/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
 * Description: implement of SparseFieldsConcatV2Grad
 */
#include "sparse_fields_concat_v2_grad_kernels.h"

#include <securec.h>
#include "unsupported/Eigen/CXX11/Tensor"

#include "cpu_tensor.h"        // Tensor定义以及相关方法
#include "cpu_tensor_shape.h"  // Tensor shape的定义以及相关方法
#include "cpu_types.h"         // 数据类型以及格式等定义
#include "cpu_attr_value.h"    // AttrValue定义及相关方法

namespace  {
    const char * SPARSE_FIELDS_CONCAT_V2_GRAD = "SparseFieldsConcatV2Grad";
}


namespace aicpu {
    template <typename T>
    inline void axpy(const size_t N, const T alpha, const T* X, T* Y) {
        for (size_t i = 0; i < N; ++i) {
            Y[i] += X[i] * alpha;
        }
    }

    template <typename T>
    uint32_t DoComputeSparseFieldsConcatV2Grad(const Tensor* weight_tensor,
    const Tensor* field_tensor, const Tensor* index_tensor, const Tensor* grad_part1_tensor,
    const Tensor* grad_part2_tensor, const Tensor* keys_per_field_tensor, Tensor* output_tensor,
    int fw_field_num_, std::vector<int64_t> & part1_fields_list_, std::vector<int64_t> & part2_fields_list_) {
        std::vector<int32_t> part1_fields_map_;
        std::vector<int32_t> part2_fields_map_;
        part1_fields_map_.resize(fw_field_num_, -1);
        for (int32_t i = 0; i < part1_fields_list_.size(); ++i)
        {
            int32_t part1_field = part1_fields_list_[i] - 1;
            if ((part1_field >= 0) && (part1_field < fw_field_num_) &&
            (part1_fields_map_[part1_field] < 0))
            {
                part1_fields_list_[i] = part1_field;
                part1_fields_map_[part1_field] = i;
            }
            else
            {
                printf("Part1 filed %ld us either an invalid field or duplicated field"
                "in attribute `part1_fields`", part1_fields_list_[i]);
            }
        }

        part2_fields_map_.resize(fw_field_num_, -1);
        for (int32_t i = 0; i < part2_fields_list_.size(); ++i)
        {
            int32_t part2_field = part2_fields_list_[i] - 1;
            if ((part2_field >= 0) && (part2_field < fw_field_num_) &&
            (part2_fields_map_[part2_field] < 0))
            {
                part2_fields_list_[i] = part2_field;
                part2_fields_map_[part2_field] = i;
            }
            else
            {
                printf("Part2 filed %ld us either an invalid field or duplicated field "
                "in attribute `part2_fields`", part2_fields_list_[i]);
            }
        }

        const int32_t sample_feature_size = weight_tensor->GetTensorShape()->GetDimSize(0);
        const int32_t embedding_size = weight_tensor->GetTensorShape()->GetDimSize(1) * weight_tensor->GetTensorShape()->GetDimSize(2);
        const T * weight_data = reinterpret_cast < T* > (weight_tensor->GetData());
        const T * grad_part1_data = reinterpret_cast < T* > (grad_part1_tensor->GetData());
        const T * grad_part2_data = reinterpret_cast < T* > (grad_part2_tensor->GetData());
        const int32_t * feat_counter = reinterpret_cast < int32_t* > (keys_per_field_tensor->GetData());
        T * output_data = reinterpret_cast < T* > (output_tensor->GetData());
        auto field_flat = reinterpret_cast < int32_t* > (field_tensor->GetData());
        auto index_flat = reinterpret_cast < int32_t* > (index_tensor->GetData());
        const int32_t index_dim_size_1 = (index_tensor->GetTensorShape()->GetDims() == 2) ? index_tensor->GetTensorShape()->GetDimSize(1) : 1;
        const int32_t seq_size = (index_tensor->GetTensorShape()->GetDims() == 2) ? grad_part2_tensor->GetTensorShape()->GetDimSize(1) : 1;
        const int32_t batch_size = grad_part2_tensor->GetTensorShape()->GetDimSize(0);
        uint64_t outputSize = output_tensor->GetDataSize();
        (void)memset_s(output_data, outputSize, 0x00, outputSize);
        int32_t s = 0;
        for (int32_t n = 0; n < batch_size; ++n)
        {
            for (; (s < sample_feature_size && index_flat[s * index_dim_size_1] == n); ++s)
            {
                const int32_t fw_field = field_flat[s * 2 + 1] - 1;
                if (fw_field < 0 || fw_field >= fw_field_num_)
                    continue;
                const int32_t sample_id = index_flat[s * index_dim_size_1];
                int32_t field_index = part1_fields_map_[fw_field];
                if (field_index >= 0)
                {
                    const int32_t base_offset = sample_id * grad_part1_tensor->GetTensorShape()->GetDimSize(1);
                    const int32_t * feat_counter_ptr = feat_counter + sample_id * fw_field_num_ * seq_size;
                    const T * grad_ptr = grad_part1_data + base_offset + field_index * embedding_size;
                    T * output_ptr = output_data + s * embedding_size;
                    if (feat_counter_ptr[fw_field * seq_size] != 0)
                    {
                        const T scale_factor = T(1.) / feat_counter_ptr[fw_field * seq_size];

                        axpy(embedding_size, scale_factor, grad_ptr, output_ptr);
                    }
                }
                field_index = part2_fields_map_[fw_field];
                if (field_index >= 0)
                {
                    int32_t base_offset = sample_id * grad_part2_tensor->GetTensorShape()->GetDimSize(1);
                    const int32_t * feat_counter_ptr = feat_counter;
                    if (grad_part2_tensor->GetTensorShape()->GetDims() > 2)
                    {
                        int32_t seq_id = index_flat[s * index_dim_size_1 + 1];
                        base_offset += seq_id;
                        base_offset *= grad_part2_tensor->GetTensorShape()->GetDimSize(2);
                        feat_counter_ptr += (sample_id * fw_field_num_ + fw_field) * seq_size + seq_id;
                    }
                    else
                    {
                        feat_counter_ptr += sample_id * fw_field_num_ + fw_field;
                    }
                    const T * grad_ptr = grad_part2_data + base_offset + field_index * embedding_size;
                    T * output_ptr = output_data + s * embedding_size;
                    if (*feat_counter_ptr != 0)
                    {
                        const T scale_factor = T(1.) / *feat_counter_ptr;
                        axpy(embedding_size, scale_factor, grad_ptr, output_ptr);
                    }
                }
            }
            // for loop of sample_feature_size
        }
        // for loop of batch_size
        return 0;

    }

    template <typename T>
    uint32_t ComputeSparseFieldsConcatV2Grad(CpuKernelContext & ctx) {
        int fw_field_num_ = ctx.GetAttr("fw_field_num")->GetInt();
        std::vector < int64_t > part1_fields = ctx.GetAttr("part1_fields")->GetListInt();
        std::vector < int64_t > part2_fields = ctx.GetAttr("part2_fields")->GetListInt();

        const Tensor* weight_tensor = ctx.Input(0);
        const Tensor* field_tensor = ctx.Input(1);
        const Tensor* index_tensor = ctx.Input(2);
        const Tensor* grad_part1_tensor = ctx.Input(3);
        const Tensor* grad_part2_tensor = ctx.Input(4);
        const Tensor* keys_per_field_tensor = ctx.Input(5);
        Tensor* output_tensor = ctx.Output(0);

        return DoComputeSparseFieldsConcatV2Grad<T>(weight_tensor, field_tensor, index_tensor,
        grad_part1_tensor, grad_part2_tensor,
        keys_per_field_tensor, output_tensor, fw_field_num_, part1_fields, part2_fields);

        return 0;
    }

    uint32_t SparseFieldsConcatV2GradCpuKernel::Compute(CpuKernelContext & ctx)
    {

        DataType input_type = ctx.Input(0)->GetDataType();
        switch (input_type) {
            case DT_FLOAT:
            return ComputeSparseFieldsConcatV2Grad<float>(ctx);
            case DT_FLOAT16:
            return ComputeSparseFieldsConcatV2Grad<Eigen::half>(ctx);
            case DT_DOUBLE:
            return ComputeSparseFieldsConcatV2Grad<double>(ctx);
            default:
            return -1;

        }
        return 0;
    }

    REGISTER_CPU_KERNEL(SPARSE_FIELDS_CONCAT_V2_GRAD, SparseFieldsConcatV2GradCpuKernel);
}
// namespace aicpu
