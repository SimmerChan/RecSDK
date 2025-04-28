
/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
 * Description: implement of SparseFieldsConcatV2
 */
// #include "log.h"
#include <algorithm>
#include <vector>
#include <iostream>
#include "cpu_tensor.h"        // Tensor定义以及相关方法
#include "cpu_tensor_shape.h"  // Tensor shape的定义以及相关方法
#include "cpu_types.h"         // 数据类型以及格式等定义
#include "cpu_attr_value.h"    // AttrValue定义及相关方法

#include "sparse_fields_concat_v2_kernels.h"
#include "unsupported/Eigen/CXX11/Tensor"

namespace  {
const char *KSparseFieldsConcatV2 = "SparseFieldsConcatV2";
}

namespace aicpu  {

template <typename T>
inline void tfol_add(const size_t N, const T* X1, const T* X2, T* Y) {
    for (size_t i = 0; i < N; ++i) {
        Y[i] = X1[i] + X2[i];
    }
}

template <typename T>
inline void tfol_scale(const size_t N, const T alpha, const T* X, T* Y) {
    for (size_t i = 0; i < N; ++i) {
        Y[i] = X[i] * alpha;
    }
}

template <typename T>
std::uint32_t DoComputeSparseFieldsConcatV2(int32_t fw_field_num,
                                            const std::vector<int64_t>& part1_index_to_field,
                                            const std::vector<int64_t>& part2_index_to_field,
                                            const std::vector<int64_t>& part1_field_to_index,
                                            const std::vector<int64_t>& part2_field_to_index,
                                            const Tensor* weight_tensor, const Tensor* field_tensor,
                                            const Tensor* index_tensor, Tensor* output_part1_tensor,
                                            Tensor* output_part2_tensor, Tensor* keys_per_field_tensor) {
    // KERNEL_LOG_INFO(
    //     "DoComputeSparseFieldsConcatV2 begin, fw_field_num: [%ld]",
    //     fw_field_num);
    const int64_t sample_feature_size = weight_tensor->GetTensorShape()->GetDimSize(0);
    const int64_t embedding_size = weight_tensor->GetTensorShape()->GetDimSize(1) * \
                                   weight_tensor->GetTensorShape()->GetDimSize(2);
    int32_t* field_flat = reinterpret_cast<int32_t*>(field_tensor->GetData());
    int32_t* index_flat = reinterpret_cast<int32_t*>(index_tensor->GetData());
    // KERNEL_LOG_INFO(
    //     "sample_feature_size: [%lld], embedding_size: [%lld], batch_size: [%lld]",
    //     sample_feature_size, embedding_size, batch_size);

    const T* weight_data = reinterpret_cast<const T*>(weight_tensor->GetData());
    T* output_part1_data = reinterpret_cast<T*>(output_part1_tensor->GetData());
    T* output_part2_data = reinterpret_cast<T*>(output_part2_tensor->GetData());
    int32_t* feat_counter = reinterpret_cast<int32_t*>(keys_per_field_tensor->GetData());

    const int32_t index_dim_size_1 = (static_cast<const int32_t>(index_tensor->GetTensorShape()->GetDims())
                                      == 2) ? index_tensor->GetTensorShape()->GetDimSize(1) : 1; // 2
    const int32_t seq_size = (static_cast<const int32_t>(index_tensor->GetTensorShape()->GetDims())
                              == 2) ? output_part2_tensor->GetTensorShape()->GetDimSize(1) : 1; // 2?
    // KERNEL_LOG_INFO(
    //     "feat_counter: [%ld], field_flat: [%ld], index_flat: [%ld], index_dim_size_1: [%ld], seq_size: [%ld]",
    //     &feat_counter, &field_flat, &index_flat, index_dim_size_1, seq_size);

    int32_t part1_fields_size = part1_index_to_field.size();
    int32_t part2_fields_size = part2_index_to_field.size();
    // KERNEL_LOG_INFO(
    //     "part1_fields_size: [%ld], part2_fields_size: [%ld]",
    //     part1_fields_size, part2_fields_size);
    int32_t index_rank = index_tensor->GetTensorShape()->GetDims();
    int32_t batch_size;
    int32_t seq_size_for_output;
    if (index_rank == 2) {
        batch_size = index_flat[sample_feature_size];
        seq_size_for_output = index_flat[sample_feature_size*2+1];
        std::fill_n(output_part1_data, batch_size*part1_fields_size*embedding_size, T(0.));
        std::fill_n(output_part2_data, batch_size*seq_size_for_output*part2_fields_size*embedding_size, T(0.));
        std::fill_n(feat_counter, batch_size*seq_size_for_output*fw_field_num, 0);
    } else {
        batch_size = index_flat[sample_feature_size];
        std::fill_n(output_part1_data, batch_size*part1_fields_size*embedding_size, T(0.));
        std::fill_n(output_part2_data, batch_size*part2_fields_size*embedding_size, T(0.));
        std::fill_n(feat_counter, batch_size*fw_field_num, 0);
    }

    std::vector<std::vector<T>> output_part1_substitute;
    output_part1_substitute.resize(5);

    for (int32_t i=0; i < output_part1_substitute.size(); i++) {
        output_part1_substitute[i].resize(16, static_cast<T>(-100.0));
    }
    int32_t substitute_index = 0;
    int32_t check_base_offset = 0;
    int32_t check_base_offset_1 = 0;
    int32_t check_sample_id = 0;
    int32_t s = 0;
    for (int32_t n = 0; n < batch_size; ++n) {
        for (; (s < sample_feature_size && index_flat[s * index_dim_size_1] == n); ++s) {
            const int32_t fw_field = field_flat[s * 2 + 1] - 1;
            if (fw_field < 0 || fw_field >= fw_field_num) continue;
            const int32_t sample_id = index_flat[s * index_dim_size_1];
            check_sample_id = sample_id;
            int32_t field_index = part1_field_to_index[fw_field];
            if (field_index >= 0) {
                const int32_t base_offset = static_cast<const int32_t>(sample_id *
                                            output_part1_tensor->GetTensorShape()->GetDimSize(1));
                const T* weight_ptr = weight_data + s * embedding_size;
                T* output_ptr = output_part1_data + base_offset + field_index * embedding_size;
                substitute_index += 1;
                check_base_offset = base_offset;
                tfol_add(embedding_size, output_ptr, weight_ptr, output_ptr);
                ++feat_counter[(sample_id * fw_field_num + fw_field) * seq_size];
            }
            field_index = part2_field_to_index[fw_field];
            if (field_index >= 0) {
                int32_t base_offset = static_cast<const int32_t>(sample_id *
                                      output_part2_tensor->GetTensorShape()->GetDimSize(1));
                if (output_part2_tensor->GetTensorShape()->GetDims() > 2) {
                    int32_t seq_id = index_flat[s * index_dim_size_1 + 1];
                    base_offset += seq_id;
                    base_offset *= static_cast<const int32_t>(output_part2_tensor->\
                                                             GetTensorShape()->GetDimSize(2));
                    ++feat_counter[(sample_id * fw_field_num + fw_field) * seq_size + seq_id];
                } else {
                    ++feat_counter[sample_id * fw_field_num + fw_field];
                }
                const T* weight_ptr = weight_data + s * embedding_size;
                T* output_ptr = output_part2_data + base_offset + field_index * embedding_size;
                tfol_add(embedding_size, output_ptr, weight_ptr, output_ptr);
            }
            // KERNEL_LOG_INFO("n: [%ld], s: [%ld]", n, s);
        } // for loop of sample_feature_size

        const int32_t base_offset_1 = n * static_cast<const int32_t>(output_part1_tensor->\
                                                               GetTensorShape()->GetDimSize(1));
        const int32_t* feat_counter_ptr = feat_counter + n * fw_field_num * seq_size;
        for (int32_t i = 0; i < fw_field_num; ++i) {
            int32_t field_index = part1_field_to_index[i];
            if (field_index >= 0) {
                T* output_ptr = output_part1_data + base_offset_1 + field_index * embedding_size;
                if (feat_counter_ptr[i * seq_size] != 0) {
                    const T scale_factor = T(1.) / feat_counter_ptr[i * seq_size];
                    tfol_scale(embedding_size, scale_factor, output_ptr, output_ptr);
                    substitute_index += 1;
                    check_base_offset_1 = base_offset_1;
                }
            }
        }
        // KERNEL_LOG_INFO(
        // "output_part1_data: [%s]",
        // VectorToString(&output_part1_data).c_str());
        if (output_part2_tensor->GetTensorShape()->GetDims() > 2) {
            const int32_t concat_embedding_size = static_cast<int32_t>(output_part2_tensor->\
                                                                       GetTensorShape()->GetDimSize(2));
            const int32_t base_offset_2 = n * seq_size * concat_embedding_size;
            const int32_t* feat_counter_ptr = feat_counter + n * fw_field_num * seq_size;
            for (int32_t j = 0; j < seq_size; ++j) {
                for (int32_t i = 0; i < fw_field_num; ++i) {
                    int32_t field_index = part2_field_to_index[i];
                    if (field_index >= 0) {
                        T* output_ptr =
                            output_part2_data + base_offset_2 + j * concat_embedding_size +
                            field_index * embedding_size;
                        if (feat_counter_ptr[i * seq_size + j] != 0) {
                            const T scale_factor = T(1.) / feat_counter_ptr[i * seq_size + j];
                            tfol_scale(embedding_size, scale_factor,
                                       output_ptr, output_ptr);
                        }
                    }
                }
            }
            // KERNEL_LOG_INFO(
            // "in if output_part2_data: [%s]",
            // VectorToString(&output_part2_data).c_str());
        } else {
            const int32_t base_offset_2 = n * static_cast<const int32_t>(output_part2_tensor->\
                                                                        GetTensorShape()->GetDimSize(1));
            const int32_t* feat_counter_ptr = feat_counter + n * fw_field_num;
            for (int32_t i = 0; i < fw_field_num; ++i) {
                int32_t field_index = part2_field_to_index[i];
                if (field_index >= 0) {
                    T* output_ptr =
                        output_part2_data + base_offset_2 +
                        field_index * embedding_size;
                    if (feat_counter_ptr[i] != 0) {
                        const T scale_factor = T(1.) / feat_counter_ptr[i];
                        tfol_scale(embedding_size, scale_factor, output_ptr, output_ptr);
                    }
                }
            }
            // KERNEL_LOG_INFO(
            // "in else output_part2_data: [%s]",
            // VectorToString(&output_part2_data).c_str());
        }
    } // for loop of batch_size
    return 0;
}

void index_field_lookup_table(int64_t& fw_field_num,
                              std::vector<int64_t>& part1_index_to_field,
                              std::vector<int64_t>& part2_index_to_field,
                              std::vector<int64_t>& part1_field_to_index,
                              std::vector<int64_t>& part2_field_to_index) {
    // KERNEL_LOG_INFO(
    //     "in index_field_lookup_table begin, fw_field_num[%ld], part1_index_to_field: [%s], part2_index_to_field: [%s]",
    //     fw_field_num, VectorToString(part1_index_to_field).c_str(), VectorToString(part2_index_to_field).c_str());
    for (uint64_t i = 0; i < part1_index_to_field.size(); ++i) {
        int64_t part1_field = part1_index_to_field[i] - 1;
        if ((part1_field >= 0) && (part1_field < fw_field_num) &&
            (part1_field_to_index[part1_field] < 0)) {
            part1_index_to_field[i] = part1_field; // [0, 2]
            part1_field_to_index[part1_field] = i; // [0, -1, 1, -1]
        }
        // else {
        //     KERNEL_LOG_INFO("Part1 field is either an invalid or duplicated number");
        // }
    }
    for (uint64_t i = 0; i < part2_index_to_field.size(); ++i) {
        int64_t part2_field = part2_index_to_field[i] - 1;
        if ((part2_field >= 0) && (part2_field < fw_field_num) &&
            (part2_field_to_index[part2_field] < 0)) {
            part2_index_to_field[i] = part2_field; // [1, 3]
            part2_field_to_index[part2_field] = i; // [-1, 0, -1, 1]
        }
        // else {
        //     KERNEL_LOG_INFO("Part2 field is either an invalid or duplicated number");
        // }
    }
    // KERNEL_LOG_INFO(
    //     "leave index_field_lookup_table, part1_field_to_index: [%s], part2_field_to_index: [%s]",
    //     VectorToString(part1_field_to_index).c_str(), VectorToString(part2_field_to_index).c_str());
}

template <typename T>
std::uint32_t ComputeSparseFieldsConcatV2(const CpuKernelContext &ctx) {
    // KERNEL_LOG_INFO("ComputeSparseFieldsConcatV2 begin");
    Tensor *weight = ctx.Input(0);
    Tensor *field = ctx.Input(1);
    Tensor *index = ctx.Input(2);
    if (weight == nullptr || field == nullptr || index == nullptr) {
        // KERNEL_LOG_INFO("some of inputs are null");
        return 1;
    }
    Tensor *output_part1 = ctx.Output(0);
    Tensor *output_part2 = ctx.Output(1);
    Tensor *keys_per_field = ctx.Output(2);
    if (output_part1 == nullptr || output_part2 == nullptr || keys_per_field == nullptr) {
        // KERNEL_LOG_INFO("some of outputs are null");
        return 1;
    }
    const AttrValue *fw_field_num_attr = ctx.GetAttr("fw_field_num");
    const AttrValue *part1_fields_attr = ctx.GetAttr("part1_fields");
    const AttrValue *part2_fields_attr = ctx.GetAttr("part2_fields");
    if (fw_field_num_attr == nullptr || part1_fields_attr == nullptr
        || part2_fields_attr == nullptr) {
        // KERNEL_LOG_INFO("some of attrs are null");
        return 1;
    }

    int32_t batch_size = static_cast<int32_t>(weight->GetTensorShape()->GetDimSize(0));
    int32_t field_num = static_cast<int32_t>(weight->GetTensorShape()->GetDimSize(1));
    int32_t embedding_size = static_cast<int32_t>(weight->GetTensorShape()->GetDimSize(2));
    int32_t fw_field_num = static_cast<int32_t>(fw_field_num_attr->GetInt());
    int32_t part1_fields_size = part1_fields_attr->ListIntSize();
    int32_t part2_fields_size = part2_fields_attr->ListIntSize();
    // KERNEL_LOG_INFO(
    //     "batch_size: [%ld], field_num: [%ld], embedding_size: [%s], fw_field_num: [%s], part1_fields_size: [%s], part2_fields_size: [%s]",
    //     batch_size, field_num, embedding_size, fw_field_num, part1_fields_size, part2_fields_size);
    auto output_part1_shape = output_part1->GetTensorShape();
    output_part1_shape->SetDimSizes({batch_size, part1_fields_size*embedding_size*field_num});
    output_part1->SetTensorShape(output_part1_shape.get());

    int32_t index_rank = index->GetTensorShape()->GetDims();
    if (index_rank == 2) {
        // KERNEL_LOG_INFO("in index_rank=2");
        auto output_part2_shape = output_part2->GetTensorShape();
        output_part2_shape->SetDimSizes({batch_size, batch_size, });
        output_part2->SetTensorShape(output_part2_shape.get());
        auto keys_per_field_shape = keys_per_field->GetTensorShape();
        keys_per_field_shape->SetDimSizes({batch_size, fw_field_num, batch_size});
        keys_per_field->SetTensorShape(keys_per_field_shape.get());
    } else {
        // KERNEL_LOG_INFO("in index_rank=1");
        auto output_part2_shape = output_part2->GetTensorShape();
        output_part2_shape->SetDimSizes({batch_size, part2_fields_size*field_num*embedding_size});
        output_part2->SetTensorShape(output_part2_shape.get());
        auto keys_per_field_shape = keys_per_field->GetTensorShape();
        keys_per_field_shape->SetDimSizes({batch_size, fw_field_num});
        keys_per_field->SetTensorShape(keys_per_field_shape.get());
    }
    std::vector<int64_t> part1_index_to_field = part1_fields_attr->GetListInt();
    std::vector<int64_t> part2_index_to_field = part2_fields_attr->GetListInt();
    std::vector<int64_t> part1_field_to_index(fw_field_num, -1);
    std::vector<int64_t> part2_field_to_index(fw_field_num, -1);
    int64_t long_fw_field_num = static_cast<int64_t>(fw_field_num);
    index_field_lookup_table(long_fw_field_num,
                             part1_index_to_field,
                             part2_index_to_field,
                             part1_field_to_index,
                             part2_field_to_index);

    // KERNEL_LOG_INFO("leave ComputeSparseFieldsConcatV2");
    return DoComputeSparseFieldsConcatV2<T>(fw_field_num, part1_index_to_field,
        part2_index_to_field, part1_field_to_index, part2_field_to_index,
        weight, field, index, output_part1, output_part2, keys_per_field);
}

std::uint32_t SparseFieldsConcatV2CpuKernel::Compute(CpuKernelContext &ctx)
{
    DataType input_type = ctx.Input(0)->GetDataType();
    switch (input_type) {
        case DT_FLOAT16:
            return ComputeSparseFieldsConcatV2<Eigen::half>(ctx);
        case DT_FLOAT:
            return ComputeSparseFieldsConcatV2<float>(ctx);
        case DT_DOUBLE:
            return ComputeSparseFieldsConcatV2<double>(ctx);
        default:
            return -1;
    }
    return 0;
}

// REGISTER_CPU_KERNEL(SPARSE_FIELDS_CONCAT_V2, SparseFieldsConcatV2CpuKernel);
REGISTER_CPU_KERNEL(KSparseFieldsConcatV2, SparseFieldsConcatV2CpuKernel);
} // namespace aicpu
