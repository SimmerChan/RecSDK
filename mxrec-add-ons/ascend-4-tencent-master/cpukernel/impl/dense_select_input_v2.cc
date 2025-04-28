
/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
 * Description: implement of SparseFM
 */
#include "dense_select_input_v2.h"
#include <securec.h>
#include "key_to_pos_map.h"
#include "cpu_tensor.h"        // Tensor定义以及相关方法
#include "cpu_tensor_shape.h"  // Tensor shape的定义以及相关方法
#include "cpu_types.h"         // 数据类型以及格式等定义
#include "cpu_attr_value.h"    // AttrValue定义及相关方法
#include "unsupported/Eigen/CXX11/Tensor"
#include <algorithm>

namespace  {
const char *DENSESELECTINPUTV2 = "DenseSelectInputV2";
}

namespace aicpu  {



inline bool IsPart2(int32_t part) { return part == 2; }

template <typename T>
void imlOp(const std::vector<Tensor*> embedding_list, const std::vector<int64_t>& field_list, const std::vector<int64_t>& select_parts,
            const std::vector<int64_t>& select_fileds, const std::vector<int64_t>& embedding_sizes,
            const std::vector<int64_t>& select_indexes, const std::vector<int64_t>& select_fields_indexes,
                    Tensor* output_tensor){
                    int64_t batch_size = embedding_list[0]->GetTensorShape()->GetDimSize(0);
                    T* output_data = reinterpret_cast<T*>(output_tensor->GetData());
                    for (size_t i = 0; i < select_indexes.size(); ++i) {
                        int64_t index = select_indexes[i];
                        int64_t field_index = select_fields_indexes[i];
                        int64_t embedding_size = embedding_sizes[field_index];
                        const Tensor* tensor = embedding_list[index];
                        auto tensor_shape = tensor->GetTensorShape();
                        int64_t input_embedding_size = tensor_shape->GetDimSize(tensor_shape->GetDims() - 1);
                        
                        if (input_embedding_size != embedding_size) {
                            for (int64_t n = 0; n < batch_size; ++n) {
                                std::fill_n(output_data + n * output_tensor->GetTensorShape()->GetDimSize(1), embedding_size, T(0.));
                            }
                        } else {
                            T* select_embedding_data = reinterpret_cast<T*>(tensor->GetData());
                            for (int64_t n = 0; n < batch_size; ++n) {
                                std::copy_n(select_embedding_data + n * embedding_size, embedding_size, output_data + n * output_tensor->GetTensorShape()->GetDimSize(1));
                            }
                        } 
                        output_data += embedding_size;
                    }
                }

uint32_t DenseSelectInputV2CpuKernel :: Compute(CpuKernelContext &ctx) {
        auto field_num_ = ctx.GetAttr("field_num")->GetInt();
        auto fields_ = ctx.GetAttr("fields")->GetListInt();
        auto select_parts_ = ctx.GetAttr("select_parts")->GetListInt();
        auto select_fields_ = ctx.GetAttr("select_fields")->GetListInt();
        auto embedding_sizes_ = ctx.GetAttr("embedding_sizes")->GetListInt();
        int64_t output_part1_embedding_size_ = 0;
        int64_t output_part2_embedding_size_ = 0;
        std::vector<int64_t> part1_select_fields_indexes_;
        std::vector<int64_t> part2_select_fields_indexes_;
        std::vector<int64_t> part1_select_indexes_;
        std::vector<int64_t> part2_select_indexes_;
        pctr::util::KeyToPosMap<int64_t> fields_map_;
        std::vector<Tensor *> outputTensor;
        fields_map_.Build(fields_);
        // group index by part type
        for (size_t i = 0; i < select_parts_.size(); ++i) {
            int64_t index;
            if (fields_map_.Find(select_fields_[i], &index)){
                if (IsPart2(select_parts_[i])) {
                    part2_select_indexes_.emplace_back(index);
                    output_part2_embedding_size_ += embedding_sizes_[i];
                    part2_select_fields_indexes_.emplace_back(i);
                } else {
                    part1_select_indexes_.emplace_back(index);
                    output_part1_embedding_size_ += embedding_sizes_[i];
                    part1_select_fields_indexes_.emplace_back(i);
                }
            }
        }
        std::vector<Tensor *> embedding_list;
        for (auto input_num = 0; input_num < field_num_; input_num ++){
            Tensor *embeddings_tensor = ctx.Input(input_num);
            embedding_list.push_back(embeddings_tensor);
        }
        auto batch_size = embedding_list[0]->GetTensorShape()->GetDimSize(0);
        Tensor* output_part1_tensor = ctx.Output(0);
        Tensor* output_part2_tensor = ctx.Output(1);
        auto output_part1_shape = output_part1_tensor->GetTensorShape();
        auto output_part2_shape = output_part2_tensor->GetTensorShape();
        if (output_part1_embedding_size_ > 0){
            std::vector<int64_t> output_part1_dims = {batch_size, output_part1_embedding_size_};
            output_part1_shape->SetDimSizes(output_part1_dims);
        } else{
            std::vector<int64_t> output_part1_dims = {1};
            output_part1_shape->SetDimSizes(output_part1_dims);
        }
        if (output_part2_embedding_size_ > 0){
            std::vector<int64_t> output_part2_dims = {batch_size, output_part2_embedding_size_};
            output_part2_shape->SetDimSizes(output_part2_dims);
        } else{
            std::vector<int64_t> output_part2_dims = {1};
            output_part2_shape->SetDimSizes(output_part2_dims);
        }
        output_part1_tensor->SetTensorShape(output_part1_shape.get());
        output_part2_tensor->SetTensorShape(output_part2_shape.get());

        DataType input_type = ctx.Input(0)->GetDataType();
        switch(input_type){
            case DT_FLOAT:
                imlOp<float>(embedding_list, fields_, select_parts_,
                                            select_fields_, embedding_sizes_,
                                            part1_select_indexes_, part1_select_fields_indexes_,
                                            output_part1_tensor);
                imlOp<float>(embedding_list, fields_, select_parts_,
                                            select_fields_, embedding_sizes_,
                                            part2_select_indexes_, part2_select_fields_indexes_,
                                            output_part2_tensor);
                break;
            case DT_FLOAT16:
                imlOp<Eigen::half>(embedding_list, fields_, select_parts_,
                                            select_fields_, embedding_sizes_,
                                            part1_select_indexes_, part1_select_fields_indexes_,
                                            output_part1_tensor);
                imlOp<Eigen::half>(embedding_list, fields_, select_parts_,
                                            select_fields_, embedding_sizes_,
                                            part2_select_indexes_, part2_select_fields_indexes_,
                                            output_part2_tensor);
                break;
            case DT_DOUBLE:
                imlOp<double>(embedding_list, fields_, select_parts_,
                                            select_fields_, embedding_sizes_,
                                            part1_select_indexes_, part1_select_fields_indexes_,
                                            output_part1_tensor);
                imlOp<double>(embedding_list, fields_, select_parts_,
                                            select_fields_, embedding_sizes_,
                                            part2_select_indexes_, part2_select_fields_indexes_,
                                            output_part2_tensor);
                break;
            default:
                return -1;
        }
        return 0;
}     

REGISTER_CPU_KERNEL(DENSESELECTINPUTV2, DenseSelectInputV2CpuKernel);
} // namespace aicpu
