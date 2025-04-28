
/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
 * Description: implement of DenseSelectInput
 */
#include "dense_select_input_kernels.h"

#include <algorithm>
#include "key_to_pos_map.h"

namespace
{
  const char *DENSE_SELECT_INPUT = "DenseSelectInput";
}

namespace aicpu
{

  template <typename T>
  uint32_t DoComputeDenseSelectInput(const std::vector<Tensor *> &embedding_list,
                                     const Tensor *batch_size_tensor, Tensor *output_tensor,
                                     int field_num, std::vector<int64_t> &fields, std::vector<int64_t> &select_indexes,
                                     std::vector<int64_t> &embedding_sizes)
  {

    int32_t batch_size = embedding_list[0]->GetTensorShape()->GetDimSize(0);
    T *output_data = reinterpret_cast<T *>(output_tensor->GetData());
    for (size_t i = 0; i < select_indexes.size(); ++i)
    {
      int32_t index = select_indexes[i];
      int32_t embedding_size = embedding_sizes[i];

      Tensor *embedding_tensor = embedding_list[index];
      auto tensor_shape = embedding_tensor->GetTensorShape();
      int32_t input_embedding_size = tensor_shape->GetDimSize(tensor_shape->GetDims() - 1);

      if (input_embedding_size != embedding_size)
      {
        for (int32_t n = 0; n < batch_size; ++n)
        {
          std::fill_n(output_data + n * output_tensor->GetTensorShape()->GetDimSize(1), embedding_size, T(0.));
        }
      }
      else
      {
        const T *select_embedding_data = reinterpret_cast<T *>(embedding_tensor->GetData());
        for (int32_t n = 0; n < batch_size; ++n)
        {
          std::copy_n(select_embedding_data + n * embedding_size, embedding_size,
                      output_data + n * output_tensor->GetTensorShape()->GetDimSize(1));
        }
        output_data += embedding_size;
      }
    }
    return 0;
  }

  template <typename T>
  uint32_t ComputeDenseSelectInput(CpuKernelContext &ctx)
  {
    int field_num = ctx.GetAttr("field_num")->GetInt();
    std::vector<int64_t> fields = ctx.GetAttr("fields")->GetListInt();
    std::vector<int64_t> select_fields = ctx.GetAttr("select_fields")->GetListInt();
    std::vector<int64_t> embedding_sizes = ctx.GetAttr("embedding_sizes")->GetListInt();

    std::vector<Tensor *> embedding_list;
    for (auto i = 0; i < field_num; ++i)
    {
      Tensor *embeddings_tensor = ctx.Input(i);
      embedding_list.push_back(embeddings_tensor);
    }
    const Tensor *batch_size_tensor = ctx.Input(field_num);
    Tensor *output_tensor = ctx.Output(0);

    std::vector<int64_t> select_indexes;
    pctr::util::KeyToPosMap<int64_t> fields_map;
    fields_map.Build(fields);
    for (size_t i = 0; i < select_fields.size(); ++i)
    {
      int64_t index;
      if (fields_map.Find(select_fields[i], &index))
      {
          select_indexes.emplace_back(index);
      }
    }

    return DoComputeDenseSelectInput<T>(embedding_list, batch_size_tensor, output_tensor,
                                        field_num, fields, select_indexes, embedding_sizes);

    return 0;
  }

  uint32_t DenseSelectInputCpuKernel::Compute(CpuKernelContext &ctx)
  {
    DataType input_type = ctx.Input(0)->GetDataType();
    switch (input_type)
    {
    case DT_FLOAT:
      return ComputeDenseSelectInput<float>(ctx);
    case DT_DOUBLE:
      return ComputeDenseSelectInput<double>(ctx);
    default:
      return -1;
    }
    return 0;
  }
  REGISTER_CPU_KERNEL(DENSE_SELECT_INPUT, DenseSelectInputCpuKernel);
} // namespace aicpu
