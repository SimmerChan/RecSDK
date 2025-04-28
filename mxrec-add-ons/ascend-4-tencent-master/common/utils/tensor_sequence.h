/**
 * Copyright (c) 2023-2023 Huawei Technologies Co., Ltd.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef AI_CPU_NORMALIZED_TENSOR_SEQUENCE_H
#define AI_CPU_NORMALIZED_TENSOR_SEQUENCE_H

#include <vector>
#include <sstream>

#include "exe_graph/runtime/shape.h"
#include "exe_graph/runtime/tensor.h"
#include "exe_graph/runtime/tensor_data.h"
#include "graph/types.h"
#include "status.h"
#include "utils/log.h"

namespace aicpu {
class TensorSeq;
using TensorSeqPtr = std::shared_ptr<TensorSeq>;
class TensorSeq {
 public:
  TensorSeq() = default;
  explicit TensorSeq(ge::DataType elem_type) noexcept : elem_type_{elem_type} {}

  struct TensorRef {
    gert::TensorData tensor_addr_;
    gert::Shape tensor_shape_;
  };

  using const_iterator = std::vector<TensorRef>::const_iterator;

  // Sets the element type after construction.
  // Expects sequence to be empty at the time.
  KernelStatus SetType(ge::DataType elem_type) {
    if (!tensors_.empty()) {
      KERNEL_LOG_ERROR(
          "tensor sequence is not empty, so can't set the elem_type.");
      return KERNEL_STATUS_PARAM_INVALID;
    }
    elem_type_ = elem_type;
    return KERNEL_STATUS_OK;
  }

  KernelStatus SetElements(std::vector<TensorRef>&& tensors) {
    if (!tensors_.empty()) {
      KERNEL_LOG_ERROR("tensor sequence is not empty, so can't set elements.");
      return KERNEL_STATUS_PARAM_INVALID;
    }
    tensors_ = std::move(tensors);
    return KERNEL_STATUS_OK;
  }

  ge::DataType DataType() const noexcept { return elem_type_; }

  bool IsSameDataType(const TensorSeq& tensor_seq) const noexcept {
    return elem_type_ == tensor_seq.elem_type_;
  }

  size_t Size() const noexcept { return tensors_.size(); }

  // Suitable for for range loop
  const_iterator begin() const noexcept { return tensors_.cbegin(); }

  const_iterator end() const noexcept { return tensors_.cend(); }

  bool ValidateSeqIdx(int64_t index) const {
    bool ret = false;
    int64_t size = static_cast<int64_t>(tensors_.size());
    if (index < 0) {
      ret = (index <= -1) && (index >= -size);
    } else {
      ret = index < size;
    }

    if (!ret) {
      KERNEL_LOG_ERROR("input index %ld is not valid, sequence's size %lu",
                       index, tensors_.size());
    }
    return ret;
  }

  // Get by index
  const TensorRef* Get(int64_t index) const {
    if (!ValidateSeqIdx(index)) {
      return nullptr;
    }

    if (index < 0) {
      index += tensors_.size();
    }

    return &tensors_[index];
  }

  KernelStatus Add(TensorRef&& tensor, ge::DataType data_type) {
    if (elem_type_ != data_type) {
      KERNEL_LOG_ERROR(
          "The data type of add tensor is not equal with element type of "
          "tensor sequence, the input data type is [%u] , tensor sequence's "
          "element type is [%u].",
          data_type, elem_type_);
      return KERNEL_STATUS_PARAM_INVALID;
    }
    tensors_.push_back(std::move(tensor));
    return KERNEL_STATUS_OK;
  }

  std::string ShapeToString(gert::Shape shape) {
    size_t dims = shape.GetDimNum();
    if (dims == 0) {
      return "";
    }

    std::stringstream ss;
    ss << "[";
    ss << shape[0];
    for (size_t i = 1; i < dims; i++) {
      ss << "," << shape[i];
    }
    ss << " ]";
    return ss.str();
  }

  KernelStatus Add(const gert::Tensor& tensor) {
    auto data_type = tensor.GetDataType();
    if (elem_type_ != data_type) {
      KERNEL_LOG_ERROR(
          "The data type of add tensor is not equal with element type of "
          "tensor sequence, the input data type is [%u] , tensor sequence's "
          "element type is [%u].",
          data_type, elem_type_);
      return KERNEL_STATUS_PARAM_INVALID;
    }
    TensorRef tensor_ref;
    if (tensor_ref.tensor_addr_.ShareFrom(tensor.GetTensorData()) !=
        ge::GRAPH_SUCCESS) {
      KERNEL_LOG_ERROR("Create tensor ref failed");
      return KERNEL_STATUS_PARAM_INVALID;
    }
    auto shape = tensor.GetStorageShape();
    tensor_ref.tensor_shape_.SetDimNum(shape.GetDimNum());
    for (size_t index = 0; index < shape.GetDimNum(); ++index) {
      tensor_ref.tensor_shape_.SetDim(index, shape.GetDim(index));
    }
    tensors_.push_back(std::move(tensor_ref));
    KERNEL_LOG_DEBUG(
        "Add tensor success, data type is %u, tensor "
        "size is %lu",
        data_type, tensor.GetSize());
    return KERNEL_STATUS_OK;
  }

  KernelStatus Add(const ge::DataType data_type,
                   const gert::TensorData& tensor_data,
                   const gert::StorageShape& storage_shape) {
    if (elem_type_ != data_type) {
      KERNEL_LOG_ERROR(
          "The data type of add tensor is not equal with element type of "
          "tensor sequence, the input data type is [%u] , tensor sequence's "
          "element type is [%u].",
          data_type, elem_type_);
      return KERNEL_STATUS_PARAM_INVALID;
    }
    TensorRef tensor_ref;
    if (tensor_ref.tensor_addr_.ShareFrom(tensor_data) != ge::GRAPH_SUCCESS) {
      KERNEL_LOG_ERROR("Create tensor ref failed");
      return KERNEL_STATUS_PARAM_INVALID;
    }
    auto shape = storage_shape.GetOriginShape();
    tensor_ref.tensor_shape_.SetDimNum(shape.GetDimNum());

    for (size_t index = 0; index < shape.GetDimNum(); ++index) {
      tensor_ref.tensor_shape_.SetDim(index, shape.GetDim(index));
    }
    tensors_.push_back(std::move(tensor_ref));
    KERNEL_LOG_DEBUG("tensor sequence add tensor ref success, tensor shape is %s",
                     ShapeToString(shape).c_str());
    return KERNEL_STATUS_OK;
  }

  KernelStatus Add(const gert::Tensor& tensor, int64_t index) {
    auto data_type = tensor.GetDataType();
    if (elem_type_ != data_type) {
      KERNEL_LOG_ERROR(
          "The data type of add tensor is not equal with element type of "
          "tensor sequence, the input data type is [%u] , tensor sequence's "
          "element type is [%u].",
          data_type, elem_type_);
      return KERNEL_STATUS_PARAM_INVALID;
    }

    if (!ValidateSeqIdx(index)) {
      return KERNEL_STATUS_PARAM_INVALID;
    }

    if (index < 0) {
      index += tensors_.size();
    }

    TensorRef tensor_ref;
    if (tensor_ref.tensor_addr_.ShareFrom(tensor.GetTensorData()) !=
        ge::GRAPH_SUCCESS) {
      KERNEL_LOG_ERROR("Create tensor ref failed");
      return KERNEL_STATUS_PARAM_INVALID;
    }
    // optimize how to consturct shape
    auto shape = tensor.GetStorageShape();
    tensor_ref.tensor_shape_.SetDimNum(shape.GetDimNum());
    for (size_t index = 0; index < shape.GetDimNum(); ++index) {
      tensor_ref.tensor_shape_.SetDim(index, shape.GetDim(index));
    }
    tensors_.insert(tensors_.begin() + index, std::move(tensor_ref));
    KERNEL_LOG_DEBUG("Add tensor success, index is %ld, tensor size is %lu",
                     index, tensor.GetSize());
    return KERNEL_STATUS_OK;
  }

  KernelStatus Add(const ge::DataType data_type,
                   const gert::TensorData& tensor_data,
                   const gert::StorageShape& storage_shape, int64_t index) {
    if (elem_type_ != data_type) {
      KERNEL_LOG_ERROR(
          "The data type of add tensor is not equal with element type of "
          "tensor sequence, the input data type is [%u] , tensor sequence's "
          "element type is [%u].",
          data_type, elem_type_);
      return KERNEL_STATUS_PARAM_INVALID;
    }

    if (!ValidateSeqIdx(index)) {
      return KERNEL_STATUS_PARAM_INVALID;
    }

    if (index < 0) {
      index += tensors_.size();
    }

    TensorRef tensor_ref;
    if (tensor_ref.tensor_addr_.ShareFrom(tensor_data) != ge::GRAPH_SUCCESS) {
      KERNEL_LOG_ERROR("Create tensor ref failed");
      return KERNEL_STATUS_PARAM_INVALID;
    }
    auto shape = storage_shape.GetOriginShape();
    tensor_ref.tensor_shape_.SetDimNum(shape.GetDimNum());

    for (size_t index = 0; index < shape.GetDimNum(); ++index) {
      tensor_ref.tensor_shape_.SetDim(index, shape.GetDim(index));
    }
    tensors_.insert(tensors_.begin() + index, std::move(tensor_ref));
    KERNEL_LOG_DEBUG(
        "tensor sequence add ref tensor success, index is %ld, tensor shape is %s",
        index, ShapeToString(shape).c_str());
    return KERNEL_STATUS_OK;
  }

  KernelStatus Erase(int64_t index) {
    if (!ValidateSeqIdx(index)) {
      return KERNEL_STATUS_PARAM_INVALID;
    }

    if (index < 0) {
      index += tensors_.size();
    }
    tensors_.erase(tensors_.begin() + index);
    return KERNEL_STATUS_OK;
  }

  void Reserve(size_t capacity) { tensors_.reserve(capacity); }

 private:
  ge::DataType elem_type_{ge::DT_FLOAT};
  std::vector<TensorRef> tensors_;
};
}  // namespace aicpu
#endif