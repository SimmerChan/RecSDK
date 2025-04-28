/**
 * Copyright (c) 2022-2022 Huawei Technologies Co., Ltd.  All rights reserved.
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

#ifndef AICPU_EIGENTENSOR_H
#define AICPU_EIGENTENSOR_H

#include "cpu_tensor.h"
#include "log.h"
#include "unsupported/Eigen/CXX11/Tensor"
#include "util.h"

namespace aicpu {
// Helper to define Tensor types given that the scalar is of type T.
template <typename T, int NDIMS = 1, typename IndexType = Eigen::DenseIndex>
struct TTypes {
  // Rank-<NDIMS> tensor of scalar type T.
  using Tensor = Eigen::TensorMap<Eigen::Tensor<T, NDIMS, Eigen::RowMajor, IndexType>,
                           Eigen::Aligned>;
  using ConstTensor = Eigen::TensorMap<
      Eigen::Tensor<const T, NDIMS, Eigen::RowMajor, IndexType>, Eigen::Aligned>;

  // Unaligned Rank-<NDIMS> tensor of scalar type T.
  using UnalignedTensor = Eigen::TensorMap<Eigen::Tensor<T, NDIMS, Eigen::RowMajor, IndexType> >;
  using UnalignedConstTensor = Eigen::TensorMap<
      Eigen::Tensor<const T, NDIMS, Eigen::RowMajor, IndexType> >;

  using Tensor32Bit = Eigen::TensorMap<Eigen::Tensor<T, NDIMS, Eigen::RowMajor, int>,
                           Eigen::Aligned>;

  // Scalar tensor (implemented as a rank-0 tensor) of scalar type T.
  using Scalar = Eigen::TensorMap<
      Eigen::TensorFixedSize<T, Eigen::Sizes<>, Eigen::RowMajor, IndexType>,
      Eigen::Aligned>;
  using ConstScalar = Eigen::TensorMap<Eigen::TensorFixedSize<const T, Eigen::Sizes<>,
                                                  Eigen::RowMajor, IndexType>,
                           Eigen::Aligned>;

  // Unaligned Scalar tensor of scalar type T.
  using UnalignedScalar = Eigen::TensorMap<
      Eigen::TensorFixedSize<T, Eigen::Sizes<>, Eigen::RowMajor, IndexType> >;
  using UnalignedConstScalar = Eigen::TensorMap<Eigen::TensorFixedSize<const T, Eigen::Sizes<>,
                                                  Eigen::RowMajor, IndexType> >;

  // Rank-1 tensor (vector) of scalar type T.
  using Flat = Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor, IndexType>,
                           Eigen::Aligned>;
  using ConstFlat = Eigen::TensorMap<
      Eigen::Tensor<const T, 1, Eigen::RowMajor, IndexType>, Eigen::Aligned>;
  using Vec = Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor, IndexType>,
                           Eigen::Aligned>;
  using ConstVec = Eigen::TensorMap<
      Eigen::Tensor<const T, 1, Eigen::RowMajor, IndexType>, Eigen::Aligned>;

  // Unaligned Rank-1 tensor (vector) of scalar type T.
  using UnalignedFlat = Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor, IndexType> >;
  using UnalignedConstFlat = Eigen::TensorMap<
      Eigen::Tensor<const T, 1, Eigen::RowMajor, IndexType> >;
  using UnalignedVec = Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor, IndexType> >;
  using UnalignedConstVec = Eigen::TensorMap<
      Eigen::Tensor<const T, 1, Eigen::RowMajor, IndexType> >;

  // Rank-2 tensor (matrix) of scalar type T.
  using Matrix = Eigen::TensorMap<Eigen::Tensor<T, 2, Eigen::RowMajor, IndexType>,
                           Eigen::Aligned>;
  using ConstMatrix = Eigen::TensorMap<
      Eigen::Tensor<const T, 2, Eigen::RowMajor, IndexType>, Eigen::Aligned>;

  // Unaligned Rank-2 tensor (matrix) of scalar type T.
  using UnalignedMatrix = Eigen::TensorMap<Eigen::Tensor<T, 2, Eigen::RowMajor, IndexType> >;
  using UnalignedConstMatrix = Eigen::TensorMap<
      Eigen::Tensor<const T, 2, Eigen::RowMajor, IndexType> >;
};
}

namespace aicpu {
class EigenTensor {
 public:
  EigenTensor() = delete;
  EigenTensor(Tensor *tensor, void *data)
      : tensor_(tensor), tensor_data_(data) {}
  ~EigenTensor() = default;

  /*
   * Get tensor
   * @return succ: tensor, error : nullptr
   */
  const Tensor *GetTensor() const;

  /*
   * Eigen vec
   * @return Eigen vec
   */
  template <typename T>
  typename TTypes<T>::Vec vec() {
    return tensor<T, NUM_VALUE1>();
  }

  /*
   * Eigen matrix
   * @return Eigen matrix
   */
  template <typename T>
  typename TTypes<T>::Matrix matrix() {
    return tensor<T, NUM_VALUE2>();
  }

  /*
   * Eigen ConstMatrix
   * @return Eigen ConstMatrix
   */
  template <typename T>
  typename TTypes<T>::ConstMatrix matrix() const {
    return tensor<T, NUM_VALUE2>();
  }

  /*
   * Eigen tensor
   * @return Eigen tensor
   */
  template <typename T, size_t NDIMS>
  typename TTypes<T, NDIMS>::Tensor tensor() {
    return typename TTypes<T, NDIMS>::Tensor(
        reinterpret_cast<T *>(tensor_data_), AsEigenDSizes<NDIMS>());
  }

  /*
   * Eigen ConstTensor
   * @return Eigen ConstTensor
   */
  template <typename T, size_t NDIMS>
  typename TTypes<T, NDIMS>::ConstTensor tensor() const {
    return typename TTypes<T, NDIMS>::ConstTensor(
        reinterpret_cast<const T *>(tensor_data_), AsEigenDSizes<NDIMS>());
  }

  /*
   * Eigen Flat
   * @return Eigen Flat
   */
  template <typename T>
  typename TTypes<T>::Flat flat() {
    return typename TTypes<T>::Flat(reinterpret_cast<T *>(tensor_data_),
                                    {tensor_->GetTensorShape()->NumElements()});
  }

  /*
   * which case we pad the rest of the sizes with 1.
   * @return Eigen::DSizes: pad the rest of the sizes with 1
   */
  template <int NDIMS, typename IndexType>
  Eigen::DSizes<IndexType, NDIMS> AsEigenDSizesWithPadding() const {
    Eigen::DSizes<IndexType, NDIMS> dsizes;
    for (int d = 0; d < tensor_->GetTensorShape()->GetDims(); d++) {
      dsizes[d] =
          static_cast<IndexType>(tensor_->GetTensorShape()->GetDimSize(d));
    }
    for (int d = tensor_->GetTensorShape()->GetDims(); d < NDIMS; d++) {
      dsizes[d] = 1;
    }
    return dsizes;
  }

  /*
   * Fill `*dsizes` from `*this`
   * @return Eigen::DSizes: pad the rest of the sizes with 1
   */
  template <int NDIMS, typename IndexType = Eigen::DenseIndex>
  Eigen::DSizes<IndexType, NDIMS> AsEigenDSizes() const {
    return AsEigenDSizesWithPadding<NDIMS, IndexType>();
  }

 private:
  Tensor *tensor_;
  void *tensor_data_;
};
}  // namespace aicpu

#endif  // AICPU_EIGENTENSOR_H
