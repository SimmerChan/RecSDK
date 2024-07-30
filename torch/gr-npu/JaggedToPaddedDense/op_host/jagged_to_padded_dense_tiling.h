/**
 * @file jagged_to_padded_dense_tiling.h
 *
 * Copyright (C) 2024. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */

#ifndef INDEX_SELECT_RANK1_BACKWARD_TILING
#define INDEX_SELECT_RANK1_BACKWARD_TILING

#include "register/tilingdata_base.h"
namespace optiling {
BEGIN_TILING_DATA_DEF(JaggedToPaddedDenseTilingData)
TILING_DATA_FIELD_DEF(int64_t, totalBatch);
TILING_DATA_FIELD_DEF(int64_t, baseBatchLen);
TILING_DATA_FIELD_DEF(int64_t, tailSplitIndex);
TILING_DATA_FIELD_DEF(int64_t, valuesDim0);
TILING_DATA_FIELD_DEF(int64_t, valuesDim1);
TILING_DATA_FIELD_DEF(int64_t, offsetDim0);
TILING_DATA_FIELD_DEF(int64_t, outDim1);
TILING_DATA_FIELD_DEF(int64_t, ubCanUsed);
TILING_DATA_FIELD_DEF(int64_t, bytesOfDataType);
TILING_DATA_FIELD_DEF(int64_t, offsetDataType);
TILING_DATA_FIELD_DEF(float, paddingValue);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(JaggedToPaddedDense, JaggedToPaddedDenseTilingData)
}  // namespace optiling
#endif