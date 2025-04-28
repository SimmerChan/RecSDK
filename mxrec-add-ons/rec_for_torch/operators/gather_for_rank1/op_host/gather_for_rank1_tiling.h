#ifndef GATHER_FOR_RANK1_TILING_H
#define GATHER_FOR_RANK1_TILING_H

#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(GatherForRank1TilingData)
TILING_DATA_FIELD_DEF(int64_t, xDim0);
TILING_DATA_FIELD_DEF(int64_t, indexDim0);
TILING_DATA_FIELD_DEF(int64_t, ubCanUsed);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(GatherForRank1, GatherForRank1TilingData)
}  // namespace optiling
#endif