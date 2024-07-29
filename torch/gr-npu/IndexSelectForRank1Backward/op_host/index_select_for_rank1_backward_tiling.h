
#include "register/tilingdata_base.h"
namespace optiling {
BEGIN_TILING_DATA_DEF(IndexSelectForRank1BackwardTilingData)
  TILING_DATA_FIELD_DEF(int64_t, totalLen);
  TILING_DATA_FIELD_DEF(int64_t, xDim0);
  TILING_DATA_FIELD_DEF(int64_t, baseLen);
  TILING_DATA_FIELD_DEF(int64_t, tailSplitIndex);
  TILING_DATA_FIELD_DEF(int64_t, keyDim0Align64B);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(IndexSelectForRank1Backward, IndexSelectForRank1BackwardTilingData)
}
