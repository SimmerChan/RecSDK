#include "register/tilingdata_base.h"

namespace optiling {
    BEGIN_TILING_DATA_DEF(LcclAllToAllTilingData)
    TILING_DATA_FIELD_DEF(int64_t, rank);
    TILING_DATA_FIELD_DEF(int64_t, rankSize);
    TILING_DATA_FIELD_DEF(int64_t, magic);
    END_TILING_DATA_DEF;

    REGISTER_TILING_DATA_CLASS(LcclAllToAll, LcclAllToAllTilingData)
}

