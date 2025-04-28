#ifndef TILING_POLICY_NORMAL_V200_H
#define TILING_POLICY_NORMAL_V200_H

#include "tiling_policy_normal.h"

namespace HstuDenseForward {

class TilingPolicyNormalv200 : public TilingPolicyNormal {
public:
    bool GeneralShapeCheck(int64_t batchSize, int64_t seqLen, int64_t headNum, int64_t dim) override;
private:
    bool TilingHeighLevelApi(gert::TilingContext* context, optiling::HstuDenseForwardTilingData &tiling) override;
    bool TilingKeySet(gert::TilingContext* context, optiling::HstuDenseForwardTilingData &tiling) override;
};
}

#endif