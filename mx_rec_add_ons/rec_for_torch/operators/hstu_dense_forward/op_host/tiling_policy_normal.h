#ifndef TILING_POLICY_NORMAL_H
#define TILING_POLICY_NORMAL_H

#include "tiling_policy.h"

namespace HstuDenseForward {

class TilingPolicyNormal : public TilingPolicy {
private:
    bool TilingShape(gert::TilingContext* context, optiling::HstuDenseForwardTilingData &tiling) override;

    bool TilingKeySet(gert::TilingContext* context, optiling::HstuDenseForwardTilingData &tiling) override;
};

}

#endif