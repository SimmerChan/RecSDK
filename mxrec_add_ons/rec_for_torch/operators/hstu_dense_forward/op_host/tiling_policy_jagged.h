#ifndef TILING_POLICY_JAGGED_H
#define TILING_POLICY_JAGGED_H

#include "tiling_policy.h"

namespace HstuDenseForward {
    
class TilingPolicyJagged : public TilingPolicy {
public:
    void DumpTiling(optiling::HstuDenseForwardTilingData &tiling) override;

private:
    bool TilingShape(gert::TilingContext* context, optiling::HstuDenseForwardTilingData &tiling) override;

    bool TilingCore(gert::TilingContext *context, optiling::HstuDenseForwardTilingData &tiling) override;

    bool TilingKeySet(gert::TilingContext *context, optiling::HstuDenseForwardTilingData &tiling) override;
};

}

#endif