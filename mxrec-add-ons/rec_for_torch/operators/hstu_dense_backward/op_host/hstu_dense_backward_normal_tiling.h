#ifndef HSTU_DENSE_BACKWARD_NORMAL_TILING_H
#define HSTU_DENSE_BACKWARD_NORMAL_TILING_H

#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

#include "hstu_dense_backward_tiling.h"

namespace optiling {
ge::graphStatus TilingNormalFunc(gert::TilingContext *context,
                                 const gert::RuntimeAttrs *attrs,
                                 HstuDenseBackwardTilingData &tiling);

ge::graphStatus CheckMaskTypeAndBias(gert::TilingContext *context, HstuDenseBackwardTilingData &tiling);

ge::graphStatus NormalInferShape(gert::InferShapeContext *context);
}

#endif