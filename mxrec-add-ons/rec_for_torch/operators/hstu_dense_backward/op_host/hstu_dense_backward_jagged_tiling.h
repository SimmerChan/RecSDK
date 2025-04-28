#ifndef HSTU_DENSE_BACKWARD_JAGGED_TILING_H
#define HSTU_DENSE_BACKWARD_JAGGED_TILING_H

#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

#include "hstu_dense_backward_tiling.h"
#include "hstu_dense_backward_normal_tiling.h"

namespace optiling {
ge::graphStatus TilingJaggedFunc(gert::TilingContext *context,
                                 const gert::RuntimeAttrs *attrs,
                                 HstuDenseBackwardTilingData &tiling);

ge::graphStatus JaggedInferShape(gert::InferShapeContext *context);
}

#endif