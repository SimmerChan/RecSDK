/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
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

/*!
 * \file SparseFwFFMPart2.cc
 * \brief
 */

#include "register/op_impl_registry.h"
#include "tiling_rt2_util.h"
#include "op_util.h"
#include "op_const.h"

using namespace ge;

namespace optiling {

struct SparseFwFFMPart2CompileInfo {
  int64_t core_num;
};
struct  SparseFwFFMPart2TilingData {
  // use aicore num
  int64_t act_core_num = 0;
  // each aicore need compute num except last aicore
  int64_t batch_each_core = 0;
  // last aicore need compute num
  int64_t batch_last_core = 0;
};

void InItRunningParams4SparseFwFFMPart2(SparseFwFFMPart2TilingData* tilingdata) {
  tilingdata->act_core_num = 0;
  tilingdata->batch_each_core = 0;
  tilingdata->batch_last_core = 0;
}

static void CalTilingParam4SparseFwFFMPart2(SparseFwFFMPart2TilingData* tilingdata, int64_t batch_size, int64_t aicore_num) {
  tilingdata->batch_each_core = (batch_size + aicore_num - 1) / aicore_num;
  tilingdata->act_core_num = (batch_size + tilingdata->batch_each_core - 1) / tilingdata->batch_each_core;
  tilingdata->batch_last_core = batch_size - (tilingdata->act_core_num - 1) * tilingdata->batch_each_core;
}

ge::graphStatus Tiling4SparseFwFFMPart2(gert::TilingContext* context) {
  auto cross_mean_sum_storage_shape = context->GetInputShape(1);
  OPS_CHECK_NULL_WITH_CONTEXT(context, cross_mean_sum_storage_shape);
  auto& cross_mean_sum_shape = cross_mean_sum_storage_shape->GetStorageShape();
  int64_t batch_size = cross_mean_sum_shape.GetDim(0);

  auto compile_info =
      reinterpret_cast<const SparseFwFFMPart2CompileInfo*>(context->GetCompileInfo());
  OPS_CHECK_NULL_WITH_CONTEXT(context, compile_info);

  int64_t core_num = compile_info->core_num;
  if (core_num == 0) {
    VECTOR_INNER_ERR_REPORT_TILIING("SparseFwFFMPart2", "core_num = 0 is not support");
    return ge::GRAPH_FAILED;
  }
  auto tilingdata = context->GetTilingData<SparseFwFFMPart2TilingData>();
  OPS_CHECK_NULL_WITH_CONTEXT(context, tilingdata);

  InItRunningParams4SparseFwFFMPart2(tilingdata);
  CalTilingParam4SparseFwFFMPart2(tilingdata, batch_size, core_num);
  OP_LOGD("SparseFwFFMPart2Tiling", tilingdata->ToString().c_str());

  context->SetBlockDim(core_num);
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepare4SparseFwFFMPart2(gert::TilingParseContext* context) {
  auto compile_info = GetCompileInfoPtr<SparseFwFFMPart2CompileInfo>(context);
  OPS_CHECK_NULL_WITH_CONTEXT(context, compile_info);
  std::unique_ptr<nlohmann::json> parsed_object_cinfo = GetCompileInfoJson(context);
  OPS_CHECK_NULL_WITH_CONTEXT(context, parsed_object_cinfo);
  const nlohmann::json& vars = (*parsed_object_cinfo)["vars"];
  OP_TILING_CHECK(vars.empty(), VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "get vars failed."),
                  return ge::GRAPH_FAILED);
  // get tiling info
  OP_TILING_CHECK(!ReadCompileItem(vars, "core_num", compile_info->core_num),
                  VECTOR_INNER_ERR_REPORT_TILIING("SparseFwFFMPart2", "TilingPrepare, get core_num error"),
                  return ge::GRAPH_FAILED);
  return ge::GRAPH_SUCCESS;
}

// register tiling interface of the SparseFwFFMPart2 op.
IMPL_OP(SparseFwFFMPart2).Tiling(Tiling4SparseFwFFMPart2).TilingParse<SparseFwFFMPart2CompileInfo>(TilingPrepare4SparseFwFFMPart2);
}  // namespace optiling
