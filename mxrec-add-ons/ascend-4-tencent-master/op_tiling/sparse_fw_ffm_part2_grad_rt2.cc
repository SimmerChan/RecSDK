/**
 * Copyright (c) 2022-2022 Huawei Technologies Co., Ltd.  All rights reserved.
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
 * \file sp.cc
 * \brief tiling function of op
 */
#include "register/op_compile_info_base.h"
#include "tiling_rt2_util.h"
#include "tiling_util.h"

namespace optiling {
using namespace ge;
using namespace std;

static constexpr size_t INPUT_CROSS_IDX = 1;
static constexpr size_t INPUT_FW_IDX = 3;

struct TilingPrepare4SparseFwFFMPart2GradCompileInfo {
  int32_t block_dim;
};

struct SparseFwFFMPart2GradTilingParams {
  int64_t tiling_key;
  int64_t batch_size;
  int64_t field_num;
  int64_t fw_field_num;
  int64_t fw_weight_len;
};

void InitSparseFwFFMPart2GradParams(SparseFwFFMPart2GradTilingParams* params) {
  params->tiling_key = 0;
  params->batch_size = 1;
  params->field_num = 1;
  params->fw_field_num = 1;
  params->fw_weight_len = 1;
}

ge::graphStatus Tiling4SparseFwFFMPart2Grad(gert::TilingContext* context) {
  OP_LOGD(context->GetNodeName(), "Tiling4SparseFwFFMPart2Grad running.");

  const TilingPrepare4SparseFwFFMPart2GradCompileInfo* compile_info =
      reinterpret_cast<const TilingPrepare4SparseFwFFMPart2GradCompileInfo*>(context->GetCompileInfo());
  OPS_CHECK_NULL_WITH_CONTEXT(context, compile_info);

  SparseFwFFMPart2GradTilingParams* tilingdata = context->GetTilingData<SparseFwFFMPart2GradTilingParams>();
  OPS_CHECK_NULL_WITH_CONTEXT(context, tilingdata);

  // get input shape info
  auto input_cross = context->GetInputShape(INPUT_CROSS_IDX);
  OP_TILING_CHECK(input_cross == nullptr,
                 VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "get input_cross failed."),
                 return ge::GRAPH_FAILED);
  const gert::Shape& input_cross_shape = input_cross->GetStorageShape();

  auto input_fw = context->GetInputShape(INPUT_FW_IDX);
  OP_TILING_CHECK(input_fw == nullptr,
                 VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "get input_fw failed."),
                 return ge::GRAPH_FAILED);
  const gert::Shape& input_fw_shape = input_fw->GetStorageShape();

  int64_t core_num = compile_info->block_dim;
  InitSparseFwFFMPart2GradParams(tilingdata);

  tilingdata->tiling_key = 0;
  tilingdata->batch_size = input_cross_shape.GetDim(0);
  tilingdata->field_num = input_cross_shape.GetDim(1);
  tilingdata->fw_field_num = input_cross_shape.GetDim(2);
  tilingdata->fw_weight_len = input_fw_shape.GetDim(0);
  // block_dim, core num used in tik op
  context->SetBlockDim(core_num);
  OP_LOGI(context->GetNodeName(), "Tiling4SparseFwFFMPart2Grad run success.");
  OP_LOGD(context->GetNodeName(), "Tiling4SparseFwFFMPart2Grad tiling_data:%s", GetTilingDataString<int64_t>(context).c_str());
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepare4SparseFwFFMPart2Grad(gert::TilingParseContext* context) {
  OP_LOGD(context->GetNodeName(), "begin to do TilingPrepare4SparseFwFFMPart2Grad.");
  auto compile_info = GetCompileInfoPtr<TilingPrepare4SparseFwFFMPart2GradCompileInfo>(context);
  OPS_CHECK_NULL_WITH_CONTEXT(context, compile_info);

  std::unique_ptr<nlohmann::json> parsed_object_cinfo = GetCompileInfoJson(context);
  OPS_CHECK_NULL_WITH_CONTEXT(context, parsed_object_cinfo);

  const nlohmann::json& vars = (*parsed_object_cinfo)["vars"];
  OP_TILING_CHECK(vars.empty(), VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "get vars failed."),
                 return ge::GRAPH_FAILED);
  if (vars.empty()) {
    return ge::GRAPH_FAILED;
  }
  OP_TILING_CHECK(!ReadCompileItem(vars, "core_num", compile_info->block_dim),
                 VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "get core_num from compile info faided."),
                 return ge::GRAPH_FAILED);
  if (!ReadCompileItem(vars, "core_num", compile_info->block_dim)) {
    return ge::GRAPH_FAILED;
  }

  return ge::GRAPH_SUCCESS;
}

IMPL_OP(SparseFwFFMPart2Grad)
    .Tiling(Tiling4SparseFwFFMPart2Grad)
    .TilingParse<TilingPrepare4SparseFwFFMPart2GradCompileInfo>(TilingPrepare4SparseFwFFMPart2Grad);
}  // namespace optiling
