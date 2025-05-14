/* Copyright 2025. Huawei Technologies Co.,Ltd. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
        limitations under the License.
==============================================================================*/

#include "tf_bridge/executable/executable.h"

#include "adaptor/acl_adaptor.h"
#include "runtime_pipeline.h"
#include "tensorflow/core/platform/env.h"

namespace tensorflow {
namespace npu_xla {

Executable::Executable(CompilationResult&& compilation_result)
{
    compilation_result_ = std::make_unique<CompilationResult>(std::move(compilation_result));
    compiled_model_path_ = compilation_result_->compiled_model_path();
}

Executable::~Executable() {}

Status Executable::Run(OpKernelContext* ctx)
{
    // prepare input for runtime
    std::vector<std::vector<int64_t>> shapeInfo;
    std::vector<void*> inputPtrForRuntime;
    for (int i = 0; i < ctx->num_inputs(); ++i) {
        const Tensor& inp = ctx->input(i);

        inputPtrForRuntime.push_back(inp.data());

        const TensorShape& shape = inp.shape();
        std::vector<int64_t> tmp;
        for (int j = 0; j < shape.dims(); ++j) {
            tmp.push_back(shape.dim_size(j));
        }

        shapeInfo.push_back(tmp);
        VLOG(1) << "input i: " << i << ", shape: " << inp.shape().DebugString()
                << ", AllocatedBytes: " << inp.AllocatedBytes() << ", data ptr: " << inp.data();
    }
    // start runtime
    auto res =
        DOPAI::GRT::runtime_pipeline(compiled_model_path_.c_str(), inputPtrForRuntime, shapeInfo, ctx->num_outputs());

    // copy result
    for (int i = 0; i < ctx->num_outputs(); ++i) {
        auto& output = res[i];
        void* dataPtr = std::get<0>(output);
        uint64_t dataSize = std::get<1>(output);
        const std::vector<int64_t> shape = std::get<2>(output);

        absl::Span<const int64_t> abslShape(shape.data(), shape.size());
        TensorShape outputShape;
        TF_RETURN_IF_ERROR(TensorShape::BuildTensorShape(abslShape, &outputShape));

        Tensor* out = nullptr;
        TF_RETURN_IF_ERROR(ctx->allocate_output(i, outputShape, &out));

        AclAdaptor& aclIns = AclAdaptor::GetInstance(0);
        if (!aclIns.MemcpyHToD(out->data(), dataSize, dataPtr, dataSize)) {
            return absl::InternalError("MemcpyHToD failed");
        }
        VLOG(1) << "output result idx: " << i << ", copy result to device ptr: " << out->data();
    }
    return absl::OkStatus();
}

Status Executable::DumpToFile(const std::string& filename) const
{
    TF_RETURN_IF_ERROR(WriteTextProto(Env::Default(), filename, *compilation_result_.get()));
}

}  // namespace npu_xla
}  // namespace tensorflow