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
// Copyright 2021 The BladeDISC Authors. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "tf_bridge/kernels/npu_xla_launch.h"
#include <memory>

#include "adaptor/acl_adaptor.h"
#include "common_hdrs/types.h"
#include "tf_bridge/executable/executable.h"
#include "tf_bridge/kernels/compilation_cache.h"
#include "tf_bridge/tf/errors.h"
#include "tf_mlir/compiler_input.pb.h"

namespace tensorflow {
namespace {
// OP_REQUIRES_OK_RETURN is the same as OP_REQUIRES_OK except that
// in error case, it returns RET instead of void.
#define OP_REQUIRES_OK_RETURN(CTX, RET, ...)                      \
    do {                                                          \
        ::tensorflow::Status _s(__VA_ARGS__);                     \
        if (!TF_PREDICT_TRUE(_s.ok())) {                          \
            (CTX)->CtxFailureWithWarning(__FILE__, __LINE__, _s); \
            return RET;                                           \
        }                                                         \
    } while (0)

// Helper static functions to construct parameters for
// XlaLocalLaunchBase constructor from OpKernelConstruction.
static std::vector<int> ConstantsVector(OpKernelConstruction* ctx)
{
    DataTypeVector constant_types;
    OP_REQUIRES_OK_RETURN(ctx, std::vector<int>(), ctx->GetAttr("Tconstants", &constant_types));
    std::vector<int> constants(constant_types.size());
    std::iota(constants.begin(), constants.end(), 0);
    return constants;
}

static std::vector<int> FixedShapesVector(OpKernelConstruction* ctx)
{
    std::vector<int> fixed_shaped;
    if (ctx->HasAttr("Tfixedshapes")) {
        DataTypeVector constant_types;
        OP_REQUIRES_OK_RETURN(ctx, std::vector<int>(), ctx->GetAttr("Tconstants", &constant_types));

        DataTypeVector fixed_shaped_types;
        OP_REQUIRES_OK_RETURN(ctx, std::vector<int>(), ctx->GetAttr("Tfixedshapes", &fixed_shaped_types));

        fixed_shaped.resize(fixed_shaped_types.size());
        std::iota(fixed_shaped.begin(), fixed_shaped.end(), constant_types.size());
    }
    return fixed_shaped;
}

static std::vector<int> HostArgsVector(OpKernelConstruction* ctx)
{
    std::vector<int> host_args;
    if (ctx->HasAttr("Thostargs") && ctx->HasAttr("Tfixedshapes")) {
        DataTypeVector constant_types;
        OP_REQUIRES_OK_RETURN(ctx, std::vector<int>(), ctx->GetAttr("Tconstants", &constant_types));

        DataTypeVector fixed_shaped_types;
        OP_REQUIRES_OK_RETURN(ctx, std::vector<int>(), ctx->GetAttr("Tfixedshapes", &fixed_shaped_types));

        DataTypeVector host_arg_types;
        OP_REQUIRES_OK_RETURN(ctx, std::vector<int>(), ctx->GetAttr("Thostargs", &host_arg_types));

        host_args.resize(host_arg_types.size());
        std::iota(host_args.begin(), host_args.end(), constant_types.size() + fixed_shaped_types.size());
    }
    return host_args;
}

static std::vector<int> ResourcesVector(OpKernelConstruction* ctx)
{
    DataTypeVector constant_types;
    OP_REQUIRES_OK_RETURN(ctx, std::vector<int>(), ctx->GetAttr("Tconstants", &constant_types));

    DataTypeVector fixed_shaped_types;
    if (ctx->HasAttr("Tfixedshapes")) {
        OP_REQUIRES_OK_RETURN(ctx, std::vector<int>(), ctx->GetAttr("Tfixedshapes", &fixed_shaped_types));
    }

    DataTypeVector host_arg_types;
    if (ctx->HasAttr("Thostargs")) {
        OP_REQUIRES_OK_RETURN(ctx, std::vector<int>(), ctx->GetAttr("Thostargs", &host_arg_types));
    }

    DataTypeVector arg_types;
    OP_REQUIRES_OK_RETURN(ctx, std::vector<int>(), ctx->GetAttr("Targs", &arg_types));

    int numResources = -1;
    OP_REQUIRES_OK_RETURN(ctx, std::vector<int>(), ctx->GetAttr("Nresources", &numResources));

    std::vector<int> resources(numResources);
    std::iota(resources.begin(), resources.end(),
              constant_types.size() + fixed_shaped_types.size() + host_arg_types.size() + arg_types.size());
    return resources;
}

static NameAttrList FunctionAttr(OpKernelConstruction* ctx, const char* const attr)
{
    const NameAttrList* func = nullptr;
    OP_REQUIRES_OK_RETURN(ctx, NameAttrList(), ctx->GetAttr(attr, &func));
    return *func;
}

#undef OP_REQUIRES_OK_RETURN

}  // namespace

namespace npu_xla {
namespace {
Status PrepareOptions(OpKernelContext* ctx, std::unique_ptr<CompilerInput>& inputPtr, const std::string& deviceType)
{
    auto& options = *(inputPtr->mutable_options());
    auto flib_def = ctx->function_library();
    if (deviceType == DEVICE_NPU) {
        *options.mutable_device_type() = "MLIR_NPU";
    } else {
        return errors::Internal("NPU_XLA unsupported device type: ", deviceType);
    }
    // get ordinal
    options.set_device_ordinal(0);
    options.set_graph_def_version(flib_def->graph_def_version());
    options.set_allow_cpu_custom_calls(false);
    options.set_use_tuple_arg(false);
    options.set_return_updated_values_for_all_resources(false);
    options.set_resolve_compile_time_constants(true);
    options.set_always_return_tuple(false);
    options.set_is_entry_computation(true);

    for (int i = 0; i < ctx->num_outputs(); ++i) {
        if (ctx->output_memory_type(i) == DEVICE_MEMORY && ctx->op_device_context()) {
            options.add_output_placements("npu");
        } else {
            options.add_output_placements("cpu");
        }
    }
    
    return absl::OkStatus();
}
}  // namespace

NpuXlaLaunchOp::NpuXlaLaunchOp(OpKernelConstruction* ctx)
    : OpKernel(ctx),
      constants_(ConstantsVector(ctx)),
      fixed_shapes_(FixedShapesVector(ctx)),
      host_args_(HostArgsVector(ctx)),
      resources_(ResourcesVector(ctx)),
      func_(FunctionAttr(ctx, "mlir_function")),
      device_type_(ctx->device_type().type_string())
{
}

void NpuXlaLaunchOp::Compute(OpKernelContext* ctx)
{
    VLOG(VLOG_LEVEL_2) << "NpuXlaLaunchOp::Compute: " << name();
    OP_REQUIRES_OK(ctx, CompileAndRunMlir(ctx));
}

Status NpuXlaLaunchOp::CompileAndRunMlir(OpKernelContext* ctx)
{
    if (VLOG_IS_ON(1)) {
        VLOG(0) << "Constant (" << constants_.size() << "):";
        for (auto&& v : constants_)
            VLOG(1) << "\tconst_idx: " << v;
        VLOG(0) << "Fix shape (" << fixed_shapes_.size() << "):";
        for (auto&& v : fixed_shapes_)
            VLOG(1) << "\tfix_shape_idx: " << v;
        VLOG(0) << "Resource (" << resources_.size() << "):";
        for (auto&& v : resources_)
            VLOG(1) << "\tresource_idx: " << v;
    }

    // We store information about the JIT-compiled Result
    // in the ResourceMgr.
    ResourceMgr* rm = ctx->resource_manager();
    if (!rm) {
        return errors::Internal("No resource manager.");
    }

    CompilationCache* cache = nullptr;
    TF_RETURN_IF_ERROR(rm->LookupOrCreate<CompilationCache>(rm->default_container(), "npu_xla_cache", &cache,
                                                            [&](CompilationCache** cache) {
                                                                *cache = new CompilationCache();
                                                                return absl::OkStatus();
                                                            }));
    // Hold the reference to the JIT during evaluation. (We could probably
    // free it sooner because the ResourceMgr will retain a reference, but
    // this is more obviously correct.)
    core::ScopedUnref cache_ref(cache);

    std::map<int, OptionalTensor> variables;
    if (!resources_.empty()) {
        return errors::Internal("NPU_XLA does not support resource variables for now.");
    }

    auto input_ptr = std::make_unique<CompilerInput>();
    TF_RETURN_IF_ERROR(PrepareOptions(ctx, input_ptr, device_type_));

    std::map<int, Tensor> constant_args;
    for (int i : constants_) {
        constant_args.insert({i, ctx->input(i)});
    }

    std::set<int> fixed_shape_args;
    std::set<int> host_args_set;
    fixed_shape_args.insert(fixed_shapes_.begin(), fixed_shapes_.end());
    host_args_set.insert(host_args_.begin(), host_args_.end());

    // not hold ownership of executable
    Executable* executable = nullptr;
    TF_RETURN_IF_ERROR(cache->Compile(std::move(input_ptr), func_, constant_args, fixed_shape_args, host_args_set,
                                      variables, ctx, &executable));
    TF_RET_CHECK(executable != nullptr);
    TF_RETURN_IF_ERROR(executable->Run(ctx));

    return absl::OkStatus();
}

}  // namespace npu_xla

REGISTER_NPU_XLA_LAUNCH_KERNEL(npu_xla::DEVICE_NPU);

}  // namespace tensorflow