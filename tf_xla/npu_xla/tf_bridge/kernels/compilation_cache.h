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
// limitations under the License

#ifndef NPU_XLA_TF_BRIDGE_KERNELS_COMPILATION_CACHE_H_
#define NPU_XLA_TF_BRIDGE_KERNELS_COMPILATION_CACHE_H_

#include "tensorflow/core/framework/resource_mgr.h"
#include "tf_bridge/common.h"
#include "tf_bridge/executable/executable.h"
#include "tf_mlir/compiler_input.pb.h"

namespace tensorflow {
namespace npu_xla {
class CompilationCache : public ResourceBase {
public:
    explicit CompilationCache();

    std::string DebugString() const override;

    Status Compile(std::unique_ptr<CompilerInput> input, const NameAttrList& function,
                   const std::map<int, Tensor>& constant_args, const std::set<int>& fixed_shape_args,
                   const std::set<int>& host_args, const std::map<int, OptionalTensor>& variable_args,
                   OpKernelContext* ctx, Executable** executable);

    struct Signature {
        std::string name;

        // List of rank/type pairs, only used with mlir dynamic shape compiler
        std::vector<std::pair<DataType, int>> arg_ranks;

        std::vector<std::pair<DataType, TensorShape>> arg_types;

        // ordinal args that are expected placed on cpu.
        // For cpu only execution, this will be empty.
        std::vector<int> host_args;

        // List of Tensor values for compile-time constant arguments to the
        // compilation, ordered by argument number. Tensors must be in host
        // memory.
        std::vector<Tensor> arg_values;

        bool operator==(const Signature& other) const;

        struct Hash {
            uint64 operator()(const Signature& signature) const;
        };
    };
    static std::string SignatureDebugString(const Signature& sig);

private:
    Status CompileImpl(std::unique_ptr<CompilerInput> input, const NameAttrList& function,
                       const std::map<int, Tensor>& constant_args, const std::set<int>& fixed_shape_args,
                       const std::set<int>& host_args, const std::map<int, OptionalTensor>& variable_args,
                       OpKernelContext* ctx, Executable** executable);

    // Builds the signature for a compilation.
    static Status BuildSignature(const NameAttrList& function, const std::map<int, Tensor>& constant_args,
                                 const std::set<int>& fixed_shape_args, const std::set<int>& host_args,
                                 const std::map<int, OptionalTensor>& variable_args, OpKernelContext* ctx,
                                 Signature* signature, bool is_dynamic = false);

    struct Entry {
        mutex mu;

        // Have we tried compiling this entry?
        bool compiled = false;

        // Did compilation succeed?
        Status compilation_status TF_GUARDED_BY(mu);

        // The XLA executable compiled from <computation>. May be null if no
        // executable has been built.
        std::unique_ptr<Executable> executable TF_GUARDED_BY(mu);

        bool compilation_slot_initialized = false;
        int compilation_slot = -1;
    };

    Entry* CreateOrGetCacheEntry(const Signature& signature, bool* new_entry);

    Status CheckAndLoadFromCache(const Signature& signature, Entry* entry, CompilationResultProto* result_proto,
                                 bool* hit_cache);

    Status CompileAndCacheResult(const NameAttrList& function, const std::map<int, Tensor>& constant_args,
                                 const std::set<int>& fixed_shape_args, const std::set<int>& host_args,
                                 const std::map<int, OptionalTensor>& variable_args, OpKernelContext* ctx,
                                 CompilerInput* input, Entry* entry, CompilationResultProto* result_proto);

    Status UpdateExecutable(const Signature& signature, Entry* entry, CompilationResultProto&& result_proto,
                            bool hit_cache);

    mutex compile_cache_mu_;
    std::unordered_map<Signature, std::unique_ptr<Entry>, Signature::Hash> cache_ TF_GUARDED_BY(compile_cache_mu_);

    std::string tf_mlir_bin_path_;
    std::string cache_path_;
};

Status PrepareCompilerInput(const NameAttrList& function, const std::map<int, Tensor>& constant_args,
                            const std::set<int>& fixed_shape_args, const std::set<int>& host_args,
                            const std::map<int, OptionalTensor>& variable_args, OpKernelContext* ctx,
                            CompilerInput* compiler_input, bool is_mlir = true);

Status CompileFunction(const std::string& func_name, const std::string& tf_mlir_bin_path,
                       const std::string& compile_dir, CompilerInput& input, bool remove_after_compile);
}  // namespace npu_xla
}  // namespace tensorflow

#endif  // NPU_XLA_TF_BRIDGE_KERNELS_COMPILATION_CACHE_H_