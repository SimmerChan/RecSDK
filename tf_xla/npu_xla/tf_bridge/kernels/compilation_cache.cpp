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

#include "tf_bridge/kernels/compilation_cache.h"

#include <chrono>

#include "compiler_pipeline.h"
#include "tensorflow/core/lib/gtl/cleanup.h"
#include "tf_bridge/common.h"
#include "tf_bridge/compilation_result.pb.h"
#include "tf_bridge/tf/errors.h"
#include "tf_bridge/tf/log.h"
#include "tf_bridge/tf/subprocess.h"
#include "tf_bridge/utils/common_utils.h"

namespace tensorflow {
namespace npu_xla {
using Signature = CompilationCache::Signature;
using std::string;

namespace {
// BladeDisc
// A class used to load/store compliatoin cache on disk.
// Note that:
// 1, we suppose the clustering result is stable across different process.
// The above assumption may not always hold. Thus be careful when using this
// feature. 2, We may have multiple intances of CompilationCache if we have
// multiple sessions in the same process. However there should be only one
// instance of PersistentCompliationCache in order to get stable result.
class PersistentCompliationCache {
public:
    // cache_dump_path: the path used to store the cache files.
    explicit PersistentCompliationCache(const string& cache_dump_path) : cache_dump_path_(cache_dump_path)
    {
        auto status = LoadFromFile();
        TF_CHECK_OK(status);
    }

    ~PersistentCompliationCache()
    {
        auto status = DumpToFile();
        if (!status.ok()) {
            VLOG(0) << "[WARNING] fail to dump compilation cache: " << status.message();
        }
    }

    PersistentCompliationCache(const PersistentCompliationCache&) = delete;
    void operator=(const PersistentCompliationCache&) = delete;

    // Returns the name of file used to store `CompilationCacheResult`.
    static string getCacheTableFileName()
    {
        return "npu_xla_cache";
    }

    // Returns the path of file used to store `CompilationCacheResult`.
    string getCacheTableFilePath()
    {
        return absl::StrCat(cache_dump_path_, "/", getCacheTableFileName());
    }

    // Returns the singleton
    static PersistentCompliationCache& Global()
    {
        static PersistentCompliationCache cache(GetTfBridgeOptions()->cache_path);
        return cache;
    }

    // Saves cache table to file located at `getCacheTableFilePath`.
    Status DumpToFile()
    {
        mutex_lock lock(mu_);
        return DumpToFileLocked();
    }

    // Loads cache table from file located at `getCacheTableFilePath`.
    Status LoadFromFile();

    // Returns true if found, otherwise return false.
    // When found, `out_filename` will be filled with the filename of the
    // compiled result proto corresponding to `sig`.
    bool find(const Signature& sig, std::string& out_filename)
    {
        mutex_lock lock(mu_);
        auto it = cache_.find(sig);
        if (it != cache_.end())
            out_filename = it->second;
        return it != cache_.end();
    }

    // Update the value for key `sig` to `filename`. Override the value if
    // `override` is ture. Returns true if updated.
    bool update(const Signature& sig, const std::string& filename, bool override = false, bool writeThrough = true)
    {
        mutex_lock lock(mu_);
        auto it = cache_.emplace(sig, filename);
        bool updated = it.second;
        if (override && !updated) {
            it.first->second = filename;
            updated = true;
        }
        if (updated) {
            isCacheDirty_ = true;
            if (writeThrough) {
                TF_CHECK_OK(DumpToFileLocked());
            }
        }
        return updated;
    }

    // Returns a (per-process-level) unique name used for saving compiled result
    // proto on disk.
    std::string getNextUniqueNameOfCompiledResultProto()
    {
        mutex_lock lock(mu_);
        std::string path;
        do {
            path = absl::StrCat(cache_dump_path_, "/", "cache_item_", next_idx_++);
        } while (tensorflow::Env::Default()->FileExists(path).ok());
        return path;
    }

private:
    // Saves cache table to file located at `getCacheTableFilePath`.
    Status DumpToFileLocked();

private:
    mutex mu_;
    string cache_dump_path_;
    std::unordered_map<Signature, string, Signature::Hash> cache_;
    std::atomic<uint64_t> next_idx_;
    bool isCacheDirty_ = false;
};

Status PersistentCompliationCache::DumpToFileLocked()
{
    if (!is_cache_dirty_)
        return absl::OkStatus();

    VLOG(VLOG_LEVEL_2) << "Dump compilation cache to: " << cache_dump_path_;
    CompilationCacheResult result;
    Status s = tensorflow::Env::Default()->RecursivelyCreateDir(cache_dump_path_);
    if (!s.ok() && !errors::IsAlreadyExists(s)) {
        errors::AppendToMessage(&s, "when creating directory ", cache_dump_path_);
        return s;
    }

    for (auto& inner : cache_) {
        auto& sig = inner.first;
        auto entry = result.add_entries();
        entry->set_filename(inner.second);
        auto& sig_proto = *entry->mutable_sig();
        *sig_proto.mutable_name() = sig.name;
        for (size_t i = 0; i < sig.arg_ranks.size(); ++i) {
            const auto& pair = sig.arg_ranks[i];
            TypeRankPair* pair_proto = sig_proto.add_arg_ranks();
            pair_proto->set_type(static_cast<int>(pair.first));
            pair_proto->set_rank(pair.second);
        }
        for (size_t i = 0; i < sig.arg_types.size(); ++i) {
            const auto& pair = sig.arg_types[i];
            tensorflow::TensorShapeProto shape_proto;
            pair.second.AsProto(&shape_proto);
            TypeShapePair* pair_proto = sig_proto.add_arg_types();
            pair_proto->set_type(static_cast<int>(pair.first));
            shape_proto.AppendToString(pair_proto->mutable_shape());
        }
        for (size_t i = 0; i < sig.host_args.size(); ++i) {
            sig_proto.add_host_args(sig.host_args[i]);
        }
        for (size_t i = 0; i < sig.arg_values.size(); ++i) {
            tensorflow::TensorProto tensor_proto;
            sig.arg_values[i].AsProtoTensorContent(&tensor_proto);
            tensor_proto.AppendToString(sig_proto.add_arg_values());
        }
    }

    TF_RETURN_IF_ERROR(WriteBinaryProto(tensorflow::Env::Default(), getCacheTableFilePath(), result));
    isCacheDirty_ = false;
    return absl::OkStatus();
}

Status PersistentCompliationCache::LoadFromFile()
{
    CompilationCacheResult result;
    // Early return if the cache file is not created before.
    if (!tensorflow::Env::Default()->FileExists(getCacheTableFilePath()).ok()) {
        VLOG(VLOG_LEVEL_2) << "Skip load compilation cache due to no " << getCacheTableFilePath();
        return absl::OkStatus();
    }
    if (!ReadBinaryProto(tensorflow::Env::Default(), getCacheTableFilePath(), &result).ok()) {
        return errors::NotFound("compilation cache file is not found: " + getCacheTableFilePath());
    }

    for (int i = 0; i < result.entries_size(); ++i) {
        auto& sig_proto = result.entries(i).sig();
        Signature sig;
        sig.name = sig_proto.name();
        for (int j = 0; j < sig_proto.arg_ranks_size(); ++j) {
            const TypeRankPair& pair_proto = sig_proto.arg_ranks(j);
            sig.arg_ranks.emplace_back(static_cast<tensorflow::DataType>(pair_proto.type()), pair_proto.rank());
        }
        for (int j = 0; j < sig_proto.arg_types_size(); ++j) {
            const TypeShapePair& pair_proto = sig_proto.arg_types(j);
            tensorflow::TensorShapeProto shape_proto;
            shape_proto.ParseFromString(pair_proto.shape());
            sig.arg_types.emplace_back(static_cast<tensorflow::DataType>(pair_proto.type()),
                                       tensorflow::TensorShape(shape_proto));
        }
        for (int j = 0; j < sig_proto.host_args_size(); ++j) {
            sig.host_args.push_back(sig_proto.host_args(j));
        }
        for (int j = 0; j < sig_proto.arg_values_size(); ++j) {
            tensorflow::TensorProto tensor_proto;
            tensor_proto.ParseFromString(sig_proto.arg_values(j));
            sig.arg_values.emplace_back();
            TF_RET_CHECK(sig.arg_values.back().FromProto(tensor_proto));
        }
        VLOG(VLOG_LEVEL_2) << "loading signature: " << CompilationCache::SignatureDebugString(sig);
        cache_[sig] = result.entries(i).filename();
    }
    LOG(INFO) << "Load compilation cache success from " << cache_dump_path_ << " with " << result.entries_size()
              << " entries.";
    return absl::OkStatus();
}

static Status CreateCompilationPath(std::string* dirPath)
{
    const std::string& tmpPath = GetTfBridgeOptions()->compilation_product_path;

    auto realPath = std::filesystem::absolute(tmpPath);
    *dirPath = realPath.string() + "/";
    if (!tensorflow::Env::Default()->CreateUniqueFileName(dirPath, "")) {
        return absl::AlreadyExistsError("Can't get unique dir name.");
    }

    TF_RETURN_IF_ERROR(tensorflow::Env::Default()->RecursivelyCreateDir(*dirPath));
    VLOG(1) << "Success to create directory: " << *dirPath;

    return absl::OkStatus();
}

}  // namespace

CompilationCache::CompilationCache()
{
    auto* opts = GetTfBridgeOptions();
    tf_mlir_bin_path_ = opts->tf_mlir_bin_path;
    cache_path_ = opts->cache_path;
}

std::string CompilationCache::SignatureDebugString(const Signature& sig)
{
    string result = sig.name;

    for (const auto& a : sig.arg_ranks) {
        absl::StrAppend(&result, ",", DataTypeString(a.first), ", rank: ", a.second);
    }

    for (const auto& a : sig.arg_types) {
        absl::StrAppend(&result, ", shape: ", DataTypeString(a.first), a.second.DebugString());
    }

    for (const auto& v : sig.host_args) {
        absl::StrAppend(&result, ", host_arg: ", v);
    }

    for (const auto& v : sig.arg_values) {
        absl::StrAppend(&result, ", value: ", v.DebugString());
    }

    return result;
}

bool CompilationCache::Signature::operator==(const Signature& other) const
{
    if (name != other.name)
        return false;
    if (arg_ranks != other.arg_ranks)
        return false;
    if (arg_types != other.arg_types)
        return false;
    if (host_args != other.host_args)
        return false;

    if (arg_values.size() != other.arg_values.size())
        return false;
    for (size_t i = 0; i < arg_values.size(); ++i) {
        if (arg_values[i].tensor_data() != other.arg_values[i].tensor_data()) {
            return false;
        }
    }

    return true;
}

uint64 CompilationCache::Signature::Hash::operator()(const Signature& signature) const
{
    uint64 h = std::hash<string>()(signature.name);
    for (const auto& arg : signature.arg_ranks) {
        h = Hash64Combine(h, std::hash<int>()(static_cast<int>(arg.first)));
        h = Hash64Combine(h, std::hash<int>()(arg.second));
    }
    for (const auto& arg : signature.arg_types) {
        h = Hash64Combine(h, std::hash<int>()(static_cast<int>(arg.first)));
        h = Hash64Combine(h, std::hash<int>()(arg.second.dims()));
        for (int dim : arg.second.dim_sizes()) {
            h = Hash64Combine(h, std::hash<int>()(dim));
        }
    }
    for (const auto& arg : signature.arg_values) {
        h = Hash64Combine(h, Hash64(arg.tensor_data().data(), arg.tensor_data().size()));
    }
    for (const int& host_arg_idx : signature.host_args) {
        h = Hash64Combine(h, std::hash<int>()(host_arg_idx));
    }
    return h;
}

Status CompilationCache::BuildSignature(const NameAttrList& function, const std::map<int, Tensor>& constant_args,
                                        const std::set<int>& fixed_shape_args, const std::set<int>& host_args,
                                        const std::map<int, OptionalTensor>& variable_args, OpKernelContext* ctx,
                                        Signature* signature, bool is_dynamic)
{
    signature->name = Canonicalize(function.name(), AttrSlice(&function.attr()));
    signature->arg_values.reserve(constant_args.size());
    if (is_dynamic) {
        signature->arg_types.reserve(fixed_shape_args.size());
        signature->arg_ranks.reserve(ctx->num_inputs() - constant_args.size() - fixed_shape_args.size());
    } else {
        signature->arg_types.reserve(ctx->num_inputs() - constant_args.size());
    }

    for (int i = 0; i < ctx->num_inputs(); ++i) {
        if (constant_args.count(i) > 0) {
            // Use the values of compile time constants in the signature.
            signature->arg_values.push_back(constant_args.at(i));
        } else if (variable_args.count(i) > 0) {
            const OptionalTensor& variable = variable_args.at(i);
            if (variable.present) {
                signature->arg_types.emplace_back(variable.value.dtype(), variable.value.shape());
            } else {
                signature->arg_types.emplace_back(tensorflow::DT_INVALID, TensorShape());
            }
        } else if ((!is_dynamic) || (fixed_shape_args.count(i))) {
            signature->arg_types.emplace_back(ctx->input_dtype(i), ctx->input(i).shape());
        } else {
            // if support dynamic
            signature->arg_ranks.emplace_back(ctx->input_dtype(i), ctx->input(i).dims());
        }
    }

    signature->host_args.insert(signature->host_args.end(), host_args.begin(), host_args.end());
    return absl::OkStatus();
}

std::string CompilationCache::DebugString() const
{
    return "NPU_XLA JIT compilation cache";
}

Status CompilationCache::Compile(std::unique_ptr<CompilerInput> input, const NameAttrList& function,
                                 const std::map<int, Tensor>& constant_args, const std::set<int>& fixed_shape_args,
                                 const std::set<int>& host_args, const std::map<int, OptionalTensor>& variable_args,
                                 OpKernelContext* ctx, Executable** executable)
{
    return CompileImpl(std::move(input), function, constant_args, fixed_shape_args, host_args, variable_args, ctx,
                       executable);
}

Status CompilationCache::CompileImpl(std::unique_ptr<CompilerInput> input, const NameAttrList& function,
                                     const std::map<int, Tensor>& constant_args, const std::set<int>& fixed_shape_args,
                                     const std::set<int>& host_args, const std::map<int, OptionalTensor>& variable_args,
                                     OpKernelContext* ctx, Executable** executable)
{
    Signature signature;
    TF_RETURN_IF_ERROR(
        BuildSignature(function, constant_args, fixed_shape_args, host_args, variable_args, ctx, &signature, false));
    VLOG(VLOG_LEVEL_2) << "Compile function " << function.name() << " with signature "
                       << SignatureDebugString(signature);

    Entry* entry = nullptr;
    bool new_entry = false;
    {
        mutex_lock l(compile_cache_mu_);
        std::unique_ptr<Entry>& e = cache_[signature];
        if (!e) {
            new_entry = true;
            e.reset(new Entry);
        }
        entry = e.get();
    }

    Signature::Hash hash;
    auto hash_value = hash(signature);
    input->mutable_options()->set_func_hash(hash_value);

    mutex_lock l(entry->mu);
    if (!entry->compiled) {
        VLOG(VLOG_LEVEL_2) << "Compile missing cache function " << function.name() << " with signature "
                           << SignatureDebugString(signature);
        entry->compiled = true;
        bool hit_cache = false;

        CompilationResultProto result_proto;
        if (!cache_path_.empty()) {
            std::string output_file_name;
            auto& persistent_cache = PersistentCompliationCache::Global();
            hit_cache = persistent_cache.find(signature, output_file_name);
            if (hit_cache) {
                TF_RETURN_IF_ERROR(ReadBinaryProto(tensorflow::Env::Default(), output_file_name, &result_proto));
            }
        }

        if (!hit_cache) {
            entry->compilation_status = PrepareCompilerInput(function, constant_args, fixed_shape_args, host_args,
                                                             variable_args, ctx, input.get());
            TF_RETURN_IF_ERROR(entry->compilation_status);

            std::string compile_dir;
            TF_RETURN_IF_ERROR(CreateCompilationPath(&compile_dir));
            entry->compilation_status = CompileFunction(function.name(), tf_mlir_bin_path_, compile_dir, *input, false);
            TF_RETURN_IF_ERROR(entry->compilation_status);
            result_proto.set_compiled_model_path(compile_dir);
        }

        CHECK_EQ(entry->executable.get(), nullptr);
        entry->executable = std::make_unique<Executable>(std::move(result_proto));
        if (!cache_path_.empty() && !hit_cache) {
            auto& disk_cache = PersistentCompliationCache::Global();
            auto filename = disk_cache.getNextUniqueNameOfCompiledResultProto();
            TF_RETURN_IF_ERROR(entry->executable->DumpToFile(filename));
            disk_cache.update(signature, filename, false, true);
        }
    }

    if (entry->compilation_status == absl::OkStatus()) {
        *executable = entry->executable.get();
    }

    return entry->compilation_status;
}

// Currently, we use a TF function to represent a compilable subgraph. TF
// function can be arbitrarily nested both using explict way (e.g. a call node)
// and implicit way (e.g. functional control flow ops).
// In order to compile a function, we need to dump all related functions (the
// closure of the entry function). Ideally we should only dump the functions
// that are called directly or indirectly by the entry function. Thus we prune
// the function library before we dump the input.
Status SerializeFunctionLibrary(CompilerInput* compiler_input, const NameAttrList& function,
                                FunctionLibraryRuntime* flib)
{
    auto flib_def = flib->GetFunctionLibraryDefinition();
    auto& options = *(compiler_input->mutable_options());
    auto func = flib_def->Find(function.name());
    TF_RET_CHECK(func != nullptr);

    auto reachable_flib = util::ReachableDefinitions(*flib_def, *func);
    TF_RET_CHECK(reachable_flib != nullptr);

    reachable_flib->ToProto().AppendToString(options.mutable_flib_def());

    return absl::OkStatus();
}

// 1. 序列化函数库并记录时间
Status SerializeFunctionLibraryWithTiming(const NameAttrList& function, FunctionLibraryRuntime* flib_def,
                                          CompilerInput* compiler_input)
{
    std::chrono::time_point<std::chrono::steady_clock> start;
    if (VLOG_IS_ON(1)) {
        start = std::chrono::steady_clock::now();
    }
    TF_RETURN_IF_ERROR(SerializeFunctionLibrary(compiler_input, function, flib_def));
    if (VLOG_IS_ON(1)) {
        std::chrono::duration<double> elapsed_sec = std::chrono::steady_clock::now() - start;
        VLOG(1) << "Dump FlibDef takes " << std::fixed << elapsed_sec.count() << " s";
    }

    function.AppendToString(compiler_input->mutable_function());
    if (VLOG_IS_ON(1)) {
        auto* func_def = flib_def->GetFunctionLibraryDefinition()->Find(function.name());
        CHECK(func_def != nullptr) << "Function not found in function library: " << function.name();
        VLOG(1) << "cluster size of function " << function.name() << ": " << func_def->node_def_size();
    }

    return absl::OkStatus();
}

// 2. 处理常量参数
void ProcessConstantArg(int64 input_num, const Tensor& input, CompilerInput::Argument* arg)
{
    CHECK(input.dtype() != tensorflow::DT_RESOURCE);
    arg->set_kind_v2(ArgumentKind::kConstant);
    arg->set_type(static_cast<int>(input.dtype()));
    tensorflow::TensorShapeProto shape_proto;
    input.shape().AsProto(&shape_proto);
    shape_proto.AppendToString(arg->mutable_shape());
    tensorflow::TensorProto tensor_proto;
    input.AsProtoTensorContent(&tensor_proto);
    tensor_proto.AppendToString(arg->mutable_constant_value());
}

// 3. 处理非常量参数
void ProcessNonConstantArg(int64 input_num, const Tensor& input, bool is_mlir, const std::set<int>& fixed_shape_args,
                           const std::set<int>& host_args, CompilerInput::Argument* arg)
{
    CHECK(input.dtype() != tensorflow::DT_RESOURCE);
    if (is_mlir) {
        if (fixed_shape_args.count(input_num) > 0) {
            // only used for Mlir dynamic shape compiler
            arg->set_kind_v2(ArgumentKind::kFixedShaped);
        } else if (host_args.count(input_num)) {
            arg->set_kind_v2(ArgumentKind::kHostArgs);
        } else {
            // only static shape for now
            // add dynamic shape support
            arg->set_kind_v2(ArgumentKind::kFixedShaped);
        }
    } else {
        if (input.NumElements() > 0) {
            arg->set_kind_v2(ArgumentKind::kParameter);
        } else {
            VLOG(2) << "set empty parameter #" << input_num << " to constant";
            arg->set_kind_v2(ArgumentKind::kConstant);
            tensorflow::TensorProto tensor_proto;
            input.AsProtoTensorContent(&tensor_proto);
            tensor_proto.AppendToString(arg->mutable_constant_value());
        }
    }
    arg->set_type(static_cast<int>(input.dtype()));
    tensorflow::TensorShapeProto shape_proto;
    input.shape().AsProto(&shape_proto);
    shape_proto.AppendToString(arg->mutable_shape());
}

// 4. 处理资源变量参数
void ProcessResourceArg(int64 input_num, const Tensor& input, const OptionalTensor& variable,
                        CompilerInput::Argument* arg)
{
    CHECK(input.dtype() == tensorflow::DT_RESOURCE);
    *arg->mutable_name() = variable.name;
    arg->set_kind_v2(ArgumentKind::kResource);
    arg->set_resource_kind_v2(ArgumentResourceKind::kVariable);
    if (variable.present) {
        const Tensor& value = variable.value;
        arg->set_type(static_cast<int>(value.dtype()));
        tensorflow::TensorShapeProto shape_proto;
        value.shape().AsProto(&shape_proto);
        shape_proto.AppendToString(arg->mutable_shape());
        arg->set_initialized(true);
    } else {
        // The values of uninitialized variables are not passed as inputs, since
        // they are meaningless. However, it is legal to assign to a resource
        // variable for the first time inside the XLA computation, so we do
        // permit uninitialized variables.
        arg->set_initialized(false);
        arg->set_type(static_cast<int>(tensorflow::DT_INVALID));
        tensorflow::TensorShapeProto shape_proto;
        tensorflow::TensorShape().AsProto(&shape_proto);
        shape_proto.AppendToString(arg->mutable_shape());
    }
}

// 5. 处理所有输入参数
Status ProcessAllInputArgs(const std::map<int, Tensor>& constant_args, const std::set<int>& fixed_shape_args,
                           const std::set<int>& host_args, const std::map<int, OptionalTensor>& variable_args,
                           OpKernelContext* ctx, CompilerInput* compiler_input, bool is_mlir)
{
    for (int64 input_num = 0; input_num < ctx->num_inputs(); ++input_num) {
        auto arg = compiler_input->add_args();
        if (constant_args.count(input_num) > 0) {
            // 处理编译时常量
            ProcessConstantArg(input_num, constant_args.at(input_num), arg);
        } else if (variable_args.count(input_num) == 0) {
            // 处理非常量参数
            ProcessNonConstantArg(input_num, ctx->input(input_num), is_mlir, fixed_shape_args, host_args, arg);
        } else {
            // 处理资源变量
            ProcessResourceArg(input_num, ctx->input(input_num), variable_args.at(input_num), arg);
        }
    }
    return absl::OkStatus();
}

// 6. 主函数 - PrepareCompilerInput
Status PrepareCompilerInput(const NameAttrList& function, const std::map<int, Tensor>& constant_args,
                            const std::set<int>& fixed_shape_args, const std::set<int>& host_args,
                            const std::map<int, OptionalTensor>& variable_args, OpKernelContext* ctx,
                            CompilerInput* compiler_input, bool is_mlir)
{
    auto flib_def = ctx->function_library();

    // 1. 序列化函数库并记录时间
    TF_RETURN_IF_ERROR(SerializeFunctionLibraryWithTiming(function, flib_def, compiler_input));

    // 2. 处理所有输入参数
    TF_RETURN_IF_ERROR(
        ProcessAllInputArgs(constant_args, fixed_shape_args, host_args, variable_args, ctx, compiler_input, is_mlir));

    return absl::OkStatus();
}

// 1. 准备编译输入文件
Status PrepareCompilationInputFiles(const std::string& compile_dir, CompilerInput& input, std::string& input_file_name,
                                    std::string& output_file_name, bool remove_after_compile)
{
    output_file_name = compile_dir + "/model.stablehlo";
    auto env = tensorflow::Env::Default();

    if (!env->LocalTempFilename(&input_file_name)) {
        return errors::Internal("couldn't get temp xla_compiler_input file name");
    }
    input_file_name = compile_dir + "/model.input";

    auto input_cleaner = tensorflow::gtl::MakeCleanup([&input_file_name, remove_after_compile] {
        if (remove_after_compile) {
            (void)tensorflow::Env::Default()->DeleteFile(input_file_name);
        } else {
            VLOG(0) << "xla_compiler_input: " << input_file_name;
        }
    });

    TF_RETURN_IF_ERROR(WriteBinaryProto(tensorflow::Env::Default(), input_file_name, input));

    if (VLOG_IS_ON(VLOG_LEVEL_2)) {
        std::string dbg_input_file_name = input_file_name + ".input_txt";
        VLOG(VLOG_LEVEL_2) << "Writing CompilerInput to " << dbg_input_file_name;
        TF_RETURN_IF_ERROR(WriteTextProto(tensorflow::Env::Default(), dbg_input_file_name, input));
    }

    return absl::OkStatus();
}

// 2. 执行TF MLIR编译过程
Status ExecuteTfMlirCompilation(const std::string& func_name, const std::string& tf_mlir_bin_path,
                                const std::string& input_file_name, const std::string& output_file_name)
{
    std::chrono::duration<double> elapsed_sec;
    auto start = std::chrono::steady_clock::now();
    tensorflow::npu_xla::SubProcess tf_mlir_bin;
    VLOG(0) << "compiling function " << func_name << ", input file is " << input_file_name << ", output file is "
            << output_file_name;

    std::vector<string> tf_mlir_args = {tf_mlir_bin_path, input_file_name, output_file_name};
    tf_mlir_bin.SetProgram(tf_mlir_bin_path, tf_mlir_args);
    tf_mlir_bin.SetChannelAction(tensorflow::npu_xla::CHAN_STDOUT, tensorflow::npu_xla::ACTION_PIPE);
    tf_mlir_bin.SetChannelAction(tensorflow::npu_xla::CHAN_STDERR, tensorflow::npu_xla::ACTION_PIPE);

    if (!tf_mlir_bin.Start()) {
        return errors::Internal("Failed to launch tf_mlir_bin: " + tf_mlir_bin_path);
    }

    string stdout_output;
    string stderr_output;
    int exitStatus = tf_mlir_bin.Communicate(nullptr, &stdout_output, &stderr_output);
    elapsed_sec = std::chrono::steady_clock::now() - start;

    std::stringstream ss;
    ss << "tf_mlir exits with errcode " << exitStatus << " in " << std::fixed << elapsed_sec.count()
       << " seconds to compile " << func_name << ":\n"
       << "============= stdout ===============\n"
       << stdout_output << "\n\n"
       << "============= stderr ===============\n"
       << stderr_output << "\n\n"
       << "====================================";

    if (VLOG_IS_ON(VLOG_LEVEL_2)) {
        VLOG(VLOG_LEVEL_2) << ss.str();
    }

    if (exitStatus != 0) {
        VLOG(0) << ss.str();
        return errors::Internal("Failed to compile function " + func_name + ", tf_mlir_bin exits with errcode " +
                                std::to_string(exitStatus));
    }

    return absl::OkStatus();
}

// 3. 执行StableHLO编译管道
Status ExecuteStableHloPipeline(const std::string& compile_dir)
{
    // clear old cache file if exist
    VLOG(1) << "Start CleanModelDirs";
    TF_RETURN_IF_ERROR(compiler::CleanModelDirs(compile_dir));

    // compile stablehlo
    VLOG(1) << "Start mlir complier::RunPipeline";
    TF_RETURN_IF_ERROR(compiler::RunPipeline(compile_dir));

    return absl::OkStatus();
}

// 4. 主函数 - CompileFunction
Status CompileFunction(const std::string& func_name, const std::string& tf_mlir_bin_path,
                       const std::string& compile_dir, CompilerInput& input, bool remove_after_compile)
{
    std::string input_file_name;
    std::string output_file_name;

    // 1. 准备编译输入文件
    TF_RETURN_IF_ERROR(
        PrepareCompilationInputFiles(compile_dir, input, input_file_name, output_file_name, remove_after_compile));

    // 2. 执行TF MLIR编译过程
    TF_RETURN_IF_ERROR(ExecuteTfMlirCompilation(func_name, tf_mlir_bin_path, input_file_name, output_file_name));

    // 3. 执行StableHLO编译管道
    TF_RETURN_IF_ERROR(ExecuteStableHloPipeline(compile_dir));

    return absl::OkStatus();
}

}  // namespace npu_xla
}  // namespace tensorflow