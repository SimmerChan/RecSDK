/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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
/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tf_bridge/tf/xla_op_registry.h"

#include <functional>
#include <memory>

#include "common_hdrs/types.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/local_device.h"
#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/framework/kernel_def.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/op_def_util.h"
#include "tensorflow/core/platform/mem.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"
#include "tensorflow/core/public/version.h"
#include "tf_bridge/tf/flags.h"
#include "tf_bridge/tf/xla_cluster_util.h"
#include "log.h"

namespace tensorflow {
namespace npu_xla {
const char* const DEVICE_NPU_XLA_JIT = "NPU_XLA_JIT";

static Status LaunchOpHasKernelForDeviceImpl(const DeviceType& device_type, const std::string& op_name)
{
    const OpDef* op_def;
    TF_RETURN_IF_ERROR(OpRegistry::Global()->LookUpOpDef(op_name, &op_def));
    NodeDef node_def;
    node_def.set_name("_" + op_name + "Op");
    node_def.set_op(op_name);
    string kernel_class_name;
    TF_RETURN_IF_ERROR(FindKernelDef(device_type, node_def, nullptr, &kernel_class_name));
    VLOG(VLOG_LEVEL_1) << "LaunchOpHasKernelForDevice"
                       << " kernel_class_name: " << kernel_class_name;
    return absl::OkStatus();
}

// Returns true if there is kernel of `TaoLaunch` or `TaoMlirLaunch` for the
// target device
static Status LaunchOpHasKernelForDevice(const DeviceType& device_type)
{
    return LaunchOpHasKernelForDeviceImpl(device_type, "NpuXlaLaunch");
}

XlaOpRegistry::XlaOpRegistry() = default;
XlaOpRegistry::~XlaOpRegistry() = default;

// compatibility if needed by future use cases.
bool XlaOpRegistry::IsCompatible(const OpRegistration& x, const OpRegistration& y)
{
    if (x.name != y.name)
        return true;
    if (x.label != y.label)
        return true;
    // The registrations refer to the same Op: ensures they are compatible and
    // are restricted to different device whitelists.
    if (x.compilation_only != y.compilation_only) {
        LOG(WARNING) << "Registrations of " << x.name << " have incompatible compilation_only settings.";
        return false;
    }
    if (x.allow_resource_types != y.allow_resource_types) {
        LOG(WARNING) << "Registrations of " << x.name << " have incompatible allow_resource_types settings.";
        return false;
    }
    if (x.allow_variant_types != y.allow_variant_types) {
        LOG(WARNING) << "Registrations of " << x.name << " have incompatible allow_variant_types settings.";
        return false;
    }
    if (x.allow_string_type != y.allow_string_type) {
        LOG(WARNING) << "Registrations of " << x.name << " have incompatible allow_string_type settings.";
        return false;
    }
    if (!x.has_device_whitelist && !y.has_device_whitelist) {
        LOG(WARNING) << "Duplicate registrations of " << x.name << "with no device whitelists.";
        return false;
    }
    if (x.has_device_whitelist && y.has_device_whitelist) {
        for (const auto& device : x.device_whitelist) {
            if (y.device_whitelist.count(device) != 0) {
                LOG(WARNING) << "Multiple registrations of " << x.name << " on device " << device;
                return false;
            }
        }
    }
    if (x.compile_time_constant_inputs != y.compile_time_constant_inputs) {
        LOG(WARNING) << "Registrations of " << x.name << " have incompatible compile time constant inputs.";
        return false;
    }
    if (x.is_metadata_op != y.is_metadata_op) {
        LOG(WARNING) << "Registrations of " << x.name << " have incompatible values for is_metadata_op.";
        return false;
    }
    return true;
}

void XlaOpRegistry::RegisterCompilationDevice(const string& device_name, const DeviceRegistration& registration)
{
    XlaOpRegistry& registry = Instance();
    mutex_lock lock(registry.mutex_);
    auto result = registry.compilation_devices_.emplace(device_name, registration);
    CHECK(result.second || result.first->second.compilation_device_name == registration.compilation_device_name);
}

void XlaOpRegistry::RegisterBackend(const string& compilation_device_name, absl::Span<const DataType> supported_types,
                                    BackendOpFilter op_filter)
{
    XlaOpRegistry& registry = Instance();
    mutex_lock lock(registry.mutex_);
    auto result = registry.backends_.emplace(compilation_device_name, Backend());
    CHECK(result.second) << "Duplicate XLA backend registration " << compilation_device_name;
    result.first->second.supported_types.insert(supported_types.begin(), supported_types.end());
    result.first->second.op_filter = op_filter;
}

bool XlaOpRegistry::GetCompilationDevice(const string& device_name, const DeviceRegistration** registration)
{
    XlaOpRegistry& registry = Instance();

    // Lazily register the CPU and GPU JIT devices the first time
    // GetCompilationDevice is called.
    static void* registration_init = [&registry]() {
        MarkForCompilationPassFlags* flags = GetMarkForCompilationPassFlags();
        bool cpu_global_jit = flags->tf_xla_cpu_global_jit;
        VLOG(VLOG_LEVEL_2) << "tf_xla_cpu_global_jit = " << cpu_global_jit;

        mutex_lock lock(registry.mutex_);
        if (LaunchOpHasKernelForDevice(DeviceType(DEVICE_NPU)).ok()) {
            DeviceRegistration& registration = registry.compilation_devices_[DEVICE_NPU];
            registration.compilation_device_name = DEVICE_NPU_XLA_JIT;
            registration.autoclustering_policy = XlaOpRegistry::AutoclusteringPolicy::kIfEnabledGlobally;
        }
        return nullptr;
    }();
    (void)registration_init;

    mutex_lock lock(registry.mutex_);
    auto it = registry.compilation_devices_.find(device_name);
    if (it == registry.compilation_devices_.end())
        return false;
    *registration = &it->second;
    return true;
}

void XlaOpRegistry::RegisterCompilationKernels()
{
    XlaOpRegistry& registry = Instance();
    mutex_lock lock(registry.mutex_);

    if (registry.jit_kernels_registered_)
        return;
    registry.jit_kernels_registered_ = true;

    OpRegistryInterface* op_registry = OpRegistry::Global();
    // Order of op registration:
    // The goal is to allow the co-existence of backend-specific kernels and
    // generic kernels. To achieve this, we enforce the following order of
    // registrations for one op:
    // 1. Process op registration with device whitelists:
    //      this pass registers backend-specific kernels for this op.
    // 2. Process op registration without device whitelists:
    //      this pass registers the kernels for all the other supported backends.
    for (auto& ops : registry.ops_) {
        const string& op_name = ops.first;
        std::vector<std::unique_ptr<OpRegistration>>& op_registrations = ops.second;
        // Partition the op registration so that the ones with device whitelists
        // precede the one without device whitelist.
        std::partition(op_registrations.begin(), op_registrations.end(),
                       [](const std::unique_ptr<OpRegistration>& op_reg) { return op_reg->has_device_whitelist; });

        // Collect a set of backend registered by ops with device whitelists.
        // The op registration without whitelists will register a generic kernel
        // for all other backends not in this set.
        std::unordered_set<string> whitelisted_backend;
        for (auto& op_registration : op_registrations) {
            if (op_registration->has_device_whitelist) {
                whitelisted_backend.insert(op_registration->device_whitelist.begin(),
                                           op_registration->device_whitelist.end());
            }
        }

        for (auto& op_registration : op_registrations) {
            const OpDef* op_def;
            Status lookup_status = op_registry->LookUpOpDef(op_name, &op_def);
            if (!lookup_status.ok()) {
                LOG(ERROR) << lookup_status.message();
            }
            TF_CHECK_OK(lookup_status);

            std::unordered_set<string> type_attrs;
            for (const OpDef::AttrDef& attr_def : op_def->attr()) {
                if (attr_def.type() == "type" || attr_def.type() == "list(type)") {
                    type_attrs.insert(attr_def.name());
                }
            }

            // Checks there are no type constraints referring to unknown attributes.
            for (const auto& constraint : op_registration->type_constraints) {
                if (type_attrs.find(constraint.first) == type_attrs.end()) {
                    LOG(FATAL) << "Unknown type attribute " << constraint.first << " in XLA op registration for "
                               << op_name;
                }
            }

            for (auto& backend : registry.backends_) {
                // If the operator has a device whitelist, only register on whitelisted
                // devices.
                if (op_registration->has_device_whitelist &&
                    op_registration->device_whitelist.find(backend.first) == op_registration->device_whitelist.end()) {
                    continue;
                }

                // If the operator does NOT has a device whitelist, skip all devices
                // that has already been registered.
                if (!op_registration->has_device_whitelist &&
                    whitelisted_backend.find(backend.first) != whitelisted_backend.end()) {
                    continue;
                }

                std::unique_ptr<KernelDef> kdef = std::make_unique<KernelDef>();
                kdef->set_op(op_registration->name);
                kdef->set_device_type(backend.first);
                kdef->set_label(op_registration->label);

                // Constrain each type attribute to the intersection of:
                // a) the types supported by the backend, and
                // b) the types allowed by the OpDef, and
                // c) the type constraints.
                bool unsatisfiable_type_constraint = false;
                for (const string& type_attr : type_attrs) {
                    KernelDef::AttrConstraint* attr_constraint = kdef->add_constraint();
                    attr_constraint->set_name(type_attr);
                    auto* allowed_values = attr_constraint->mutable_allowed_values()->mutable_list();

                    const OpDef::AttrDef& op_def_attr = *FindAttr(type_attr, *op_def);
                    const auto* op_def_allowed_types =
                        op_def_attr.has_allowed_values() ? &op_def_attr.allowed_values().list().type() : nullptr;
                    auto constraint_it = op_registration->type_constraints.find(type_attr);
                    const std::set<DataType>* type_constraints =
                        constraint_it != op_registration->type_constraints.end() ? &constraint_it->second : nullptr;
                    for (DataType dtype : backend.second.supported_types) {
                        // Filter out types that aren't allowed by the OpDef.
                        if (op_def_allowed_types != nullptr &&
                            std::find(op_def_allowed_types->begin(), op_def_allowed_types->end(), dtype) ==
                                op_def_allowed_types->end()) {
                            continue;
                        }
                        // Filter out types based on the type constraints.
                        if (type_constraints != nullptr && type_constraints->find(dtype) == type_constraints->end()) {
                            continue;
                        }
                        // Passed all the filters, this type is allowed.
                        allowed_values->add_type(dtype);
                    }
                    if (op_registration->allow_resource_types) {
                        allowed_values->add_type(DT_RESOURCE);
                    }
                    if (op_registration->allow_variant_types) {
                        allowed_values->add_type(DT_VARIANT);
                    }
                    if (op_registration->allow_string_type) {
                        allowed_values->add_type(DT_STRING);
                    }
                    // Don't build KernelDefs that have unsatisfiable type constraints.
                    if (allowed_values->type().empty()) {
                        unsatisfiable_type_constraint = true;
                        break;
                    }
                }
                if (unsatisfiable_type_constraint)
                    continue;

                if (backend.second.op_filter != nullptr && !backend.second.op_filter(kdef.get())) {
                    continue;
                }
                VLOG(VLOG_LEVEL_2) << "XLA op registration: device: " << backend.first << " op: " << op_name;
                registry.kernel_registrars_.emplace_back(
                    new kernel_factory::OpKernelRegistrar(new KernelDef(*kdef), "XlaJitOp", op_registration->factory));
                backend.second.kernel_defs.push_back(std::move(kdef));
            }
        }
    }
}

std::vector<const KernelDef*> XlaOpRegistry::DeviceKernels(const string& compilation_device_name,
                                                           bool include_compilation_only_kernels)
{
    // Ensure compilation kernels registered.
    RegisterCompilationKernels();
    std::vector<const KernelDef*> kernels;
    XlaOpRegistry& registry = Instance();
    mutex_lock lock(registry.mutex_);
    auto it = registry.backends_.find(compilation_device_name);
    CHECK(it != registry.backends_.end()) << "Unknown backend " << compilation_device_name;
    for (const std::unique_ptr<KernelDef>& k : it->second.kernel_defs) {
        auto op_iter = registry.ops_.find(k->op());
        CHECK(op_iter != registry.ops_.end() && !op_iter->second.empty());
        // The test in IsCompatible ensures that if there are multiple matching
        // registrations for this op name, they all have the same value of
        // compilation_only, so only the first match needs to be tested.
        if (include_compilation_only_kernels || !op_iter->second.front()->compilation_only) {
            kernels.push_back(k.get());
        }
    }
    return kernels;
}

std::vector<string> XlaOpRegistry::GetAllRegisteredOps()
{
    std::vector<string> ops;
    XlaOpRegistry& registry = Instance();
    mutex_lock lock(registry.mutex_);
    for (const auto& pair : registry.ops_) {
        ops.push_back(pair.first);
    }
    std::sort(ops.begin(), ops.end());
    return ops;
}

Status XlaOpRegistry::CompileTimeConstantInputs(const NodeDef& node_def, const OpKernel* op_kernel, const OpDef* op_def,
                                                std::vector<int>* result, CompileTimeConstType type)
{
    result->clear();

    DCHECK(op_def != nullptr || op_kernel != nullptr);

    std::unordered_set<string> compile_time_special_inputs_from_attr;
    std::vector<string> compile_time_special_inputs_vect_from_attr;

    const std::unordered_set<string>* compile_time_special_inputs;

    std::string Attr;
    if (type == CompileTimeConstType::kMlirCompileTimeConstantInput) {
        Attr = kMlirCompileTimeConstantInputsAttr;
    } else if (type == CompileTimeConstType::kMlirCompileTimeFixedShapeInput) {
        Attr = kMlirCompileTimeFixedShapeInputsAttr;
    } else {
        Attr = kXlaCompileTimeConstantInputsAttr;
    }

    if (GetNodeAttr(node_def, Attr, &compile_time_special_inputs_vect_from_attr).ok()) {
        std::copy(compile_time_special_inputs_vect_from_attr.begin(), compile_time_special_inputs_vect_from_attr.end(),
                  std::inserter(compile_time_special_inputs_from_attr, compile_time_special_inputs_from_attr.end()));
        compile_time_special_inputs = &compile_time_special_inputs_from_attr;
    } else {
        const string& op = node_def.op();

        XlaOpRegistry& registry = Instance();
        mutex_lock lock(registry.mutex_);
        auto it = registry.ops_.find(op);
        if (it == registry.ops_.end() || it->second.empty()) {
            return absl::OkStatus();
        } else {
            // The test in IsCompatible ensures that if there are multiple matching
            // registrations for this op name, they all have the same value of
            // compile_time_special_inputs, so only the first match is returned.
            if (type == CompileTimeConstType::kXlaCompileTimeConstantInput) {
                compile_time_special_inputs = &it->second.front()->compile_time_constant_inputs;
            } else if (type == CompileTimeConstType::kMlirCompileTimeConstantInput) {
                compile_time_special_inputs = &it->second.front()->mlir_compile_time_constant_inputs;
            } else if (type == CompileTimeConstType::kMlirCompileTimeFixedShapeInput) {
                compile_time_special_inputs = &it->second.front()->mlir_compile_time_fixed_shape_inputs;
            }
        }
    }

    for (const string& input : *compile_time_special_inputs) {
        if (op_def) {
            NameRangeMap input_name_ranges;
            TF_RETURN_IF_ERROR(NameRangesForNode(node_def, *op_def, &input_name_ranges, nullptr));
            auto name_range = input_name_ranges.find(input);
            if (name_range == input_name_ranges.end()) {
                continue;
            }

            for (int i = name_range->second.first; i < name_range->second.second; i++) {
                result->push_back(i);
            }
        } else {
            int start;
            int stop;
            TF_CHECK_OK(op_kernel->InputRange(input, &start, &stop));
            for (int i = start; i < stop; ++i) {
                result->push_back(i);
            }
        }
    }

    std::sort(result->begin(), result->end());
    return absl::OkStatus();
}

bool XlaOpRegistry::IsMetadataOp(const string& op)
{
    XlaOpRegistry& registry = Instance();
    mutex_lock lock(registry.mutex_);
    auto it = registry.ops_.find(op);
    if (it == registry.ops_.end() || it->second.empty()) {
        return false;
    }

    // The test in IsCompatible ensures that if there are multiple matching
    // registrations for this op name, they all have the same value of
    // is_metadata_op, so only the first match is returned.
    return it->second.front()->is_metadata_op;
}

std::vector<string> XlaOpRegistry::BackendNames()
{
    std::vector<string> names;
    XlaOpRegistry& registry = Instance();
    mutex_lock lock(registry.mutex_);
    for (const auto& backend_pair : registry.backends_) {
        names.push_back(backend_pair.first);
    }
    return names;
}

bool XlaOpRegistry::IsBackendRegistered(const string& name)
{
    XlaOpRegistry& registry = Instance();
    mutex_lock lock(registry.mutex_);
    return registry.backends_.find(name) != registry.backends_.end();
}

XlaOpRegistry& XlaOpRegistry::Instance()
{
    static XlaOpRegistry* r = new XlaOpRegistry;
    return *r;
}

XlaOpRegistrationBuilder::XlaOpRegistrationBuilder(absl::string_view name)
{
    registration_.reset(new XlaOpRegistry::OpRegistration);
    registration_->name = string(name);
}

XlaOpRegistrationBuilder XlaOpRegistrationBuilder::Name(absl::string_view name)
{
    XlaOpRegistrationBuilder registration(name);
    return registration;
}

XlaOpRegistrationBuilder& XlaOpRegistrationBuilder::Device(absl::Span<const absl::string_view> devices)
{
    registration_->has_device_whitelist = true;
    for (absl::string_view device : devices) {
        registration_->device_whitelist.emplace(device);
    }
    return *this;
}

XlaOpRegistrationBuilder& XlaOpRegistrationBuilder::Device(absl::string_view device)
{
    registration_->has_device_whitelist = true;
    registration_->device_whitelist.emplace(device);
    return *this;
}

XlaOpRegistrationBuilder& XlaOpRegistrationBuilder::CompilationOnly()
{
    registration_->compilation_only = true;
    return *this;
}

XlaOpRegistrationBuilder& XlaOpRegistrationBuilder::AllowResourceTypes()
{
    registration_->allow_resource_types = true;
    return *this;
}

XlaOpRegistrationBuilder& XlaOpRegistrationBuilder::AllowVariantTypes()
{
    registration_->allow_variant_types = true;
    return *this;
}

XlaOpRegistrationBuilder& XlaOpRegistrationBuilder::AllowStringType()
{
    registration_->allow_string_type = true;
    return *this;
}

XlaOpRegistrationBuilder& XlaOpRegistrationBuilder::TypeConstraint(absl::string_view attr_name, DataType allowed)
{
    std::set<DataType>& types = registration_->type_constraints[string(attr_name)];
    types.insert(allowed);
    return *this;
}

XlaOpRegistrationBuilder& XlaOpRegistrationBuilder::TypeConstraint(absl::string_view attr_name,
                                                                   absl::Span<const DataType> allowed)
{
    std::set<DataType>& types = registration_->type_constraints[string(attr_name)];
    for (DataType t : allowed) {
        types.insert(t);
    }
    return *this;
}

XlaOpRegistrationBuilder& XlaOpRegistrationBuilder::CompileTimeConstInput(absl::string_view input_name)
{
    registration_->compile_time_constant_inputs.emplace(input_name);
    return *this;
}

XlaOpRegistrationBuilder& XlaOpRegistrationBuilder::MlirCompileTimeFixedShapeInput(absl::string_view input_name)
{
    registration_->mlir_compile_time_fixed_shape_inputs.emplace(input_name);
    return *this;
}

XlaOpRegistrationBuilder& XlaOpRegistrationBuilder::MlirCompileTimeConstInput(absl::string_view input_name)
{
    registration_->mlir_compile_time_constant_inputs.emplace(input_name);
    return *this;
}

XlaOpRegistrationBuilder& XlaOpRegistrationBuilder::IsMetadataOp()
{
    registration_->is_metadata_op = true;
    return *this;
}

XlaOpRegistrationBuilder& XlaOpRegistrationBuilder::Label(std::string label)
{
    registration_->label = label;
    return *this;
}

std::unique_ptr<XlaOpRegistry::OpRegistration> XlaOpRegistrationBuilder::Build(XlaOpRegistry::Factory factory)
{
    registration_->factory = factory;
    return std::move(registration_);
}

XlaOpRegistrar::XlaOpRegistrar(std::unique_ptr<XlaOpRegistry::OpRegistration> registration)
{
    XlaOpRegistry& registry = XlaOpRegistry::Instance();
    mutex_lock lock(registry.mutex_);
    auto& existing_ops = registry.ops_[registration->name];
    for (auto& existing : existing_ops) {
        if (!XlaOpRegistry::IsCompatible(*existing, *registration)) {
            LOG(FATAL) << "XLA op registration " << registration->name
                       << " is incompatible with existing registration of the same name.";
        }
    }
    existing_ops.emplace_back(std::move(registration));
}

XlaBackendRegistrar::XlaBackendRegistrar(absl::string_view name, absl::Span<const DataType> types,
                                         XlaOpRegistry::BackendOpFilter op_filter)
{
    XlaOpRegistry& registry = XlaOpRegistry::Instance();
    registry.RegisterBackend(string(name), types, op_filter);
    AddSymbolicExecutionDevice(name);
}

namespace {

void AddDtypeToKernalDefConstraint(absl::string_view name, DataType dtype, KernelDef* kdef)
{
    for (KernelDef::AttrConstraint& constraint : *kdef->mutable_constraint()) {
        if (constraint.name() == name) {
            constraint.mutable_allowed_values()->mutable_list()->add_type(dtype);
        }
    }
}

bool GpuOpFilter(KernelDef* kdef)
{
    if (kdef->op() == "Const") {
        AddDtypeToKernalDefConstraint("dtype", DT_STRING, kdef);
    }
    if (kdef->op() == "Assert") {
        AddDtypeToKernalDefConstraint("T", DT_STRING, kdef);
    }
    return true;
}

}  // namespace

// clang-format off
REGISTER_XLA_BACKEND_FOR_NPU(DEVICE_NPU_XLA_JIT, kGpuAllTypes, GpuOpFilter);

// register ops
REGISTER_XLA_OP_FOR_NPU(Name("MaxPool"));
REGISTER_XLA_OP_FOR_NPU(Name("MaxPoolV2")
                            .CompileTimeConstInput("ksize")
                            .CompileTimeConstInput("strides")
                            .MlirCompileTimeFixedShapeInput("ksize")
                            .MlirCompileTimeFixedShapeInput("strides"));
REGISTER_XLA_OP_FOR_NPU(Name("MaxPool3D"));
REGISTER_XLA_OP_FOR_NPU(Name("AvgPool"));
REGISTER_XLA_OP_FOR_NPU(Name("AvgPool3D"));
REGISTER_XLA_OP_FOR_NPU(Name("MaxPoolGrad"));
REGISTER_XLA_OP_FOR_NPU(Name("MaxPoolGradV2")
                            .CompileTimeConstInput("ksize")
                            .CompileTimeConstInput("strides")
                            .MlirCompileTimeFixedShapeInput("ksize")
                            .MlirCompileTimeFixedShapeInput("strides"));
REGISTER_XLA_OP_FOR_NPU(Name("MaxPool3DGrad"));
REGISTER_XLA_OP_FOR_NPU(Name("AvgPoolGrad")
                            .CompileTimeConstInput("orig_input_shape")
                            .MlirCompileTimeFixedShapeInput("orig_input_shape"));
REGISTER_XLA_OP_FOR_NPU(Name("AvgPool3DGrad")
                            .CompileTimeConstInput("orig_input_shape")
                            .MlirCompileTimeFixedShapeInput("orig_input_shape"));
REGISTER_XLA_OP_FOR_NPU(Name("MaxPoolGradGrad").TypeConstraint("T", DT_FLOAT));
REGISTER_XLA_OP_FOR_NPU(Name("MaxPoolGradGradV2")
                            .TypeConstraint("T", DT_FLOAT)
                            .CompileTimeConstInput("ksize")
                            .CompileTimeConstInput("strides")
                            .MlirCompileTimeFixedShapeInput("ksize")
                            .MlirCompileTimeFixedShapeInput("strides"));
REGISTER_XLA_OP_FOR_NPU(Name("MaxPool3DGradGrad").TypeConstraint("T", DT_FLOAT));
REGISTER_XLA_OP_FOR_NPU(Name("Softmax"));
REGISTER_XLA_OP_FOR_NPU(Name("LogSoftmax"));
REGISTER_XLA_OP_FOR_NPU(Name("SoftmaxCrossEntropyWithLogits"));
REGISTER_XLA_OP_FOR_NPU(Name("SparseSoftmaxCrossEntropyWithLogits"));
REGISTER_XLA_OP_FOR_NPU(Name("NoOp").CompilationOnly());
REGISTER_XLA_OP_FOR_NPU(Name("ControlTrigger").CompilationOnly());
REGISTER_XLA_OP_FOR_NPU(Name("_Arg").AllowResourceTypes().CompilationOnly());
REGISTER_XLA_OP_FOR_NPU(Name("Reshape")
                            .CompileTimeConstInput("shape")
                            .MlirCompileTimeFixedShapeInput("shape"));
REGISTER_XLA_OP_FOR_NPU(Name("MatMul").TypeConstraint(
    "T", {DT_HALF, DT_BFLOAT16, DT_FLOAT, DT_DOUBLE, DT_COMPLEX64}));
REGISTER_XLA_OP_FOR_NPU(Name("SparseMatMul"));
REGISTER_XLA_OP_FOR_NPU(Name("Concat")
                            .CompileTimeConstInput("concat_dim")
                            .MlirCompileTimeConstInput("concat_dim"));
REGISTER_XLA_OP_FOR_NPU(Name("ConcatV2")
                            .TypeConstraint("Tidx", DT_INT32)
                            .CompileTimeConstInput("axis")
                            .MlirCompileTimeConstInput("axis"));
REGISTER_XLA_OP_FOR_NPU(Name("ConcatOffset")
                            .CompileTimeConstInput("concat_dim")
                            .CompileTimeConstInput("shape")
                            .MlirCompileTimeConstInput("concat_dim")
                            .MlirCompileTimeFixedShapeInput("shape"));
REGISTER_XLA_OP_FOR_NPU(Name("Fill")
                            .CompileTimeConstInput("dims")
                            .MlirCompileTimeFixedShapeInput("dims")
                            );
REGISTER_XLA_OP_FOR_NPU(Name("RGBToHSV"));
REGISTER_XLA_OP_FOR_NPU(Name("HSVToRGB"));
REGISTER_XLA_OP_FOR_NPU(Name("AdjustContrastv2"));
REGISTER_XLA_OP_FOR_NPU(Name("AdjustSaturation"));
REGISTER_XLA_OP_FOR_NPU(Name("AdjustHue"));
REGISTER_XLA_OP_FOR_NPU(Name("NonMaxSuppressionV4")
                            .CompileTimeConstInput("max_output_size")
                            .MlirCompileTimeFixedShapeInput("max_output_size"));
REGISTER_XLA_OP_FOR_NPU(Name("StatelessRandomUniform")
                            .CompileTimeConstInput("shape")
                            .MlirCompileTimeFixedShapeInput("shape")
                            .TypeConstraint("dtype", DT_FLOAT)
                            .TypeConstraint("Tseed", DT_INT32));
REGISTER_XLA_OP_FOR_NPU(Name("StatelessRandomNormal")
                            .CompileTimeConstInput("shape")
                            .MlirCompileTimeFixedShapeInput("shape")
                            .TypeConstraint("dtype", DT_FLOAT)
                            .TypeConstraint("Tseed", DT_INT32));
REGISTER_XLA_OP_FOR_NPU(Name("StatelessTruncatedNormal")
                            .CompileTimeConstInput("shape")
                            .MlirCompileTimeFixedShapeInput("shape")
                            .TypeConstraint("dtype", DT_FLOAT)
                            .TypeConstraint("Tseed", DT_INT32));
REGISTER_XLA_OP_FOR_NPU(Name("MatrixSetDiag"));
REGISTER_XLA_OP_FOR_NPU(Name("Identity").AllowResourceTypes().CompilationOnly());
REGISTER_XLA_OP_FOR_NPU(Name("IdentityN").AllowResourceTypes().CompilationOnly());
REGISTER_XLA_OP_FOR_NPU(Name("PlaceholderWithDefault"));
REGISTER_XLA_OP_FOR_NPU(Name("PreventGradient"));
REGISTER_XLA_OP_FOR_NPU(Name("StopGradient"));
REGISTER_XLA_OP_FOR_NPU(Name("Snapshot"));
REGISTER_XLA_OP_FOR_NPU(Name("BatchToSpaceND")
                            .CompileTimeConstInput("block_shape")
                            .CompileTimeConstInput("crops")
                            .MlirCompileTimeFixedShapeInput("block_shape")
                            .MlirCompileTimeFixedShapeInput("crops"));
REGISTER_XLA_OP_FOR_NPU(Name("BatchToSpace")
                            .CompileTimeConstInput("crops")
                            .MlirCompileTimeFixedShapeInput("crops"));
REGISTER_XLA_OP_FOR_NPU(Name("ListDiff")
                            .TypeConstraint("T", {DT_INT32, DT_INT64})
                            .CompileTimeConstInput("x")
                            .CompileTimeConstInput("y")
                            .MlirCompileTimeFixedShapeInput("x")
                            .MlirCompileTimeFixedShapeInput("y"));
REGISTER_XLA_OP_FOR_NPU(Name("BatchMatMul"));
REGISTER_XLA_OP_FOR_NPU(Name("MatrixBandPart"));
REGISTER_XLA_OP_FOR_NPU(Name("Range")
                            .CompileTimeConstInput("start")
                            .CompileTimeConstInput("limit")
                            .CompileTimeConstInput("delta")
                            .MlirCompileTimeFixedShapeInput("start")
                            .MlirCompileTimeFixedShapeInput("limit")
                            .MlirCompileTimeFixedShapeInput("delta"));
REGISTER_XLA_OP_FOR_NPU(Name("LinSpace")
                            .CompileTimeConstInput("start")
                            .CompileTimeConstInput("stop")
                            .CompileTimeConstInput("num")
                            .MlirCompileTimeFixedShapeInput("start")
                            .MlirCompileTimeFixedShapeInput("stop")
                            .MlirCompileTimeFixedShapeInput("num"));
REGISTER_XLA_OP_FOR_NPU(Name("FusedBatchNorm"));
REGISTER_XLA_OP_FOR_NPU(Name("FusedBatchNormV2"));
REGISTER_XLA_OP_FOR_NPU(Name("FusedBatchNormGrad"));
REGISTER_XLA_OP_FOR_NPU(Name("FusedBatchNormGradV2"));
REGISTER_XLA_OP_FOR_NPU(Name("TensorArrayV3").CompileTimeConstInput("size"));
REGISTER_XLA_OP_FOR_NPU(Name("TensorArrayWriteV3"));
REGISTER_XLA_OP_FOR_NPU(Name("TensorArrayReadV3"));
REGISTER_XLA_OP_FOR_NPU(Name("TensorArrayGatherV3"));
REGISTER_XLA_OP_FOR_NPU(Name("TensorArrayScatterV3"));
REGISTER_XLA_OP_FOR_NPU(Name("TensorArrayConcatV3"));
REGISTER_XLA_OP_FOR_NPU(Name("TensorArraySplitV3")
                            .CompileTimeConstInput("lengths")
                            .MlirCompileTimeFixedShapeInput("lengths"));
REGISTER_XLA_OP_FOR_NPU(Name("TensorArraySizeV3"));
REGISTER_XLA_OP_FOR_NPU(Name("TensorArrayGradV3"));
REGISTER_XLA_OP_FOR_NPU(Name("TensorArrayCloseV3"));
REGISTER_XLA_OP_FOR_NPU(Name("_Retval").AllowResourceTypes().CompilationOnly());
REGISTER_XLA_OP_FOR_NPU(Name("Tile")
                            .CompileTimeConstInput("multiples")
                            .MlirCompileTimeFixedShapeInput("multiples"));
REGISTER_XLA_OP_FOR_NPU(Name("AddN"));
REGISTER_XLA_OP_FOR_NPU(Name("While").AllowResourceTypes());
REGISTER_XLA_OP_FOR_NPU(Name("StatelessWhile").AllowResourceTypes());
REGISTER_XLA_OP_FOR_NPU(Name("FFT").TypeConstraint("Tcomplex", DT_COMPLEX64));
REGISTER_XLA_OP_FOR_NPU(Name("FFT2D").TypeConstraint("Tcomplex", DT_COMPLEX64));
REGISTER_XLA_OP_FOR_NPU(Name("FFT3D").TypeConstraint("Tcomplex", DT_COMPLEX64));
REGISTER_XLA_OP_FOR_NPU(Name("IFFT").TypeConstraint("Tcomplex", DT_COMPLEX64));
REGISTER_XLA_OP_FOR_NPU(Name("IFFT2D").TypeConstraint("Tcomplex",
                                                      DT_COMPLEX64));
REGISTER_XLA_OP_FOR_NPU(Name("IFFT3D").TypeConstraint("Tcomplex",
                                                      DT_COMPLEX64));
REGISTER_XLA_OP_FOR_NPU(Name("RFFT")
                            .CompileTimeConstInput("fft_length")
                            .MlirCompileTimeFixedShapeInput("fft_length"));
REGISTER_XLA_OP_FOR_NPU(Name("RFFT2D")
                            .CompileTimeConstInput("fft_length")
                            .MlirCompileTimeFixedShapeInput("fft_length"));
REGISTER_XLA_OP_FOR_NPU(Name("RFFT3D")
                            .CompileTimeConstInput("fft_length")
                            .MlirCompileTimeFixedShapeInput("fft_length"));
REGISTER_XLA_OP_FOR_NPU(Name("IRFFT")
                            .CompileTimeConstInput("fft_length")
                            .MlirCompileTimeFixedShapeInput("fft_length"));
REGISTER_XLA_OP_FOR_NPU(Name("IRFFT2D")
                            .CompileTimeConstInput("fft_length")
                            .MlirCompileTimeFixedShapeInput("fft_length"));
REGISTER_XLA_OP_FOR_NPU(Name("IRFFT3D")
                            .CompileTimeConstInput("fft_length")
                            .MlirCompileTimeFixedShapeInput("fft_length"));
REGISTER_XLA_OP_FOR_NPU(Name("QuantizeAndDequantizeV2"));
REGISTER_XLA_OP_FOR_NPU(Name("QuantizeAndDequantizeV3"));
REGISTER_XLA_OP_FOR_NPU(Name("Conv2D"));
REGISTER_XLA_OP_FOR_NPU(Name("Conv3D"));
REGISTER_XLA_OP_FOR_NPU(Name("DepthwiseConv2dNative"));
REGISTER_XLA_OP_FOR_NPU(Name("Conv2DBackpropInput")
                            .CompileTimeConstInput("input_sizes")
                            .MlirCompileTimeFixedShapeInput("input_sizes"));
REGISTER_XLA_OP_FOR_NPU(Name("Conv3DBackpropInputV2")
                            .CompileTimeConstInput("input_sizes")
                            .MlirCompileTimeFixedShapeInput("input_sizes"));
REGISTER_XLA_OP_FOR_NPU(Name("DepthwiseConv2dNativeBackpropInput")
                            .CompileTimeConstInput("input_sizes")
                            .MlirCompileTimeFixedShapeInput("input_sizes"));
REGISTER_XLA_OP_FOR_NPU(Name("Conv2DBackpropFilter")
                            .CompileTimeConstInput("filter_sizes")
                            .MlirCompileTimeFixedShapeInput("filter_sizes"));
REGISTER_XLA_OP_FOR_NPU(Name("Conv3DBackpropFilterV2")
                            .CompileTimeConstInput("filter_sizes")
                            .MlirCompileTimeFixedShapeInput("filter_sizes"));
REGISTER_XLA_OP_FOR_NPU(Name("DepthwiseConv2dNativeBackpropFilter")
                            .CompileTimeConstInput("filter_sizes")
                            .MlirCompileTimeFixedShapeInput("filter_sizes"));
REGISTER_XLA_OP_FOR_NPU(Name("MirrorPad")
                            .CompileTimeConstInput("paddings")
                            .MlirCompileTimeFixedShapeInput("paddings"));
REGISTER_XLA_OP_FOR_NPU(Name("Elu"));
REGISTER_XLA_OP_FOR_NPU(Name("EluGrad"));
REGISTER_XLA_OP_FOR_NPU(Name("Selu"));
REGISTER_XLA_OP_FOR_NPU(Name("SeluGrad"));
constexpr std::array<DataType, 3> kScanOpTypes = {
    {DT_HALF, DT_BFLOAT16, DT_FLOAT}};
REGISTER_XLA_OP_FOR_NPU(Name("Cumsum")
                            .TypeConstraint("T", kScanOpTypes)
                            .CompileTimeConstInput("axis")
                            .MlirCompileTimeConstInput("axis"));
REGISTER_XLA_OP_FOR_NPU(Name("Cumprod")
                            .TypeConstraint("T", kScanOpTypes)
                            .CompileTimeConstInput("axis")
                            .MlirCompileTimeConstInput("axis"));
REGISTER_XLA_OP_FOR_NPU(Name("Split")
                            .CompileTimeConstInput("split_dim")
                            .MlirCompileTimeConstInput("split_dim")
                            );
REGISTER_XLA_OP_FOR_NPU(Name("SplitV")
                            .CompileTimeConstInput("split_dim")
                            .CompileTimeConstInput("size_splits")
                            .MlirCompileTimeConstInput("split_dim")
                            .MlirCompileTimeFixedShapeInput("size_splits"));
REGISTER_XLA_OP_FOR_NPU(Name("ClipByValue"));
REGISTER_XLA_OP_FOR_NPU(Name("ExtractImagePatches"));
REGISTER_XLA_OP_FOR_NPU(Name("Shape").CompilationOnly().IsMetadataOp());
REGISTER_XLA_OP_FOR_NPU(Name("ShapeN").CompilationOnly().IsMetadataOp());
REGISTER_XLA_OP_FOR_NPU(Name("Rank").CompilationOnly().IsMetadataOp());
REGISTER_XLA_OP_FOR_NPU(Name("Size").CompilationOnly().IsMetadataOp());
REGISTER_XLA_OP_FOR_NPU(Name("ExpandDims")
                            .CompileTimeConstInput("dim")
                            .MlirCompileTimeConstInput("dim"));
REGISTER_XLA_OP_FOR_NPU(Name("Squeeze"));
REGISTER_XLA_OP_FOR_NPU(Name("ZerosLike"));
REGISTER_XLA_OP_FOR_NPU(Name("OnesLike"));
REGISTER_XLA_OP_FOR_NPU(Name("If").AllowResourceTypes());
REGISTER_XLA_OP_FOR_NPU(Name("StatelessIf").AllowResourceTypes());
REGISTER_XLA_OP_FOR_NPU(Name("RandomUniform")
                            .CompileTimeConstInput("shape")
                            .MlirCompileTimeFixedShapeInput("shape"));
REGISTER_XLA_OP_FOR_NPU(Name("RandomShuffle"));
REGISTER_XLA_OP_FOR_NPU(Name("RandomUniformInt")
                            .CompileTimeConstInput("shape")
                            .MlirCompileTimeFixedShapeInput("shape"));
REGISTER_XLA_OP_FOR_NPU(Name("RandomStandardNormal")
                            .CompileTimeConstInput("shape")
                            .MlirCompileTimeFixedShapeInput("shape"));
REGISTER_XLA_OP_FOR_NPU(Name("TruncatedNormal")
                            .CompileTimeConstInput("shape")
                            .MlirCompileTimeFixedShapeInput("shape")
                            .TypeConstraint("dtype", DT_FLOAT));
REGISTER_XLA_OP_FOR_NPU(Name("Multinomial")
                            .CompileTimeConstInput("num_samples")
                            .MlirCompileTimeFixedShapeInput("num_samples"));
REGISTER_XLA_OP_FOR_NPU(Name("SpaceToDepth"));
REGISTER_XLA_OP_FOR_NPU(Name("Pad")
                            .CompileTimeConstInput("paddings")
                            .MlirCompileTimeFixedShapeInput("paddings"));
REGISTER_XLA_OP_FOR_NPU(Name("PadV2")
                            .CompileTimeConstInput("paddings")
                            .MlirCompileTimeFixedShapeInput("paddings"));
REGISTER_XLA_OP_FOR_NPU(Name("Sum")
                            .CompileTimeConstInput("reduction_indices")
                            .MlirCompileTimeConstInput("reduction_indices"));
REGISTER_XLA_OP_FOR_NPU(Name("Prod")
                            .CompileTimeConstInput("reduction_indices")
                            .MlirCompileTimeConstInput("reduction_indices"));
REGISTER_XLA_OP_FOR_NPU(Name("Min")
                            .CompileTimeConstInput("reduction_indices")
                            .MlirCompileTimeConstInput("reduction_indices"));
REGISTER_XLA_OP_FOR_NPU(Name("Max")
                            .CompileTimeConstInput("reduction_indices")
                            .MlirCompileTimeConstInput("reduction_indices"));
REGISTER_XLA_OP_FOR_NPU(Name("Mean")
                            .CompileTimeConstInput("reduction_indices")
                            .MlirCompileTimeConstInput("reduction_indices"));
REGISTER_XLA_OP_FOR_NPU(Name("All")
                            .CompileTimeConstInput("reduction_indices")
                            .MlirCompileTimeConstInput("reduction_indices"));
REGISTER_XLA_OP_FOR_NPU(Name("Any")
                            .CompileTimeConstInput("reduction_indices")
                            .MlirCompileTimeConstInput("reduction_indices"));
REGISTER_XLA_OP_FOR_NPU(Name("Reverse")
                            .CompileTimeConstInput("dims")
                            .MlirCompileTimeConstInput("dims"));
REGISTER_XLA_OP_FOR_NPU(Name("ReverseV2")
                            .CompileTimeConstInput("axis")
                            .MlirCompileTimeConstInput("axis"));
REGISTER_XLA_OP_FOR_NPU(Name("UnsortedSegmentSum")
                            .CompileTimeConstInput("num_segments")
                            .MlirCompileTimeFixedShapeInput("num_segments"));
REGISTER_XLA_OP_FOR_NPU(Name("UnsortedSegmentProd")
                            .CompileTimeConstInput("num_segments")
                            .MlirCompileTimeFixedShapeInput("num_segments"));
REGISTER_XLA_OP_FOR_NPU(Name("UnsortedSegmentMin")
                            .CompileTimeConstInput("num_segments")
                            .MlirCompileTimeFixedShapeInput("num_segments"));
REGISTER_XLA_OP_FOR_NPU(Name("UnsortedSegmentMax")
                            .CompileTimeConstInput("num_segments")
                            .MlirCompileTimeFixedShapeInput("num_segments"));
REGISTER_XLA_OP_FOR_NPU(Name("SpaceToBatchND")
                            .CompileTimeConstInput("paddings")
                            .CompileTimeConstInput("block_shape")
                            .MlirCompileTimeFixedShapeInput("paddings")
                            .MlirCompileTimeFixedShapeInput("block_shape"));
REGISTER_XLA_OP_FOR_NPU(Name("SpaceToBatch")
                            .CompileTimeConstInput("paddings")
                            .MlirCompileTimeFixedShapeInput("paddings"));
REGISTER_XLA_OP_FOR_NPU(Name("BroadcastTo")
                            .CompileTimeConstInput("shape")
                            .MlirCompileTimeFixedShapeInput("shape"));
REGISTER_XLA_OP_FOR_NPU(Name("Cross"));
REGISTER_XLA_OP_FOR_NPU(Name("StridedSlice")
                            .CompileTimeConstInput("begin")
                            .CompileTimeConstInput("end")
                            .CompileTimeConstInput("strides")
                            .MlirCompileTimeFixedShapeInput("begin")
                            .MlirCompileTimeFixedShapeInput("end")
                            .MlirCompileTimeConstInput("strides"));
REGISTER_XLA_OP_FOR_NPU(Name("StridedSliceGrad")
                            .CompileTimeConstInput("shape")
                            .CompileTimeConstInput("begin")
                            .CompileTimeConstInput("end")
                            .CompileTimeConstInput("strides")
                            .MlirCompileTimeFixedShapeInput("shape")
                            .MlirCompileTimeFixedShapeInput("begin")
                            .MlirCompileTimeFixedShapeInput("end")
                            .MlirCompileTimeFixedShapeInput("strides"));
REGISTER_XLA_OP_FOR_NPU(Name("ResourceStridedSliceAssign")
                            .CompileTimeConstInput("begin")
                            .CompileTimeConstInput("end")
                            .CompileTimeConstInput("strides")
                            .MlirCompileTimeFixedShapeInput("begin")
                            .MlirCompileTimeFixedShapeInput("end")
                            .MlirCompileTimeFixedShapeInput("strides"));
REGISTER_XLA_OP_FOR_NPU(Name("ScatterNd")
                            .CompileTimeConstInput("shape")
                            .MlirCompileTimeFixedShapeInput("shape"));
REGISTER_XLA_OP_FOR_NPU(Name("Diag"));
REGISTER_XLA_OP_FOR_NPU(Name("DiagPart"));
REGISTER_XLA_OP_FOR_NPU(Name("MatrixDiag"));
REGISTER_XLA_OP_FOR_NPU(Name("MatrixDiagPart"));
REGISTER_XLA_OP_FOR_NPU(Name("Cholesky").TypeConstraint("T", kFloatTypes));
REGISTER_XLA_OP_FOR_NPU(Name("Transpose")
                            .CompileTimeConstInput("perm")
                            .MlirCompileTimeConstInput("perm"));
REGISTER_XLA_OP_FOR_NPU(Name("ConjugateTranspose")
                            .CompileTimeConstInput("perm")
                            .MlirCompileTimeConstInput("perm"));
REGISTER_XLA_OP_FOR_NPU(Name("InvertPermutation")
                            .TypeConstraint("T", DT_INT32)
                            .CompileTimeConstInput("x")
                            .MlirCompileTimeConstInput("x"));
REGISTER_XLA_OP_FOR_NPU(
    Name("ResourceApplyGradientDescent").TypeConstraint("T", kFloatTypes));
REGISTER_XLA_OP_FOR_NPU(Name("ResourceApplyProximalGradientDescent")
                            .TypeConstraint("T", kFloatTypes));
REGISTER_XLA_OP_FOR_NPU(
    Name("ResourceApplyMomentum").TypeConstraint("T", kFloatTypes));
REGISTER_XLA_OP_FOR_NPU(
    Name("ResourceApplyAdagrad").TypeConstraint("T", kFloatTypes));
REGISTER_XLA_OP_FOR_NPU(
    Name("ResourceApplyProximalAdagrad").TypeConstraint("T", kFloatTypes));
REGISTER_XLA_OP_FOR_NPU(
    Name("ResourceApplyAdagradDA").TypeConstraint("T", kFloatTypes));
REGISTER_XLA_OP_FOR_NPU(
    Name("ResourceApplyAdam").TypeConstraint("T", kFloatTypes));
REGISTER_XLA_OP_FOR_NPU(
    Name("ResourceApplyAdaMax").TypeConstraint("T", kFloatTypes));
REGISTER_XLA_OP_FOR_NPU(
    Name("ResourceApplyRMSProp").TypeConstraint("T", kFloatTypes));
REGISTER_XLA_OP_FOR_NPU(
    Name("ResourceApplyCenteredRMSProp").TypeConstraint("T", kFloatTypes));
REGISTER_XLA_OP_FOR_NPU(
    Name("ResourceApplyFtrl").TypeConstraint("T", kFloatTypes));
REGISTER_XLA_OP_FOR_NPU(
    Name("ResourceApplyFtrlV2").TypeConstraint("T", kFloatTypes));
REGISTER_XLA_OP_FOR_NPU(
    Name("ResourceApplyAdadelta").TypeConstraint("T", kFloatTypes));
REGISTER_XLA_OP_FOR_NPU(
    Name("ResourceApplyAddSign").TypeConstraint("T", kFloatTypes));
REGISTER_XLA_OP_FOR_NPU(
    Name("ResourceApplyPowerSign").TypeConstraint("T", kFloatTypes));
REGISTER_XLA_OP_FOR_NPU(Name("ReverseSequence"));
REGISTER_XLA_OP_FOR_NPU(Name("IsNan"));
REGISTER_XLA_OP_FOR_NPU(Name("Erf"));
REGISTER_XLA_OP_FOR_NPU(Name("Erfc"));
REGISTER_XLA_OP_FOR_NPU(Name("Lgamma"));
REGISTER_XLA_OP_FOR_NPU(Name("Digamma"));
REGISTER_XLA_OP_FOR_NPU(Name("Slice")
                            .CompileTimeConstInput("begin")
                            .CompileTimeConstInput("size")
                            .MlirCompileTimeFixedShapeInput("begin")
                            .MlirCompileTimeFixedShapeInput("size"));
REGISTER_XLA_OP_FOR_NPU(Name("Bucketize"));
REGISTER_XLA_OP_FOR_NPU(Name("BiasAdd"));
REGISTER_XLA_OP_FOR_NPU(Name("BiasAddV1"));
REGISTER_XLA_OP_FOR_NPU(Name("BiasAddGrad"));
REGISTER_XLA_OP_FOR_NPU(Name("FakeQuantWithMinMaxArgs"));
REGISTER_XLA_OP_FOR_NPU(Name("FakeQuantWithMinMaxArgsGradient"));
REGISTER_XLA_OP_FOR_NPU(Name("FakeQuantWithMinMaxVars"));
REGISTER_XLA_OP_FOR_NPU(Name("FakeQuantWithMinMaxVarsGradient"));
REGISTER_XLA_OP_FOR_NPU(Name("Const").CompilationOnly());
REGISTER_XLA_OP_FOR_NPU(Name("VarIsInitializedOp"));
REGISTER_XLA_OP_FOR_NPU(Name("VariableShape"));
REGISTER_XLA_OP_FOR_NPU(Name("ReadVariableOp").CompilationOnly());
REGISTER_XLA_OP_FOR_NPU(Name("AssignVariableOp").CompilationOnly());
REGISTER_XLA_OP_FOR_NPU(Name("AssignAddVariableOp").TypeConstraint("dtype", kNumericTypes));
REGISTER_XLA_OP_FOR_NPU(Name("AssignSubVariableOp").TypeConstraint("dtype", kNumericTypes));
REGISTER_XLA_OP_FOR_NPU(Name("ResourceGather"));
REGISTER_XLA_OP_FOR_NPU(Name("ResourceScatterAdd"));
REGISTER_XLA_OP_FOR_NPU(Name("ResourceScatterSub"));
REGISTER_XLA_OP_FOR_NPU(Name("ResourceScatterMul"));
REGISTER_XLA_OP_FOR_NPU(Name("ResourceScatterDiv"));
REGISTER_XLA_OP_FOR_NPU(Name("ResourceScatterMin"));
REGISTER_XLA_OP_FOR_NPU(Name("ResourceScatterMax"));
REGISTER_XLA_OP_FOR_NPU(Name("ResourceScatterUpdate"));
REGISTER_XLA_OP_FOR_NPU(Name("ResourceScatterNdUpdate"));
REGISTER_XLA_OP_FOR_NPU(Name("ResourceScatterNdAdd"));
REGISTER_XLA_OP_FOR_NPU(Name("Pack"));
REGISTER_XLA_OP_FOR_NPU(Name("DynamicStitch")
                            .CompileTimeConstInput("indices")
                            .MlirCompileTimeFixedShapeInput("indices"));
REGISTER_XLA_OP_FOR_NPU(Name("ParallelDynamicStitch")
                            .CompileTimeConstInput("indices")
                            .MlirCompileTimeFixedShapeInput("indices"));
REGISTER_XLA_OP_FOR_NPU(Name("Cast"));
REGISTER_XLA_OP_FOR_NPU(Name("Bitcast"));
REGISTER_XLA_OP_FOR_NPU(Name("MatrixTriangularSolve"));
REGISTER_XLA_OP_FOR_NPU(Name("Gather"));
REGISTER_XLA_OP_FOR_NPU(Name("GatherV2")
                            .CompileTimeConstInput("axis")
                            .MlirCompileTimeConstInput("axis"));
REGISTER_XLA_OP_FOR_NPU(Name("GatherNd"));
REGISTER_XLA_OP_FOR_NPU(Name("Relu"));
REGISTER_XLA_OP_FOR_NPU(Name("Relu6"));
REGISTER_XLA_OP_FOR_NPU(Name("LeakyRelu"));
REGISTER_XLA_OP_FOR_NPU(Name("LeakyReluGrad"));
REGISTER_XLA_OP_FOR_NPU(Name("ReluGrad"));
REGISTER_XLA_OP_FOR_NPU(Name("Relu6Grad"));
REGISTER_XLA_OP_FOR_NPU(Name("ArgMax")
                            .TypeConstraint("T", DT_FLOAT)
                            .Device(DEVICE_NPU_XLA_JIT)
                            .CompileTimeConstInput("dimension")
                            .MlirCompileTimeConstInput("dimension"));
REGISTER_XLA_OP_FOR_NPU(Name("_ListToArray"));
REGISTER_XLA_OP_FOR_NPU(Name("_ArrayToList"));
REGISTER_XLA_OP_FOR_NPU(Name("SymbolicGradient"));
REGISTER_XLA_OP_FOR_NPU(Name("StackV2")
                            .CompileTimeConstInput("max_size")
                            .MlirCompileTimeConstInput("max_size"));
REGISTER_XLA_OP_FOR_NPU(Name("StackPushV2"));
REGISTER_XLA_OP_FOR_NPU(Name("StackPopV2"));
REGISTER_XLA_OP_FOR_NPU(Name("StackCloseV2"));
REGISTER_XLA_OP_FOR_NPU(Name("DepthToSpace"));
REGISTER_XLA_OP_FOR_NPU(Name("OneHot")
                            .CompileTimeConstInput("depth")
                            .MlirCompileTimeFixedShapeInput("depth"));
REGISTER_XLA_OP_FOR_NPU(Name("BroadcastArgs")
                            .CompileTimeConstInput("s0")
                            .CompileTimeConstInput("s1")
                            .MlirCompileTimeFixedShapeInput("s0")
                            .MlirCompileTimeFixedShapeInput("s1"));
REGISTER_XLA_OP_FOR_NPU(Name("BroadcastGradientArgs")
                            .CompileTimeConstInput("s0")
                            .CompileTimeConstInput("s1")
                            .MlirCompileTimeFixedShapeInput("s0")
                            .MlirCompileTimeFixedShapeInput("s1"));
REGISTER_XLA_OP_FOR_NPU(Name("ApproximateEqual"));
REGISTER_XLA_OP_FOR_NPU(Name("ResizeBilinear")
                            .CompileTimeConstInput("size")
                            .MlirCompileTimeFixedShapeInput("size"));
REGISTER_XLA_OP_FOR_NPU(Name("ResizeBilinearGrad"));
REGISTER_XLA_OP_FOR_NPU(Name("L2Loss"));
REGISTER_XLA_OP_FOR_NPU(Name("Unpack"));
REGISTER_XLA_OP_FOR_NPU(Name("ArgMin")
                            .CompileTimeConstInput("dimension")
                            .MlirCompileTimeConstInput("dimension"));
REGISTER_XLA_OP_FOR_NPU(Name("LRN"));
REGISTER_XLA_OP_FOR_NPU(Name("LRNGrad"));
REGISTER_XLA_OP_FOR_NPU(Name("TopKV2")
                            .CompileTimeConstInput("k")
                            .MlirCompileTimeFixedShapeInput("k")
                            .TypeConstraint("T", {DT_UINT32, DT_INT32, DT_FLOAT, DT_BFLOAT16}));
REGISTER_XLA_OP_FOR_NPU(Name("SparseToDense")
                            .CompileTimeConstInput("output_shape")
                            .MlirCompileTimeFixedShapeInput("output_shape"));
REGISTER_XLA_OP_FOR_NPU(Name("Select"));
REGISTER_XLA_OP_FOR_NPU(Name("Qr").TypeConstraint("T", kFloatTypes));


// binary
REGISTER_XLA_OP_FOR_NPU(Name("Add"));
REGISTER_XLA_OP_FOR_NPU(Name("Sub"));
REGISTER_XLA_OP_FOR_NPU(Name("Mul"));
REGISTER_XLA_OP_FOR_NPU(Name("Div"));
REGISTER_XLA_OP_FOR_NPU(Name("Atan2"));
REGISTER_XLA_OP_FOR_NPU(Name("Complex"));
REGISTER_XLA_OP_FOR_NPU(Name("DivNoNan"));
REGISTER_XLA_OP_FOR_NPU(Name("FloorDiv"));
REGISTER_XLA_OP_FOR_NPU(Name("Xlogy"));
REGISTER_XLA_OP_FOR_NPU(Name("Xdivy"));
REGISTER_XLA_OP_FOR_NPU(Name("FloorMod"));
REGISTER_XLA_OP_FOR_NPU(Name("BitwiseAnd"));
REGISTER_XLA_OP_FOR_NPU(Name("BitwiseOr"));
REGISTER_XLA_OP_FOR_NPU(Name("BitwiseXor"));
REGISTER_XLA_OP_FOR_NPU(Name("LeftShift"));
REGISTER_XLA_OP_FOR_NPU(Name("RightShift"));
REGISTER_XLA_OP_FOR_NPU(Name("LogicalAnd"));
REGISTER_XLA_OP_FOR_NPU(Name("LogicalOr"));
REGISTER_XLA_OP_FOR_NPU(Name("Mod"));
REGISTER_XLA_OP_FOR_NPU(Name("Maximum"));
REGISTER_XLA_OP_FOR_NPU(Name("Minimum"));
REGISTER_XLA_OP_FOR_NPU(Name("RealDiv"));
REGISTER_XLA_OP_FOR_NPU(Name("ReciprocalGrad"));
REGISTER_XLA_OP_FOR_NPU(Name("RsqrtGrad"));
REGISTER_XLA_OP_FOR_NPU(Name("SqrtGrad"));
REGISTER_XLA_OP_FOR_NPU(Name("SquaredDifference"));
REGISTER_XLA_OP_FOR_NPU(Name("TruncateDiv"));
REGISTER_XLA_OP_FOR_NPU(Name("TruncateMod"));
REGISTER_XLA_OP_FOR_NPU(Name("Equal"));
REGISTER_XLA_OP_FOR_NPU(Name("NotEqual"));
REGISTER_XLA_OP_FOR_NPU(Name("Greater"));
REGISTER_XLA_OP_FOR_NPU(Name("GreaterEqual"));
REGISTER_XLA_OP_FOR_NPU(Name("Less"));
REGISTER_XLA_OP_FOR_NPU(Name("LessEqual"));
REGISTER_XLA_OP_FOR_NPU(Name("SigmoidGrad"));
REGISTER_XLA_OP_FOR_NPU(Name("SoftplusGrad"));
REGISTER_XLA_OP_FOR_NPU(Name("SoftsignGrad"));
REGISTER_XLA_OP_FOR_NPU(Name("TanhGrad"));
REGISTER_XLA_OP_FOR_NPU(Name("Pow"));

// unary
REGISTER_XLA_OP_FOR_NPU(Name("ComplexAbs"));
REGISTER_XLA_OP_FOR_NPU(Name("Angle"));
REGISTER_XLA_OP_FOR_NPU(Name("Conj"));
REGISTER_XLA_OP_FOR_NPU(Name("Abs"));
REGISTER_XLA_OP_FOR_NPU(Name("Acos"));
REGISTER_XLA_OP_FOR_NPU(Name("Acosh"));
REGISTER_XLA_OP_FOR_NPU(Name("Asin"));
REGISTER_XLA_OP_FOR_NPU(Name("Asinh"));
REGISTER_XLA_OP_FOR_NPU(Name("Atan"));
REGISTER_XLA_OP_FOR_NPU(Name("Atanh"));
REGISTER_XLA_OP_FOR_NPU(Name("Ceil"));
REGISTER_XLA_OP_FOR_NPU(Name("Cos"));
REGISTER_XLA_OP_FOR_NPU(Name("Cosh"));
REGISTER_XLA_OP_FOR_NPU(Name("Sin"));
REGISTER_XLA_OP_FOR_NPU(Name("Exp"));
REGISTER_XLA_OP_FOR_NPU(Name("Expm1"));
REGISTER_XLA_OP_FOR_NPU(Name("Floor"));
REGISTER_XLA_OP_FOR_NPU(Name("IsFinite"));
REGISTER_XLA_OP_FOR_NPU(Name("IsInf"));
REGISTER_XLA_OP_FOR_NPU(Name("Inv"));
REGISTER_XLA_OP_FOR_NPU(Name("Reciprocal"));
REGISTER_XLA_OP_FOR_NPU(Name("Log"));
REGISTER_XLA_OP_FOR_NPU(Name("Log1p"));
REGISTER_XLA_OP_FOR_NPU(Name("Invert"));
REGISTER_XLA_OP_FOR_NPU(Name("LogicalNot"));
REGISTER_XLA_OP_FOR_NPU(Name("Neg"));
REGISTER_XLA_OP_FOR_NPU(Name("Rint"));
REGISTER_XLA_OP_FOR_NPU(Name("Round"));
REGISTER_XLA_OP_FOR_NPU(Name("Rsqrt"));
REGISTER_XLA_OP_FOR_NPU(Name("Sigmoid"));
REGISTER_XLA_OP_FOR_NPU(Name("Sign"));
REGISTER_XLA_OP_FOR_NPU(Name("Sinh"));
REGISTER_XLA_OP_FOR_NPU(Name("Softplus"));
REGISTER_XLA_OP_FOR_NPU(Name("Softsign"));
REGISTER_XLA_OP_FOR_NPU(Name("Sqrt"));
REGISTER_XLA_OP_FOR_NPU(Name("Square"));
REGISTER_XLA_OP_FOR_NPU(Name("Tan"));
REGISTER_XLA_OP_FOR_NPU(Name("Tanh"));
REGISTER_XLA_OP_FOR_NPU(Name("Real"));
REGISTER_XLA_OP_FOR_NPU(Name("Imag"));

REGISTER_XLA_OP_FOR_NPU(Name("AddV2"));
REGISTER_XLA_OP_FOR_NPU(Name("BatchMatMulV2"));
REGISTER_XLA_OP_FOR_NPU(Name("FusedBatchNormV3"));
REGISTER_XLA_OP_FOR_NPU(Name("FusedBatchNormGradV3"));
REGISTER_XLA_OP_FOR_NPU(Name("SelectV2"));

REGISTER_XLA_OP_FOR_NPU(Name("QuantizeV2"));
REGISTER_XLA_OP_FOR_NPU(Name("Dequantize"));
REGISTER_XLA_OP_FOR_NPU(Name("QuantizedConv2DWithBiasAndRequantize"));
REGISTER_XLA_OP_FOR_NPU(Name("SparseReshape"));
REGISTER_XLA_OP_FOR_NPU(Name("SparseFillEmptyRows"));
REGISTER_XLA_OP_FOR_NPU(Name("SparseSegmentMean"));
REGISTER_XLA_OP_FOR_NPU(Name("SparseSegmentSum"));
REGISTER_XLA_OP_FOR_NPU(Name("Where"));

// clang-format on

}  // namespace npu_xla
}  // namespace tensorflow
