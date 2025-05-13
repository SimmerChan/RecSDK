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

#include "tf_mlir/mlir_converter.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringExtras.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/Timing.h"
#include "mlir/Transforms/Passes.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/bridge.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/import_model.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/error_util.h"
#include "tensorflow/compiler/mlir/tf2xla/transforms/passes.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/common_runtime/graph_constructor.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/public/version.h"
#include "tf_mlir/mlir/npu_hlo/npu_hlo_utils.h"
#include "tf_mlir/mlir/npu_hlo/transforms/passes.h"
#include "tf_mlir/mlir_converter.h"
#include "xla/mlir_hlo/mhlo/transforms/passes.h"
#include "xla/mlir_hlo/transforms/passes.h"

namespace tensorflow {
namespace npu_xla {

using llvm::SmallVector;
using mlir::DenseElementsAttr;
using mlir::RankedTensorType;
using mlir::func::FuncOp;

absl::StatusOr<std::vector<TensorShape>> ParseArgShapes(const CompilerInput& input)
{
    std::vector<TensorShape> args;
    args.reserve(input.args_size());
    for (int i = 0; i < input.args_size(); ++i) {
        auto& arg = input.args(i);
        tensorflow::TensorShapeProto tensor_shape_proto;
        if (!tensor_shape_proto.ParseFromString(arg.shape())) {
            return errors::Internal("Parse failed for arg shape");
        }
        args.push_back(tensorflow::TensorShape(tensor_shape_proto));
        for (int j = 0; j < args.back().dims(); ++j) {
            VLOG(0) << "arg #" << i << " dim #" << j << ": " << args.back().dim_size(j);
        }
    }
    return args;
}

absl::StatusOr<std::unordered_map<std::string, PartialTensorShape>> ParseKnownArgShapes(const CompilerInput& input)
{
    std::unordered_map<std::string, PartialTensorShape> arg_shapes;
    FunctionDefLibrary input_flib_def;
    if (!input_flib_def.ParseFromString(input.options().flib_def())) {
        return tensorflow::errors::Internal("Parse failed for input flib_def");
    }
    auto func_def = input_flib_def.function(0);
    if (func_def.attr().count("input_args_info") == 0) {
        return arg_shapes;
    }
    auto attr_value = func_def.attr().at("input_args_info");
    if (!attr_value.has_func()) {
        return arg_shapes;
    }
    auto name_attr_list = attr_value.func();
    if (name_attr_list.name() != "shape_info") {
        return arg_shapes;
    }
    for (auto iter : name_attr_list.attr()) {
        arg_shapes[iter.first] = PartialTensorShape(iter.second.shape());
        VLOG(2) << "KnownArgShapes: " << iter.first << ", " << arg_shapes[iter.first];
    }
    return arg_shapes;
}

Status ConvertInputInfo(const CompilerInput& input, Graph* graph, GraphImportConfig* specs)
{
    std::vector<std::string> array_names;
    std::vector<std::string> data_types;
    std::vector<std::vector<int>> shapes;

    TF_ASSIGN_OR_RETURN(auto arg_shapes, ParseArgShapes(input));
    TF_ASSIGN_OR_RETURN(auto known_arg_shapes, ParseKnownArgShapes(input));

    for (Node* n : graph->op_nodes()) {
        VLOG(2) << "ConvertInputInfo: " << n->type_string() << "@" << n->name();
        if (n->type_string() == "_Arg") {
            int index;
            TF_RETURN_IF_ERROR(GetNodeAttr(n->attrs(), "index", &index));
            VLOG(2) << "_Arg " << index << ", " << n->name();
            if (array_names.size() <= index) {
                array_names.resize(index + 1);
                data_types.resize(index + 1);
                shapes.resize(index + 1);
            }
            array_names[index] = n->name();
            DataType dtype;
            TF_RETURN_IF_ERROR(GetNodeAttr(n->attrs(), "T", &dtype));
            data_types[index] = (dtype == DT_INVALID ? "" : DataType_Name(dtype));
            // Force to codegen dynamic shape (not dynamic rank) code
            std::vector<int> dims(arg_shapes[index].dims(), -1);
            if (known_arg_shapes.count(absl::AsciiStrToLower(n->name())) > 0) {
                auto known_shape = known_arg_shapes[absl::AsciiStrToLower(n->name())];
                for (auto i = 0; i < dims.size(); i++) {
                    dims[i] = known_shape.dim_size(i);
                }
            }
            shapes[index] = std::move(dims);
            for (int i = 0; i < shapes[index].size(); ++i) {
                VLOG(2) << "input #" << index << " dim #" << i << ": " << shapes[index][i];
            }
        }
    }
    std::vector<std::optional<std::vector<int>>> optional_shapes;
    for (auto& shape : shapes)
        optional_shapes.emplace_back(shape);
    return ParseInputArrayInfo(array_names, data_types, optional_shapes, &specs->inputs);
}

Status ConvertOutputInfo(Graph* graph, GraphImportConfig* specs)
{
    std::vector<std::string> array_names;
    for (Node* n : graph->op_nodes()) {
        if (n->type_string() == "_Retval") {
            int index;
            TF_RETURN_IF_ERROR(GetNodeAttr(n->attrs(), "index", &index));
            const Edge* e = nullptr;
            TF_RETURN_IF_ERROR(n->input_edge(0, &e));
            if (array_names.size() <= index) {
                array_names.resize(index + 1);
            }
            array_names[index] = absl::StrCat(e->src()->name(), ":", e->src_output());
        }
    }
    return ParseOutputArrayInfo(array_names, &specs->outputs);
}

mlir::Type DataTypeToMlirType(mlir::OpBuilder b, DataType dtype)
{
    if (dtype == DataType::DT_FLOAT) {
        return b.getF32Type();
    } else if (dtype == DataType::DT_DOUBLE) {
        return b.getF64Type();
    } else if (dtype == DataType::DT_HALF) {
        return b.getF16Type();
    } else if (dtype == DataType::DT_INT64) {
        return b.getIntegerType(64);
    } else if (dtype == DataType::DT_INT32) {
        return b.getIntegerType(32);
    } else if (dtype == DataType::DT_BOOL) {
        return b.getIntegerType(1);
    } else {
        LOG(FATAL) << "Unimplemented DataTypeToMlirType conversion";
    }
}

Status AppendIOAttr(mlir::ModuleOp module, const GraphImportConfig& specs, const CompilerInput& input,
                    const std::string& default_device)
{
    auto main_func = module.lookupSymbol<mlir::func::FuncOp>("main");
    auto dict_attr = main_func->getAttrOfType<mlir::DictionaryAttr>("tf.entry_function");
    if (!dict_attr) {
        return errors::Internal("main_func must has tf.entry_function attr");
    }
    SmallVector<mlir::NamedAttribute, 2> attributes;
    for (auto attr : dict_attr) {
        attributes.push_back(attr);
    }
    mlir::OpBuilder builder(module);
    SmallVector<mlir::StringRef, 4> input_placements;
    SmallVector<mlir::StringRef, 4> output_placements;

    for (int i = 0; i < specs.inputs.size(); ++i) {
        auto& arg_proto = input.args(i);
        if (arg_proto.kind_v2() == ArgumentKind::kConstant) {
            // compile_time_const
            input_placements.push_back("const");
        } else if (arg_proto.kind_v2() == ArgumentKind::kFixedShaped) {
            input_placements.push_back("cpu");
        } else if (arg_proto.kind_v2() == ArgumentKind::kHostArgs) {
            input_placements.push_back("cpu");
        } else {
            input_placements.push_back(default_device);
        }
    }
    for (int i = 0; i < specs.outputs.size(); ++i) {
        if (input.options().output_placements_size() > i) {
            output_placements.push_back(input.options().output_placements(i));
        } else {
            output_placements.push_back(default_device);
        }
    }
    attributes.push_back(
        builder.getNamedAttr("input_placements", builder.getStringAttr(llvm::join(input_placements, ","))));
    attributes.push_back(
        builder.getNamedAttr("output_placements", builder.getStringAttr(llvm::join(output_placements, ","))));

    // extract const inputs info
    for (int i = 0; i < specs.inputs.size(); ++i) {
        auto& arg_proto = input.args(i);
        if (arg_proto.kind_v2() == ArgumentKind::kConstant) {
            auto attr_name = (mlir::npu_hlo::kHloInputValueAttr + ("_" + llvm::Twine(i))).str();
            TensorProto tensor_proto;
            if (!tensor_proto.ParseFromString(arg_proto.constant_value())) {
                return tensorflow::errors::Internal("Mlir parse failed for arg constant value");
            }
            Tensor constant_tensor;
            CHECK(constant_tensor.FromProto(tensor_proto));
            DenseElementsAttr attr;
            auto elem_type = DataTypeToMlirType(builder, DataType(arg_proto.type()));
            SmallVector<int64_t, 4> shape;
            for (int dim = 0; dim < constant_tensor.dims(); ++dim) {
                shape.push_back(constant_tensor.dim_size(dim));
            }
            auto shaped_type = mlir::RankedTensorType::get(shape, elem_type);
            if (arg_proto.type() == DataType::DT_FLOAT) {
                VLOG(0) << "Warning: usually there shouldn't be float const inputs";
                auto data = constant_tensor.flat<float>();
                attr = DenseElementsAttr::get(shaped_type, llvm::ArrayRef(data.data(), data.size()));
            } else if (arg_proto.type() == DataType::DT_INT64) {
                auto data = constant_tensor.flat<int64>();
                attr = DenseElementsAttr::get(shaped_type, llvm::ArrayRef(data.data(), data.size()));
            } else if (arg_proto.type() == DataType::DT_INT32) {
                auto data = constant_tensor.flat<int32>();
                attr = DenseElementsAttr::get(shaped_type, llvm::ArrayRef(data.data(), data.size()));
            } else if (arg_proto.type() == DataType::DT_BOOL) {
                auto data = constant_tensor.flat<bool>();
                attr = DenseElementsAttr::get(shaped_type, llvm::ArrayRef(data.data(), data.size()));
            } else {
                return tensorflow::errors::Internal("Mlir datatype not implemented for constant input");
            }
            attributes.push_back(builder.getNamedAttr(attr_name, attr));
        } else if (arg_proto.kind_v2() == ArgumentKind::kFixedShaped) {
            auto attr_name = (mlir::npu_hlo::kHloInputShapeAttr + ("_" + llvm::Twine(i))).str();
            SmallVector<int64_t, 4> input_shape;
            tensorflow::TensorShapeProto shape_proto;
            shape_proto.ParseFromString(arg_proto.shape());
            for (int dim = 0; dim < shape_proto.dim_size(); ++dim) {
                input_shape.push_back(shape_proto.dim(dim).size());
            }
            auto elem_tp = DataTypeToMlirType(builder, DataType(arg_proto.type()));
            auto type = RankedTensorType::get(input_shape, elem_tp);
            // a DenseElementsAttr with Splat zero value is to represent the
            // shape/dtype
            mlir::Attribute attr;
            if (elem_tp.isSignlessInteger()) {
                attr = DenseElementsAttr::get(type, mlir::IntegerAttr::get(elem_tp, 0));
            } else {
                attr = DenseElementsAttr::get(type, mlir::FloatAttr::get(elem_tp, 0));
            }
            attributes.push_back(builder.getNamedAttr(attr_name, attr));
        }
    }

    main_func->setAttr("tf.entry_function", builder.getDictionaryAttr(attributes));
    return absl::OkStatus();
}

MlirConverter::MlirConverter() {}
MlirConverter::~MlirConverter() {}

Status MlirConverter::ConvertGraphdefToStablehlo(const CompilerInput& compiler_input, const std::string& output_path)
{
    TF_RETURN_IF_ERROR(ImportGraphDef(compiler_input));
    TF_RETURN_IF_ERROR(ConvertTfToStablehlo());
    std::error_code EC;
    llvm::raw_fd_ostream os(output_path, EC);
    if (EC) {
        return errors::Internal("Failed to open file: %s", output_path);
    }
    module_->print(os);
    os.close();
    VLOG(1) << "Write stablehlo to file: " << output_path;
    return absl::OkStatus();
}

Status MlirConverter::ImportGraphDef(const CompilerInput& input)
{
    NameAttrList fn_name_attrs;
    if (!fn_name_attrs.ParseFromString(input.function())) {
        return errors::Internal("Parse failed for input function");
    }

    FunctionDefLibrary input_flib_def;
    if (!input_flib_def.ParseFromString(input.options().flib_def())) {
        return errors::Internal("Parse failed for input flib_def");
    }

    auto flib_def = absl::make_unique<FunctionLibraryDefinition>(OpRegistry::Global(), input_flib_def);

    OptimizerOptions opts;
    std::unique_ptr<ProcessFunctionLibraryRuntime> pflr(new ProcessFunctionLibraryRuntime(
        nullptr, Env::Default(), nullptr, TF_GRAPH_DEF_VERSION, flib_def.get(), opts));
    FunctionLibraryRuntime* lib_runtime = pflr->GetFLR(ProcessFunctionLibraryRuntime::kDefaultFLRDevice);

    FunctionLibraryRuntime::Handle func_handle;
    tensorflow::FunctionLibraryRuntime::InstantiateOptions inst_ops;
    TF_RETURN_IF_ERROR(
        lib_runtime->Instantiate(fn_name_attrs.name(), AttrSlice(&fn_name_attrs.attr()), inst_ops, &func_handle));

    const FunctionBody* fbody = lib_runtime->GetFunctionBody(func_handle);
    std::unique_ptr<Graph> graph(new Graph(lib_runtime->GetFunctionLibraryDefinition()));
    CopyGraph(*fbody->graph, graph.get());
    GraphDef graph_def;
    graph->ToGraphDef(&graph_def);
    *graph_def.mutable_library() = lib_runtime->GetFunctionLibraryDefinition()->ToProto();

    std::string graph_def_str = graph_def.SerializeAsString();

    mlir::DialectRegistry registry;
    // mlir::registerAllDialects(registry);
    context_.reset(new mlir::MLIRContext(registry));
    auto& context = *context_;
    GraphDebugInfo debug_info;
    GraphImportConfig specs;
    specs.prune_unused_nodes = false;
    specs.convert_legacy_fed_inputs = false;
    specs.graph_as_function = false;
    specs.upgrade_legacy = true;

    TF_RETURN_IF_ERROR(ConvertInputInfo(input, graph.get(), &specs));
    TF_RETURN_IF_ERROR(ConvertOutputInfo(graph.get(), &specs));

    VLOG(2) << "Input size = " << specs.inputs.size() << ", Output size = " << specs.outputs.size();

    TF_ASSIGN_OR_RETURN(auto module, ConvertGraphdefToMlir(graph_def, debug_info, specs, &context));
    module_ = std::move(module);
    TF_RETURN_IF_ERROR(ConvertTfExecutorToTf());
    TF_RETURN_IF_ERROR(AppendIOAttr(*module_, specs, input, "npu"));
    return absl::OkStatus();
}

Status MlirConverter::ConvertTfExecutorToTf()
{
    TF_RETURN_IF_ERROR(mlir::TF::RunBridgeWithStandardPipeline(*module_, VLOG_IS_ON(1), true));
    return absl::OkStatus();
}

Status MlirConverter::ConvertTfToStablehlo()
{
    auto module_op = *module_;
    mlir::DefaultTimingManager tm;
    mlir::applyDefaultTimingManagerCLOptions(tm);
    // Records elapsed time for each pass in the passpipe
    tm.setEnabled(true);
    mlir::TimingScope timing = tm.getRootScope();

    mlir::PassManager pm(module_op.getContext());
    (void)mlir::applyPassManagerCLOptions(pm);
    pm.enableTiming(timing);
    pm.getContext()->disableMultithreading();
    auto printingFlags = mlir::OpPrintingFlags();
    printingFlags.elideLargeElementsAttrs(16);
    printingFlags.elideLargeResourceString(16);
    pm.enableIRPrinting(
        nullptr, [](mlir::Pass* pass, mlir::Operation*) { return VLOG_IS_ON(2); }, false, true, false, llvm::dbgs(),
        printingFlags);
    bool prefer_tf2xla = false;
    llvm::StringRef device_type = "XLA_CPU_JIT";

    // Replace const arguments to ConstOp and update argument type if it is a
    // fixed-shaped input
    pm.addPass(mlir::npu_hlo::createReviseArgsForStaticRankPass());

    // Note that the region-based control-flow produced here still contains
    // function call ops which get inlined by the subsequent inliner pass.
    pm.addPass(mlir::TF::CreateTFFunctionalControlFlowToRegions());
    pm.addPass(mlir::createInlinerPass());
    pm.addNestedPass<mlir::func::FuncOp>(mlir::TF::CreateDropWhileShapeInvariantPass());
    // Create a replicated TensorList initialization ops for all of its uses. This
    // pass undo some CSE because shape_inference is not correctly able to
    // identify the shapes of TensorList initialization ops.
    // This pass requires CanonicalizerPass before
    // CreateTensorListOpsDecompositionPass for clean-ups.
    pm.addNestedPass<mlir::func::FuncOp>(mlir::TF::CreateReplicateTensorListInitOpsPass());
    pm.addNestedPass<mlir::func::FuncOp>(mlir::createCanonicalizerPass());
    // The SCCP pass performs constant propagation across the IR, which, for
    // example, propagates constant arguments into callee functions.
    // TOOD(hinsu): Investigate if we really need SCCP pass before shape inference
    // and can do with just one pass after the shape inference.
    pm.addPass(mlir::createSCCPPass());
    // Guarantee all functions have one use, which enables shape inference.
    pm.addPass(mlir::TF::CreateGuaranteeAllFuncsOneUsePass());
    // Run shape inference pass before tensorlist decomposition to get buffer
    // shape of uninitialized TensorLists.
    pm.addPass(mlir::TF::CreateTFShapeInferencePass());

    // Run SCCP pass again as the availability of shapes may open up new
    // opportunities for constant propagation. Note that the shape inference pass
    // doesn't materialize new constants even if those are computed internally for
    // the purpose of shape inference. These constants might be required by the
    // legalization passes.
    pm.addPass(mlir::createSCCPPass());

    pm.addPass(mlir::TF::CreateTensorListOpsDecompositionPass());
    pm.addPass(mlir::TF::CreateStackOpsDecompositionPass());
    pm.addPass(mlir::TF::CreateTensorArrayOpsDecompositionPass());
    pm.addNestedPass<mlir::func::FuncOp>(mlir::TFDevice::CreateDecomposeResourceOpsPass());
    pm.addPass(mlir::TF::CreatePromoteResourcesToArgsPass());
    pm.addPass(mlir::createSymbolDCEPass());

    // Sink constants to regions so that ops requiring constant operands can
    // access the constant and there is no indirection through control flow region
    // arguments. Also, note that this pass is in MHLO but it is generic and sinks
    // constants for all ops with regions.
    pm.addNestedPass<mlir::func::FuncOp>(mlir::mhlo::createSinkConstantsToControlFlowPass());
    pm.addNestedPass<mlir::func::FuncOp>(mlir::createCanonicalizerPass());
    pm.addPass(mlir::TF::CreateTFShapeInferencePass());
    pm.addPass(mlir::createSCCPPass());

    // Legalize any StableHLO ops to MHLO. Bridge still doesn't use StableHLO but
    // such ops might be present in the input from upstream like TFRT compilation.
    // Later on, this could be merged in the legalization pass when we migrate
    // bridge to StableHLO.

    pm.addNestedPass<mlir::func::FuncOp>(mlir::TF::CreateLowerQuantizedPass());
    pm.addPass(mlir::mhlo::createLegalizeTFPass(
        /*legalize_chlo=*/true,
        /*tf2xla_fallback_device_type=*/device_type, prefer_tf2xla));

    pm.addNestedPass<mlir::func::FuncOp>(mlir::mhlo::CreateInfeedsOpsXlaAdjustLayoutPass());
    pm.addPass(mlir::mhlo::CreateLegalizeTFCollectivePass());
    // This has to run after legalization to delete non legal but dead ops.
    // This must run before Shape Inference.
    pm.addNestedPass<mlir::func::FuncOp>(mlir::createCanonicalizerPass());
    // Run shape inference pass to propagate shapes through tensor_cast operations
    // from static to dynamic shapes. This could be generated if the shape
    // inference was originally missing in a TF op but the corresponding HLO op
    // had static shape after lowering.
    pm.addPass(mlir::TF::CreateTFShapeInferencePass());
    pm.addPass(mlir::createSCCPPass());
    // Run LegalizeTFPass again because the previous legalization passes can
    // expose more graph pruning and canonicalization opportunities that are
    // necessary for the second LegalizeTFPass(allow_partial_conversion=false)
    // invocation.
    // TODO(wyzero): we can not set `allow_partial_conversion = false` since
    // `createLegalizeTFPass` pass does not known ops from hlo_disc dialect.
    pm.addPass(mlir::mhlo::createLegalizeTFPass(true, device_type, prefer_tf2xla));

    pm.addPass(mlir::npu_hlo::createTFToMHLOLegalizationPass());

    // This pass operates on MHLO control flow ops so it should be legalized after
    // the control flow ops are legalized.
    pm.addPass(mlir::mhlo::CreateLegalizeTFCommunicationPass());

    // In order to export to XLA, we must sink constants to control flow regions,
    // since XLA uses functional control flow.
    pm.addNestedPass<mlir::func::FuncOp>(mlir::mhlo::createSinkConstantsToControlFlowPass());

    // convert mhlo to stablehlo
    pm.addPass(mlir::mhlo::createExpandHloTuplesPass("main"));
    pm.addNestedPass<mlir::func::FuncOp>(mlir::mhlo::createFlattenTuplePass());
    pm.addNestedPass<mlir::func::FuncOp>(mlir::mhlo::createLegalizeTorchIndexSelectToGatherPass());
    pm.addNestedPass<mlir::func::FuncOp>(mlir::mhlo::createLegalizeGeneralDotPass());
    pm.addNestedPass<mlir::func::FuncOp>(mlir::mhlo::createBroadcastPropagationPass());
    pm.addPass(mlir::createCSEPass());
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addNestedPass<FuncOp>(mlir::mhlo::createLegalizeDotGeneralToDotPass());
    pm.addNestedPass<FuncOp>(mlir::mhlo::createHloCanonicalizeDotPass());
    pm.addPass(mlir::mhlo::createHloLegalizeToStablehloPass());

    pm.addPass(mlir::createCSEPass());
    pm.addPass(mlir::createCanonicalizerPass());

    // Make sure we catch any error reported by MLIR and forward it to the TF
    // error reporting system. Report a generic error if pass manager failed
    // without emitting a diagnostic.
    mlir::StatusScopedDiagnosticHandler error_handler(module_op.getContext());

    if (failed(pm.run(module_op))) {
        return error_handler.Combine(errors::Internal("MLIR TF to StableHLO legalization failed"));
    }

    return absl::OkStatus();
}

}  // namespace npu_xla
}  // namespace tensorflow