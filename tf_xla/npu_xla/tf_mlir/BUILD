load("@com_google_protobuf//:protobuf.bzl", "cc_proto_library")
load("@org_tensorflow//tensorflow:tensorflow.bzl", "tf_cc_binary")

package(default_visibility = ["//visibility:public"])

cc_proto_library(
    name = "compiler_input",
    srcs = [
        "compiler_input.proto",
    ],
)

cc_library(
    name = "tf_mlir",
    srcs = [
            "mlir_converter.cpp",
            ":compiler_input_genproto",
    ],
    hdrs = [
            "mlir_converter.h"
    ],
    deps = [
            "//tf_mlir/mlir/npu_hlo:revise_args_for_static_rank",
            "//tf_mlir/mlir/npu_hlo:tf_to_mhlo",
            "@llvm-project//llvm:Support",
            "@llvm-project//mlir:FuncDialect",
            "@llvm-project//mlir:FuncTransforms",
            "@llvm-project//mlir:IR",
            "@llvm-project//mlir:Pass",
            "@llvm-project//mlir:Support",
            "@org_tensorflow//tensorflow/cc:cc_ops",
            "@org_tensorflow//tensorflow/cc:ops",
            "@org_tensorflow//tensorflow/cc:resource_variable_ops",
            "@org_tensorflow//tensorflow/core:framework",
            "@org_tensorflow//tensorflow/core:lib",
            "@org_tensorflow//tensorflow/compiler/mlir/tensorflow/transforms:bridge",
            "@org_tensorflow//tensorflow/compiler/mlir/tensorflow/transforms:tensorflow_passes",
            "@org_tensorflow//tensorflow/compiler/mlir/tensorflow/translate:import_model",
            "@org_tensorflow//tensorflow/compiler/mlir/tensorflow:import_utils",
            "@org_tensorflow//tensorflow/compiler/mlir/tf2xla/transforms:tf_xla_passes",
            "@local_xla//xla/mlir_hlo:all_passes",
            "@local_xla//xla/mlir_hlo:mhlo_passes",
            "@local_xla//xla/mlir_hlo:transforms_passes",
        ],
    alwayslink = 1,
)

tf_cc_binary(
    name = "tf-mlir-opt",
    srcs = ["main.cpp"],
    # linkopts = ["-fno-lto -lutil -export-dynamic"],
    linkopts = ["-lutil"],
    linkstatic = 1,
    deps = [
        ":tf_mlir",
        "@com_google_absl//absl/strings",
    ],
)