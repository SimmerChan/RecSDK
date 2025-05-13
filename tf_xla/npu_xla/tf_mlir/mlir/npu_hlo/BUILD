load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library")
load("@org_tensorflow//tensorflow:tensorflow.default.bzl", "get_compatible_with_portable")


package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

gentbl_cc_library(
    name = "NpuHloPassIncGen",
    compatible_with = get_compatible_with_portable(),
    strip_include_prefix = "",
    tbl_outs = [
        (
            [
                "-gen-pass-decls",
                "-name=NPU_HLO",
            ],
            "transforms/passes.h.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "transforms/passes.td",
    deps = [
        "@llvm-project//mlir:PassBaseTdFiles",
    ],
)

cc_library(
    name = "revise_args_for_static_rank",
    srcs = [
        "transforms/revise_args_for_static_rank.cpp",
    ],
    hdrs = ["transforms/passes.h"],
    deps = [
        ":NpuHloPassIncGen",
        ":placement_utils",
        ":npu_hlo_utils",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow:tensorflow",
        "@llvm-project//mlir:Dialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:Transforms",
        "@llvm-project//llvm:Core",
    ],
    alwayslink = 1,
)

cc_library(
    name = "tf_to_mhlo",
    srcs = [
        "transforms/tf_to_mhlo.cpp",
    ],
    hdrs = ["transforms/passes.h"],
    deps = [
        ":NpuHloPassIncGen",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow:tensorflow",
        "@local_xla//xla/mlir_hlo:mlir_hlo",
        "@llvm-project//mlir:Dialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:Transforms",
        "@llvm-project//mlir:TransformUtils",
        "@llvm-project//llvm:Core",
    ],
    alwayslink = 1,
)

cc_library(
    name = "placement_utils",
    srcs = ["transforms/placement_utils.cpp"],
    hdrs = ["transforms/placement_utils.h"],
    deps = [
        "@llvm-project//mlir:Support",
    ],
    alwayslink = 1,
)

cc_library(
    name = "npu_hlo_utils",
    srcs = [
        "npu_hlo_utils.cpp",
    ],
    hdrs = [
        "npu_hlo_utils.h",
    ],
    deps = [
        "@llvm-project//llvm:Support",
    ],
)