load("@com_google_protobuf//:protobuf.bzl", "cc_proto_library")
load("@org_tensorflow//tensorflow:tensorflow.bzl", "tf_cc_shared_object", "tf_copts")
load("@local_config_tf//:build_defs.bzl", "tf_major_version")

package(default_visibility = ["//visibility:public"])

cc_proto_library(
    name = "xla_activity",
    srcs = [
        "tf_bridge/xla_activity.proto",
    ],
)

cc_library(
    name = "acl_adaptor",
    srcs = ["adaptor/acl_adaptor.cpp"],
    hdrs = ["adaptor/acl_adaptor.h"],
    deps = [
        "@acl_base//:acl_deps",
        "@com_google_absl//absl/numeric:int128",
        "@eigen_archive//:eigen3",
        "@local_config_tf//:tf_header_lib",
        # "@local_config_tf//:libtensorflow_framework",
    ],
)

cc_library(
    name = "common_hdrs",
    hdrs = glob(
        include = [
            "common_hdrs/**/*.h"
        ],
    ),
)

cc_library(
    name = "tf_bridge",
    srcs = glob(
        include = [
            "tf_bridge/**/*.cc",
            "tf_bridge/**/*.cpp",
        ],
        exclude = [
            "tf_bridge/**/*test.cc",
            "tf_bridge/**/*test.cpp",
        ],
    ) + [
            ":xla_activity_genproto",
            "//tf_mlir:compiler_input_genproto",
        ],
    hdrs = glob(
        include = [
            "tf_bridge/**/*.h"
        ],
    ),
    deps = [
            "acl_adaptor",
            ":common_hdrs",
            "@local_config_tf//:tf_header_lib",
            "@local_config_tf//:libtensorflow_framework",
            "@local_config_tf//:libtensorflow_cc",
            "@com_google_absl//absl/numeric:int128",
            "@com_google_protobuf//:protobuf",
            "@eigen_archive//:eigen3",
            "@inference_graph_base//:inference_graph",
    ],
    copts = tf_copts(is_external = True),
    linkopts = [
        "-lm",
        "-ldl",
    ],
    alwayslink = 1,  # targets depending on it should carry all symbols in its children.
    linkstatic = 1,
)

cc_library(
    name = "npu_device",
    srcs = glob(
        include = [
            "npu_device/**/*.cc",
            "npu_device/**/*.cpp"
        ],
        exclude = [
            "npu_device/**/*test.cc",
            "npu_device/**/*test.cpp",
        ],
    ),
    hdrs = glob(
        include = [
            "npu_device/**/*.h"
        ],
    ),
    deps = [
            "acl_adaptor",
            "common_hdrs",
            "//acl_deps:acl_rt_hdrs",
            "@acl_base//:acl_deps",
            "@local_config_tf//:tf_header_lib",
            "@local_config_tf//:libtensorflow_framework",
            "@com_google_absl//absl/numeric:int128",
            "@eigen_archive//:eigen3",
    ],
    copts = tf_copts(is_external = True),
    linkopts = [
        "-lm",
    ],
    alwayslink = 1,  # targets depending on it should carry all symbols in its children.
    linkstatic = 1,
)

tf_cc_shared_object(
    name = "libnpu_xla.so",
    deps = [
        ":tf_bridge",
    ],
    linkopts = [
        "-z defs",
    ],
)

tf_cc_shared_object(
    name = "libnpu_device.so",
    deps = [
        ":npu_device",
    ],
    linkopts = [
        "-z defs",
    ],
)