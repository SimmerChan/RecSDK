cc_import(
    name = "runtime_pipeline",
    shared_library = "libruntime_pipeline.so",
    visibility = ["//visibility:public"],
)

cc_import(
    name = "graph_runtime",
    shared_library = "libsitere_grt.so",
    visibility = ["//visibility:public"],
)

cc_library(
    name = "inference_graph",
    hdrs = [
        "runtime_pipeline.h",
        "compiler_pipeline.h",
    ],
    srcs = ["compiler_pipeline.cpp"],
    deps = [
        "@inference_graph_base//:runtime_pipeline",
        "@inference_graph_base//:graph_runtime",
        "@com_google_absl//absl/status:statusor",
        "@local_tsl//tsl/platform:logging",
    ],
    visibility = ["//visibility:public"],
)