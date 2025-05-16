acl_hdrs=glob(["include/**/*.h"])

cc_import(
    name = "acl",
    shared_library = "lib64/libascendcl.so",
    visibility = ["//visibility:public"],
)

cc_import(
    name = "acl_rt",
    shared_library = "lib64/libruntime.so",
    visibility = ["//visibility:public"],
)

cc_import(
    name = "nnopbase",
    shared_library = "lib64/libnnopbase.so",
    visibility = ["//visibility:public"],
)

cc_import(
    name = "opapi",
    shared_library = "lib64/libopapi.so",
    visibility = ["//visibility:public"],
)

cc_library(
    name = "aclnn_deps",
    hdrs = acl_hdrs,
    includes = ["include/"],
    deps = [
        ":acl",
        ":nnopbase",
        ":opapi",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "acl_deps",
    hdrs = acl_hdrs,
    includes = ["include/"],
    deps = [
        ":acl",
        ":acl_rt",
    ],
    visibility = ["//visibility:public"],
)
