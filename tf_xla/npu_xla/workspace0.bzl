load("@org_third_party//bazel/tf:tf_configure.bzl", "tf_configure")

def _tf_bridge_toolchains():
    tf_configure(name = "local_config_tf")

def workspace():
    _tf_bridge_toolchains()

npu_xla_workspace0 = workspace