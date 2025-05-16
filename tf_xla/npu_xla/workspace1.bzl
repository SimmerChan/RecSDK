def _npu_xla_repositories():
    native.local_repository(
        name = "org_tensorflow",
        path = "../tf_community/",
    )
    native.local_repository(
        name = "org_third_party",
        path = "../third_party/",
    )

def workspace():
    _npu_xla_repositories()

# Alias so it can be loaded without assigning to a different symbol to prevent
# shadowing previous loads and trigger a buildifier warning.
npu_xla_workspace1 = workspace