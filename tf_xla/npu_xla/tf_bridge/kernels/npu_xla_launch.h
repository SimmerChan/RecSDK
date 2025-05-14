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

#ifndef NPU_XLA_TF_BRIDGE_KERNELS_NPU_XLA_LAUNCH_H_
#define NPU_XLA_TF_BRIDGE_KERNELS_NPU_XLA_LAUNCH_H_

#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {
namespace npu_xla {

class NpuXlaLaunchOp : public OpKernel {
public:
    explicit NpuXlaLaunchOp(OpKernelConstruction* ctx);

    void Compute(OpKernelContext* context) override;

private:
    Status CompileAndRunMlir(OpKernelContext* context);

    Status CreateLocalTempDirPath(std::string* dirPath);

    std::vector<int> constants_;
    std::vector<int> fixed_shapes_;
    std::vector<int> host_args_;
    std::vector<int> resources_;
    NameAttrList func_;
    std::string device_type_;
};

}  // namespace npu_xla

#define REGISTER_NPU_XLA_LAUNCH_KERNEL(DEVICE)             \
    REGISTER_KERNEL_BUILDER(Name("NpuXlaLaunch")           \
                                .Device(DEVICE)            \
                                .HostMemory("constants")   \
                                .HostMemory("fixedshapes") \
                                .HostMemory("hostargs")    \
                                .HostMemory("hostresults") \
                                .HostMemory("resources"),  \
                            npu_xla::NpuXlaLaunchOp)
}  // namespace tensorflow

#endif  // NPU_XLA_TF_BRIDGE_KERNELS_NPU_XLA_LAUNCH_H_