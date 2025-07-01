# Copyright 2025. Huawei Technologies Co.,Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

cmake_minimum_required(VERSION 3.12)
set(CMAKE_CXX_STANDARD 14)
add_compile_options(-D_GLIBCXX_USE_CXX11_ABI=0)

set(TF_PATH "/opt/conda/envs/tf1/lib/python3.7/site-packages/tensorflow_core")
include_directories(${TF_PATH}/include)
link_directories(${TF_PATH})

file(GLOB_RECURSE MXREC_OP_SRC ./*.cpp)
add_library(cust_ops SHARED ${MXREC_OP_SRC})

set(ASCEND_DRIVER_PATH /usr/local/Ascend/driver)
set(ASCEND_PATH /usr/local/Ascend/ascend-toolkit/latest)

target_include_directories(cust_ops
    PUBLIC
        ${ASCEND_DRIVER_PATH}/include
)

target_link_directories(cust_ops
    PUBLIC
        ${ASCEND_PATH}/../../tfplugin/latest/python/site-packages/npu_bridge
        ${ASCEND_PATH}/fwkacllib/lib64
        ${ASCEND_PATH}/compiler/lib64
        ${ASCEND_PATH}/runtime/lib64
        ${ASCEND_DRIVER_PATH}/lib64/driver
)

target_link_libraries(cust_ops PUBLIC ascendcl msprofiler ge_executor gert runtime ge_common register graph ascend_protobuf
    profapi opt_feature error_manager exe_graph acl_tdt_channel acl_tdt_queue)
target_link_libraries(cust_ops PUBLIC
        -l:_tf_adapter.so
)
