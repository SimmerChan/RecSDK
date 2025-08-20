# Copyright 2024. Huawei Technologies Co.,Ltd. All rights reserved.
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
set(CMAKE_CXX_STANDARD 17)


if(NOT SECUREC_PATH)
    set(SECUREC_PATH ${PROJECT_SOURCE_DIR}/../platform/securec)
endif()
message("SECUREC_PATH: " ${SECUREC_PATH})

if (${TF_PATH} MATCHES "tensorflow_core")
    include_directories(${PYTHON_PATH}/lib/python3.7/site-packages/tensorflow_core/include)
else()
    include_directories(${PYTHON_PATH}/lib/python3.7/site-packages/tensorflow/include)
endif()

link_libraries(stdc++fs)

file(GLOB_RECURSE COMMON_SRC ./*.cpp ./*.h)
add_library(COMMON_ASC SHARED ${COMMON_SRC})

target_include_directories(COMMON_ASC
    PUBLIC
        ${SECUREC_PATH}/include
        ${ASCEND_DRIVER_PATH}/include
        ${ASCEND_DRIVER_PATH}/kernel/inc
)

target_link_directories(COMMON_ASC
    PUBLIC
        ${ASCEND_PATH}/fwkacllib/lib64
        ${ASCEND_PATH}/compiler/lib64
        ${ASCEND_PATH}/runtime/lib64
        ${SECUREC_PATH}/lib
        ${ASCEND_DRIVER_PATH}/lib64/driver
)

target_link_libraries(COMMON_ASC PUBLIC ascendcl ge_executor gert runtime ge_common register graph ascend_protobuf
     opt_feature error_manager exe_graph acl_tdt_channel acl_tdt_queue securec drvdsmi_host)
target_link_libraries(COMMON_ASC PUBLIC
        -l:_tf_adapter.so
        ${PYTHON_LIBRARY}
)

install(TARGETS COMMON_ASC LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX})
