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

set(CMAKE_INCLUDE_CURRENT_DIR ON)
include_directories(${TF_DIR}/include)
include_directories(${ASCEND_DIR}/include)
include_directories(${PYBIND11_DIR}/include)
include_directories(${SPDLOG_DIR}/include)

file(GLOB_RECURSE SRC_FILES *.cpp)
list(REMOVE_ITEM SRC_FILES ${CMAKE_CURRENT_SOURCE_DIR}/binding.cpp)

add_library(host STATIC ${SRC_FILES})
target_link_libraries(host PRIVATE ${TF_LIB})
target_link_libraries(host PRIVATE ${ACL_LIB})
target_link_libraries(host PRIVATE ${ACL_TDT_LIB})
target_link_libraries(host PRIVATE spdlog)

pybind11_add_module(binding binding.cpp)
target_link_libraries(binding PRIVATE host)
