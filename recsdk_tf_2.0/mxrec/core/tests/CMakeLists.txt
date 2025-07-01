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

if(NOT EXISTS ${PROJECT_SOURCE_DIR}/third_party/googletest)
    message(FATAL_ERROR
        "The essential dependency ==> googletest <== is not available, "
        "please check existence of path: `mxrec/third_party/googletest`!"
    )
endif()
set(GTEST_DIR ${PROJECT_SOURCE_DIR}/third_party/googletest/googletest)
set(GMOCK_DIR ${PROJECT_SOURCE_DIR}/third_party/googletest/googlemock)

set(CMAKE_INCLUDE_CURRENT_DIR ON)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../host)
include_directories(${ASCEND_DIR}/include)
include_directories(${TF_DIR}/include)
include_directories(${SPDLOG_DIR}/include)
include_directories(${GTEST_DIR}/include)
include_directories(${GMOCK_DIR}/include)

file(GLOB_RECURSE TEST_SRC_FILES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
find_library(host PATHS ${CMAKE_BUILD_RPATH}/mxrec/core/host)

set(CMAKE_CXX_FLAGS "--coverage")

add_executable(runTests ${TEST_SRC_FILES})
target_link_libraries(runTests PRIVATE ${TF_LIB})
target_link_libraries(runTests PRIVATE ${ACL_LIB})
target_link_libraries(runTests PRIVATE ${ACL_TDT_LIB})
target_link_libraries(runTests PRIVATE host)
target_link_libraries(runTests PRIVATE gtest)
target_link_libraries(runTests PRIVATE gtest_main)
