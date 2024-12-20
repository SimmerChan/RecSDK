# Copyright (c) Huawei Technologies Co., Ltd. 2022-2024. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

add_compile_options(-ftest-coverage -fprofile-arcs)
link_libraries(gcov)

set(LCCL_UTIL_INSTALL_DIR        ${PROJECT_SOURCE_DIR}/../AccCTR/install)
set(LCCL_SRC_DIR ${PROJECT_SOURCE_DIR}/src)
message("src" ${LCCL_SRC_DIR})

# 包含所有组件的cmake
include("${CMAKE_CURRENT_SOURCE_DIR}/../conf/toolchain.cmake")
set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../src)
set(TOP_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../)

file(GLOB_RECURSE TEST_LCCL_FILES *.cpp *.h)
add_executable(test_lccl_files ${TEST_LCCL_FILES})
include_directories(${LCCL_UTIL_INSTALL_DIR}/googletest-release-1.8.1/include)
link_directories(${LCCL_UTIL_INSTALL_DIR}/googletest-release-1.8.1/lib64)

SET(LIB_3RD_GMOCK ${LCCL_UTIL_INSTALL_DIR}/googletest-release-1.8.1/lib64/libgmock.a)
SET(LIB_3RD_GTEST ${LCCL_UTIL_INSTALL_DIR}/googletest-release-1.8.1/lib64/libgtest.a)


message(${LCCL_SRC_DIR}/include)
# 添加库文件的搜索路径
target_link_directories(test_lccl_files
        PUBLIC
        ${PROJECT_SOURCE_DIR}/output/lccl_common/lib
        )

# 添加头文件的搜索路径
target_include_directories(test_lccl_files
        PUBLIC
        ${LCCL_SRC_DIR}/include
        ${PROJECT_SOURCE_DIR}
        ${LCCL_SRC_DIR}/common/util
        )

# 用来指定要链接的库
target_link_libraries(test_lccl_files
        PUBLIC
        -Wl,--start-group
        _ock_ctr_common
        pthread
        dl
        ${LIB_3RD_GTEST}
        ${LIB_3RD_GMOCK}
        -Wl,--end-group)

# 打印构建选项
get_target_property(COMPILE_FLAGS test_lccl_files COMPILE_OPTIONS)
get_target_property(LINK_FLAGS test_lccl_files LINK_OPTIONS)
message(STATUS "Compiler id: ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "Compile flags: ${COMPILE_FLAGS}")
message(STATUS "Link flags: ${LINK_FLAGS}")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")

