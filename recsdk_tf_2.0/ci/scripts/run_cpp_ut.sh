#!/bin/bash
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

set -e

if [ -z "$1" ]; then
    echo "Usage: $0 <tf_version>"
    exit 1
fi

python_home="$(dirname "$(dirname "$(readlink -f "$(which python3.7)")")")"
echo "$python_home"
echo "$ASCEND_TOOLKIT_HOME"
tf_version=$1

scripts_dir=$(dirname "$(readlink -f "$0")")
project_dir=$(dirname "$(dirname "${scripts_dir}")")
build_dir=$project_dir/build

echo "$scripts_dir"
echo "$project_dir"

cmake -S "$project_dir" -B "$build_dir" \
    -DCMAKE_BUILD_TYPE=Debug \
    -DCMAKE_C_COMPILER=gcc \
    -DCMAKE_CXX_COMPILER=g++ \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    -DPYTHON_HOME="$python_home" \
    -DASCEND_TOOLKIT_HOME="$ASCEND_TOOLKIT_HOME" \
    -DTF_VERSION="$tf_version"

num_cores=$(($(nproc)-1))
cmake --build "$build_dir" --config Debug --parallel $num_cores

cd "$build_dir"/mxrec/core/tests
./runTests --gtest_output=xml:test_detail.xml
cd -

mv "$project_dir"/build ./

COVERAGE_FILE=coverage.info
REPORT_FOLDER=coverage_report
lcov --rc lcov_branch_coverage=1 -c -d ./build -o "${COVERAGE_FILE}"_tmp
lcov -r "${COVERAGE_FILE}"_tmp "$project_dir"/third_party/spdlog/include/spdlog/* \
        "$project_dir"/mxrec/core/tests/ut/host/feat/* \
        "$project_dir"/third_party/spdlog/include/spdlog/fmt/bundled/* \
        "$project_dir"/third_party/spdlog/include/spdlog/sinks/* \
        "$project_dir"/third_party/googletest/googletest/include/gtest/internal/* \
        "$project_dir"/third_party/googletest/googletest/include/gtest/* \
        "$project_dir"/third_party/spdlog/include/spdlog/details/* '/usr/*' \
        '/opt/buildtools/gcc-7.3.0/include/c++/*' \
        '/opt/buildtools/gcc-7.3.0/lib/gcc/x86_64-pc-linux-gnu/7.3.0/include/*' \
        --rc lcov_branch_coverage=1 -o "${COVERAGE_FILE}"
genhtml --rc genhtml_branch_coverage=1 "${COVERAGE_FILE}" -o "${REPORT_FOLDER}"

[ -e "${COVERAGE_FILE}"_tmp ] && rm -rf "${COVERAGE_FILE}"_tmp
[ -e "${COVERAGE_FILE}" ] && rm -rf "${COVERAGE_FILE}"
