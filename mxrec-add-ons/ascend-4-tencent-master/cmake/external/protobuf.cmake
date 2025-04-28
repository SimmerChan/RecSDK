# Copyright 2020 Huawei Technologies Co., Ltd
#
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
# ============================================================================

ExternalProject_Add(external_protobuf
  URL                ${TOP_DIR}/third_party/protobuf/v3.13.0.tar.gz
  URL_MD5           1a6274bc4a65b55a6fa70e264d796490
  DOWNLOAD_DIR      download/protobuf
  PREFIX            third_party
  SOURCE_SUBDIR     cmake
  CMAKE_CACHE_ARGS
      -DProtobuf_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
      -Dprotobuf_BUILD_TESTS:BOOL=OFF
      -Dprotobuf_BUILD_EXAMPLES:BOOL=OFF
      -Dprotobuf_BUILD_SHARED_LIBS:BOOL=OFF
      -DProtobuf_CXX_COMPILER:STRING=${CMAKE_CXX_COMPILER}
  INSTALL_COMMAND   ""
)

ExternalProject_Get_Property(external_protobuf SOURCE_DIR)
ExternalProject_Get_Property(external_protobuf BINARY_DIR)

set(Protobuf_INCLUDE ${SOURCE_DIR}/src)
set(Protobuf_PATH ${BINARY_DIR})
set(Protobuf_PROTOC_EXECUTABLE ${Protobuf_PATH}/protoc)

add_custom_command(
  OUTPUT ${Protobuf_PROTOC_EXECUTABLE}
  DEPENDS external_protobuf
)
add_custom_target(
  protoc ALL DEPENDS ${Protobuf_PROTOC_EXECUTABLE}
)
