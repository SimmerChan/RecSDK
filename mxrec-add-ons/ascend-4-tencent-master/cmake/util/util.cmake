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

function(cann_install)
  cmake_parse_arguments(
    MY_INSTALL
    "COPY_CONTENTS"
    "TARGET;DESTINATION"
    "FILES;DIRECTORIES"
    ${ARGN}
  )
  if(NOT MY_INSTALL_TARGET)
    message(FATAL_ERROR "Target required for installing files or dirs.")
  endif()

  if(MY_INSTALL_FILES)
    add_custom_command(
      TARGET "${MY_INSTALL_TARGET}" POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E make_directory ${MY_INSTALL_DESTINATION}
      COMMAND ${CMAKE_COMMAND} -E copy_if_different ${MY_INSTALL_FILES} ${MY_INSTALL_DESTINATION}
      COMMENT "Install files: ${MY_INSTALL_FILES} to ${MY_INSTALL_DESTINATION}"
    )
  endif()
  foreach(_src_path ${MY_INSTALL_DIRECTORIES})
    if(NOT IS_DIRECTORY ${_src_path})
      message(FATAL_ERROR "Directory is not exist. ${_src_path}")
    endif()

    set(_dest_path ${MY_INSTALL_DESTINATION})
    if(MY_INSTALL_COPY_CONTENTS)
      # copy contents of the directory
    else()
      # copy directory itself
      get_filename_component(_dir_name ${_src_path} NAME)
      set(_dest_path ${MY_INSTALL_DESTINATION}/${_dir_name})
    endif()
    add_custom_command(
      TARGET "${MY_INSTALL_TARGET}" POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_directory ${_src_path} ${_dest_path}
      COMMENT "Install dirs: ${MY_INSTALL_DIRECTORY} to ${MY_INSTALL_DESTINATION}"
    )
  endforeach()
endfunction()


function(protobuf_generate project cc_files h_files)
  if(NOT ARGN)
    message(SEND_ERROR "Error: protobuf_generate() called without any proto files")
    return()
  endif()

  set(${h_files})
  set(${cc_files})
  set(_add_target FALSE)

  foreach(_file ${ARGN})
    if("${_file}" STREQUAL "TARGET")
      set(_add_target TRUE)
      continue()
    endif()
    if(NOT EXISTS ${_file})
      message(SEND_ERROR "Error: files ars not exist ${_file}")
      return()
    endif()
    get_filename_component(_abs_file ${_file} ABSOLUTE)
    get_filename_component(_proto_name ${_abs_file} NAME)
    get_filename_component(_proto_path ${_abs_file} PATH)
    get_filename_component(_proto_parent_dir ${_proto_path} NAME)

    if("${_proto_parent_dir}" STREQUAL "proto")
      set(_proto_outpath "${PROTO_BINARY_DIR}/${project}/proto")
    else()
      set(_proto_outpath "${PROTO_BINARY_DIR}/${project}/proto/${_proto_parent_dir}")
    endif()
    file(MAKE_DIRECTORY ${_proto_outpath})

    string(REPLACE ".proto" ".pb.h"  _proto_pb_h ${_proto_name})
    string(REPLACE ".proto" ".pb.cc" _proto_pb_cc ${_proto_name})
    set(_out_proto_h  "${_proto_outpath}/${_proto_pb_h}")
    set(_out_proto_cc "${_proto_outpath}/${_proto_pb_cc}")
    list(APPEND ${h_files}  ${_out_proto_h})
    list(APPEND ${cc_files} ${_out_proto_cc})

    add_custom_command(
      OUTPUT ${_out_proto_h} ${_out_proto_cc}
      COMMAND ${Protobuf_PROTOC_EXECUTABLE} -I${_proto_path} --cpp_out=${_proto_outpath} ${_file}
      DEPENDS protoc
      COMMENT "Running Protubuf to generate ${_proto_pb_cc} and ${_proto_pb_h}"
    )
  endforeach()

  if(_add_target)
    add_custom_target(
      ${project} DEPEND ${${cc_files}} ${${h_files}}
    )
  endif()

  set(${h_files}  ${${h_files}} PARENT_SCOPE)
  set(${cc_files} ${${cc_files}} PARENT_SCOPE)
  set_source_files_properties(${${h_files}} ${${cc_files}} PROPERTIES GENERATED TRUE)

endfunction()


#[[
  module - the name of export imported target
  name   - find the library name
  path   - find the library path
#]]
function(find_module module name path)
  if (TARGET ${module})
    return()
  endif()
  find_library(${module}_LIBRARY_DIR NAMES ${name} NAMES_PER_DIR PATHS ${path}
    PATH_SUFFIXES lib
  )

  message(STATUS "find ${name} location ${${module}_LIBRARY_DIR}")
  if ("${${module}_LIBRARY_DIR}" STREQUAL "${module}_LIBRARY_DIR-NOTFOUND")
    message(FATAL_ERROR "${name} not found in ${path}")
  endif()

  add_library(${module} SHARED IMPORTED)
  set_target_properties(${module} PROPERTIES
    IMPORTED_LOCATION ${${module}_LIBRARY_DIR}
  )
endfunction()
