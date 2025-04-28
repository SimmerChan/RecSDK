# Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
if("x${CMAKE_CXX_COMPILER}" STREQUAL "x")
    set(CMAKE_CXX_COMPILER g++)
endif()
if("x${CMAKE_C_COMPILER}" STREQUAL "x")
    set(CMAKE_C_COMPILER gcc)
endif()

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS_RELEASE "-fPIC -O2 -Wall -s")
set(CMAKE_CXX_FLAGS_DEBUG "-fPIC -O0 -g -Wall")
set(CAMKE_SKIP_RPATH TRUE)
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-z,relro,-z,now,-z,noexecstack -pie")
add_compile_options(
    -fstack-protector-all
    -D_FORTIFY_SOURCE=2 -O2
    )
# add source files
aux_source_directory(. SRCS)
aux_source_directory(${OPS_SRC} SRCS)


if("x${SRCS}" STREQUAL "x")
    add_custom_target(${OP_PROTO_TARGET}
            COMMAND mkdir -p ${OP_PROTO_TARGET_OUT_DIR}
            COMMAND echo "no source to make lib${OP_PROTO_TARGET}.so")
    return(0)
endif()

set(LIBRARY_OUTPUT_PATH ${OP_PROTO_TARGET_OUT_DIR})

message( STATUS "OP_PROTO_TARGET=${OP_PROTO_TARGET}")
add_library(${OP_PROTO_TARGET} SHARED ${SRCS})
target_include_directories(${OP_PROTO_TARGET} PRIVATE
    ${METADEF_INCLUDE}
    ${OPS_INCLUDE}
    ${PROTO_BINARY_DIR}/opp
    ${Protobuf_INCLUDE}
    ${METADEF_INCLUDE}/exe_graph
    ${METADEF_INCLUDE}/external
     ${METADEF_INCLUDE}/external/exe_graph
    ${OPENSDK_DIR}/c_sec/include
    ${OPENSDK_DIR}/include/air
    ${OPENSDK_DIR}/include/slog
    ${ASCEND_OPP_PATH}/built-in/op_proto/inc
    ${OPENSDK_DIR}/include/air/external/
)
target_link_libraries(${OP_PROTO_TARGET} ${ASCEND_INC}/../lib64/libgraph.so ${ASCEND_INC}/../lib64/libregister.so)
