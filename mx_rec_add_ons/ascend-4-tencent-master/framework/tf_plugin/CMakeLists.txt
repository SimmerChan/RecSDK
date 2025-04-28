# Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
set(_proto_srcs
	"${OPENSDK_DIR}/proto/tensorflow/versions.proto"
	"${OPENSDK_DIR}/proto/tensorflow/resource_handle.proto"
	"${OPENSDK_DIR}/proto/tensorflow/tensor_shape.proto"
	"${OPENSDK_DIR}/proto/tensorflow/function.proto"
  "${OPENSDK_DIR}/proto/tensorflow/node_def.proto"
  "${OPENSDK_DIR}/proto/tensorflow/graph.proto"
  "${OPENSDK_DIR}/proto/tensorflow/tensor.proto"
  "${OPENSDK_DIR}/proto/tensorflow/attr_value.proto"
  "${OPENSDK_DIR}/proto/tensorflow/graph_library.proto"
  "${OPENSDK_DIR}/proto/tensorflow/types.proto"
  "${OPENSDK_DIR}/proto/tensorflow/op_def.proto"
)


protobuf_generate(opp _proto_cc _proto_h ${_proto_srcs})

aux_source_directory(. SRCS)
message(STATUS "SRCS = ${SRCS}")

if("x${SRCS}" STREQUAL "x")
    add_custom_target(${TF_PLUGIN_TARGET}
            COMMAND mkdir -p ${TF_PLUGIN_TARGET_OUT_DIR}
            COMMAND echo "no source to make lib${TF_PLUGIN_TARGET}.so")
    return(0)
endif()

set(LIBRARY_OUTPUT_PATH ${TF_PLUGIN_TARGET_OUT_DIR})

add_library(${TF_PLUGIN_TARGET} SHARED ${SRCS} ${_proto_h})

target_compile_definitions(${TF_PLUGIN_TARGET} PRIVATE
    google=ascend_private
)

target_include_directories(${TF_PLUGIN_TARGET} PRIVATE
    ${METADEF_INCLUDE}
    ${PROTO_BINARY_DIR}/opp
    ${Protobuf_INCLUDE}
    ${ASCEND_OPP_PATH}/built-in/op_proto/inc
    ${OPENSDK_DIR}/include/metadef/external
    ${OPENSDK_DIR}/include/air

)

target_link_libraries(${TF_PLUGIN_TARGET} ${ASCEND_INC}/../lib64/libgraph.so  ${ASCEND_INC}/../lib64/libregister.so)
