# cpu
set(ASCEND_INSTALL_PATH "/usr/local/Ascend/ascend-toolkit/latest")
set(smoke_testcase "debug")
set(CMAKE_PREFIX_PATH ${ASCEND_INSTALL_PATH}/tools/tikicpulib/lib/cmake)

find_package(tikicpulib REQUIRED)

file(GLOB SRC_FILES
    ${CMAKE_SOURCE_DIR}/op_kernel/*.cpp
)
add_library(${smoke_testcase}_cpu
    ${SRC_FILES}
)
target_include_directories(${smoke_testcase}_cpu PRIVATE
    ${ASCEND_INSTALL_PATH}/acllib/include
)


target_link_libraries(${smoke_testcase}_cpu PRIVATE
    tikicpulib::ascend910
    ascendcl
)

target_compile_options(${smoke_testcase}_cpu PRIVATE
    -g
)

set_target_properties(${smoke_testcase}_cpu PROPERTIES
    OUTPUT_NAME ${smoke_testcase}_cpu
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}
)
