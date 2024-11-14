
cmake_minimum_required(VERSION 3.12)
set(CMAKE_CXX_STANDARD 14)

include_directories(/usr/local/python3.7.5/lib/python3.7/site-packages/tensorflow_core/include)
link_directories(/usr/local/python3.7.5/lib/python3.7/site-packages/tensorflow_core/)

file(GLOB_RECURSE COMM_OP_SRC ./*.cpp)
add_library(rma_dyn_ops SHARED ${COMM_OP_SRC})
target_link_libraries(rma_dyn_ops /usr/local/python3.7.5/lib/python3.7/site-packages/tensorflow_core/libtensorflow_framework.so.1)
target_include_directories(rma_dyn_ops PUBLIC /usr/local/python3.7.5/lib/python3.7/site-packages/tensorflow_core/include)
install(TARGETS rma_dyn_ops LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX})
