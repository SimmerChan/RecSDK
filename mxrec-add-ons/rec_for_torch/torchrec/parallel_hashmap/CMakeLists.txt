cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(feature_map)
find_package(OpenMP REQUIRED)
find_package(gflags REQUIRED)
find_package(folly REQUIRED)

set(CXX_FLAGS
-O3
-Wall
-fPIC
-fms-extensions
-Wunused-variable
-Wunused-value
-Wcast-align
-Wcast-qual
-Winvalid-pch
-Wwrite-strings
-Wsign-compare
-Wfloat-equal
-Wextra
-D_FORTIFY_SOURCE=2
-std=c++17
-fstack-protector-all
-fstack-protector-strong
)
set(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -ffunction-sections -O3 -Wfatal-errors -DNDEBUG -fPIC -fstack-protector-all -D_FORTIFY_SOURCE=2 -s")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,-z,relro -Wl,-z,now -Wl,-z,noexecstack")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
include_directories(include)

add_library(parallel_hashmap SHARED parallel_hash_map.cpp)
target_link_libraries(parallel_hashmap OpenMP::OpenMP_CXX Folly::folly gflags) # MAIN