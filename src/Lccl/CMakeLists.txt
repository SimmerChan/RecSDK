set(ATB_PATH ${PROJECT_SOURCE_DIR})
cmake_minimum_required(VERSION 3.12)
project(Lccl LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 14)
find_package(MPI)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
message("CMAKE_MODULE_PATH: ${CMAKE_MODULE_PATH}")
add_definitions(-DLCAL_REPO)

option(USE_CXX11_ABI "USE_CXX11_ABI" 0)

IF (CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "aarch64")
    SET(ARCH aarch64)
ELSEIF (CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "x86_64")
    SET(ARCH x86_64)
ENDIF ()

if(USE_CXX11_ABI)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=1")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
endif()

message("CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")
message("CMAKE_SOURCE_DIR: ${CMAKE_SOURCE_DIR}")

# 获取环境变量
if (DEFINED ENV{ASCEND_HOME_PATH})
    set(ASCEND_HOME_PATH $ENV{ASCEND_HOME_PATH})
else ()
    message("ASCEND_HOME_PATH not set! using default path!")
    set(ASCEND_HOME_PATH /usr/local/Ascend/ascend-toolkit/latest)
endif ()

message("ASCEND_HOME_PATH: ${ASCEND_HOME_PATH}")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-error=deprecated-declarations -Wno-deprecated-declarations")
add_link_options(-Wl,-z,relro,-z,now)
add_link_options(-s)

include_directories(${MPI_INCLUDE_PATH}
        ${ASCEND_HOME_PATH}/${ARCH}-linux/include/
        ${ASCEND_HOME_PATH}/${ARCH}-linux/include/hccl/
        ${ASCEND_HOME_PATH}/${ARCH}-linux/include/experiment/runtime/
        ${ASCEND_HOME_PATH}/${ARCH}-linux/include/experiment/msprof/
        ${CMAKE_CURRENT_SOURCE_DIR}/src/3rdparty/
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        )
link_directories(${ASCEND_HOME_PATH}/${ARCH}-linux/lib64/)

message("MPI_FOUND: ${MPI_FOUND}")

set(AIV_ARCH dav-c220-vec)
set(AIC_ARCH dav-c220-cube)

add_subdirectory(src)