cmake_minimum_required(VERSION 3.14.1)

project(ctr CXX C)

execute_process(COMMAND ${CMAKE_C_COMPILER} -dumpfullversion -dumpversion
                OUTPUT_VARIABLE GCC_VERSION)
        string(REGEX MATCHALL "[0-9]+" GCC_VERSION_COMPONENTS ${GCC_VERSION})
        list(GET GCC_VERSION_COMPONENTS 0 GCC_MAJOR)
        list(GET GCC_VERSION_COMPONENTS 1 GCC_MINOR)

        message(STATUS "cmake version=${CMAKE_VERSION}")

        set(GCC_VERSION "${GCC_MAJOR}.${GCC_MINOR}")
        message(STATUS "gcc version=${GCC_VERSION}")

        execute_process(
            COMMAND ${CMAKE_CXX_COMPILER} --version
            OUTPUT_VARIABLE GXX_VERSION_OUTPUT
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )

        # 打印 g++ 版本信息
        message(STATUS "G++ version: ${GXX_VERSION_OUTPUT}")

set(COMMON_CXX_FLAGS 
        -Wall
        -fPIC
        -fopenmp
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
        -std=c++17
        -march=armv8-a+sve
        -fno-lto
        -msve-vector-bits=256
)
if (${BUILD_MODE} MATCHES "release")
    message("======BUILD_MODE release")
    set(CXX_FLAGS
            -O3
            ${COMMON_CXX_FLAGS}
            -D_FORTIFY_SOURCE=2
            -fstack-protector-all
            -fstack-protector-strong
            -Wl,--disable-new-dtags
            )
else()
    message("======BUILD_MODE ut")
    set(CXX_FLAGS
            -g
            ${COMMON_CXX_FLAGS}
    )

endif (${BUILD_MODE} MATCHES "release")

string(REPLACE ";" " " CMAKE_CXX_FLAGS "${CXX_FLAGS} ${CMAKE_CXX_FLAGS}")

if (${CMAKE_HOST_SYSTEM_PROCESSOR} MATCHES "aarch64")
    add_definitions(-D__AARCH64__)
    set(CMAKE_ARC linux-aarch64)
else ()
    message(FATAL_ERROR "don't support ${CMAKE_HOST_SYSTEM_PROCESSOR}")
endif (${CMAKE_HOST_SYSTEM_PROCESSOR} MATCHES "aarch64")

include_directories(
        ${PROJECT_SOURCE_DIR}/SRC
)

add_subdirectory(src)

if (${BUILD_MODE} MATCHES "ut")
    add_subdirectory(3rdparty)
    add_subdirectory(tests)
endif (${BUILD_MODE} MATCHES "ut")