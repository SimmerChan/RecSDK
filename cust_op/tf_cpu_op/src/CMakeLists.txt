set(REC_BASE_LINK_FLAGS -Wl,-z,relro,-z,now,-z,noexecstack -s)

set(REC_BASE_SRC_DIR                    ${PROJECT_SOURCE_DIR})
set(REC_BASE_SRC_INCLUDE_DIR                ${PROJECT_SOURCE_DIR}/src/include)
set(REC_BASE_COMMON_DIR                 ${PROJECT_SOURCE_DIR}/src/common)
set(REC_BASE_BUILD_PATH                 ${PROJECT_SOURCE_DIR}/build)

set(OUTPUT                              ${PROJECT_SOURCE_DIR}/output)
set(REC_BASE_PLATFORM_UTIL_DIR          ${PROJECT_SOURCE_DIR}/3rdparty)
set(REC_BASE_UTIL_INSTALL_DIR           ${PROJECT_SOURCE_DIR}/install)

if (${BUILD_MODE} MATCHES "ut")
    add_compile_options(-ftest-coverage -fprofile-arcs)
    link_libraries(gcov)
endif (${BUILD_MODE} MATCHES "ut")

message("include : " ${REC_BASE_SRC_INCLUDE_DIR})

add_subdirectory(include)
add_subdirectory(kunpeng)

add_library(_rec_base SHARED)

target_include_directories(_rec_base
        PUBLIC
        ${PROJECT_SOURCE_DIR}
        ${REC_BASE_SRC_INCLUDE_DIR}
        ${REC_BASE_COMMON_DIR}
        )

target_link_libraries(_rec_base  PUBLIC
        -Wl,--start-group
        dl
        ${REC_BASE_LINK_FLAGS}
        kunpeng
        -Wl,--end-group
        )

set(TARGET_INSTALL_LIB ${OUTPUT}/rec_base/lib)
install(TARGETS _rec_base DESTINATION ${TARGET_INSTALL_LIB}/ PERMISSIONS OWNER_WRITE OWNER_READ OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)

