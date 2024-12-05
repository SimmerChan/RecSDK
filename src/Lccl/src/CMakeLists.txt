set(LCAL_SOURCE_FILE
        lcal_comm.cpp
        tools/socket/lcal_sock_exchange.h
        tools/socket/lcal_sock_exchange.cpp
        ${ATB_LOG_FILES})

add_library(lcal SHARED ${LCAL_SOURCE_FILE})
add_library(lcal_static STATIC ${LCAL_SOURCE_FILE})
set_target_properties(lcal_static PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_link_libraries(lcal ascendcl runtime profapi c_sec)
target_link_libraries(lcal_static ascendcl runtime profapi c_sec)

install(TARGETS lcal LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX})
install(TARGETS lcal_static DESTINATION ${CMAKE_INSTALL_PREFIX})
