cmake_minimum_required(VERSION 3.13)
project(pisad_bridge)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(PkgConfig REQUIRED)

# Find jsoncpp
pkg_check_modules(JSONCPP REQUIRED jsoncpp)
if(NOT JSONCPP_FOUND)
    # Fallback to manual linking
    set(JSONCPP_LIBRARIES jsoncpp)
endif()

# Plugin source files
set(SOURCES
    pisad_bridge.cpp
)

set(HEADERS
    pisad_bridge.h
)

# Create the plugin as a shared library
add_library(pisad_bridge SHARED ${SOURCES} ${HEADERS})

# Set output name and properties
set_target_properties(pisad_bridge PROPERTIES
    PREFIX ""
    SUFFIX ".so"
    OUTPUT_NAME "pisad_bridge"
)

# Include directories for SDR++
target_include_directories(pisad_bridge PRIVATE
    # SDR++ core includes - adjust paths as needed
    "${SDRPP_INCLUDE_DIR}"
    "${SDRPP_INCLUDE_DIR}/core"
    "${SDRPP_INCLUDE_DIR}/misc_modules"
    # System includes
    /usr/include
    /usr/local/include
)

# Link libraries
target_link_libraries(pisad_bridge PRIVATE
    ${JSONCPP_LIBRARIES}
    pthread
)

# Compiler flags
target_compile_options(pisad_bridge PRIVATE
    -Wall
    -Wextra
    -O3
    -fPIC
)

# Define plugin metadata
target_compile_definitions(pisad_bridge PRIVATE
    SDRPP_MODULE_NAME="pisad_bridge"
    SDRPP_MODULE_VERSION_MAJOR=1
    SDRPP_MODULE_VERSION_MINOR=0
    SDRPP_MODULE_VERSION_PATCH=0
)

# Installation rules (adjust install path as needed)
if(DEFINED SDRPP_MODULE_DIR)
    install(TARGETS pisad_bridge
        LIBRARY DESTINATION ${SDRPP_MODULE_DIR}
    )
else()
    install(TARGETS pisad_bridge
        LIBRARY DESTINATION lib/sdrpp/plugins
    )
endif()

# Create a test executable for development
add_executable(test_pisad_plugin test_main.cpp)
target_include_directories(test_pisad_plugin PRIVATE
    ${JSONCPP_INCLUDE_DIRS}
)
target_link_libraries(test_pisad_plugin PRIVATE
    ${JSONCPP_LIBRARIES}
    pthread
)

# Development convenience targets
add_custom_target(install_plugin
    COMMAND ${CMAKE_COMMAND} --build . --target install
    DEPENDS pisad_bridge
    COMMENT "Installing PISAD bridge plugin"
)

add_custom_target(uninstall_plugin
    COMMAND rm -f ${CMAKE_INSTALL_PREFIX}/lib/sdrpp/plugins/pisad_bridge.so
    COMMENT "Uninstalling PISAD bridge plugin"
)
