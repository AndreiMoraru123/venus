cmake_minimum_required(VERSION 4.5)
project(venus LANGUAGES CXX)

# Until CMake has a C++26 flag for MSVC, 23 maps to -std:c++latest
if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC" AND CMAKE_CXX_STANDARD_LATEST LESS 26)
    set(CMAKE_CXX_STANDARD 23)
else()
    set(CMAKE_CXX_STANDARD 26)
endif()
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

enable_testing()
include(CTest)
include(FetchContent)

add_library(venus INTERFACE)
target_include_directories(venus INTERFACE
     $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/core>
     $<INSTALL_INTERFACE:include>
)

file(GLOB_RECURSE VENUS_HEADERS CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/core/*.hpp"
)

target_sources(venus INTERFACE
    FILE_SET HEADERS
    BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/core
    FILES ${VENUS_HEADERS}
)

option(VENUS_BUILD_MODULE "Build the venus module and run tests and demos against it" OFF)

if(VENUS_BUILD_MODULE)
    if(NOT CMAKE_GENERATOR MATCHES "Ninja")
        message(FATAL_ERROR "VENUS_BUILD_MODULE needs a Ninja generator")
    endif()

    add_library(venus_module STATIC)
    target_sources(venus_module
        PUBLIC
        FILE_SET CXX_MODULES
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/module
        FILES ${CMAKE_CURRENT_SOURCE_DIR}/module/venus.cppm
    )
    target_link_libraries(venus_module PUBLIC venus)

    # Each venus header, except the policy macros, as a file holding import venus
    set(VENUS_IMPORT_HEADERS_DIR ${CMAKE_CURRENT_BINARY_DIR}/venus_import_headers)
    foreach(header ${VENUS_HEADERS})
        file(RELATIVE_PATH header ${CMAKE_CURRENT_SOURCE_DIR}/core ${header})
        if(NOT header MATCHES "policy_macro_")
            file(CONFIGURE OUTPUT ${VENUS_IMPORT_HEADERS_DIR}/${header}
                CONTENT "#pragma once\nimport venus;\n")
        endif()
    endforeach()
endif()

# Tests and demos include venus headers, which become import venus with the module
function(venus_link_tests_and_demos target)
    if(VENUS_BUILD_MODULE)
        target_link_libraries(${target} PRIVATE venus_module)
        target_include_directories(${target} BEFORE PRIVATE ${VENUS_IMPORT_HEADERS_DIR})
    else()
        target_link_libraries(${target} PRIVATE venus)
    endif()
endfunction()

if(BUILD_TESTING)
    FetchContent_Declare(
        Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG v3.8.0
    )
    FetchContent_MakeAvailable(Catch2)
    add_subdirectory(test)
endif()

option(BUILD_DEMO "Build demo executable" ON)
option(BUILD_SINGLE_HEADER "Build single-header generator tool" OFF)

if(BUILD_DEMO AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/demo")
    add_subdirectory(demo)
endif()

if (BUILD_SINGLE_HEADER AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/scripts")
    add_subdirectory(scripts)
endif()

find_program(RLWRAP_EXECUTABLE rlwrap)
find_program(CLANG_REPL_EXECUTABLE clang-repl)

if(CLANG_REPL_EXECUTABLE)
    if (RLWRAP_EXECUTABLE)
        set(REPL_COMMAND ${RLWRAP_EXECUTABLE} --always-readline ${CLANG_REPL_EXECUTABLE})
    else()
        set(REPL_COMMAND ${CLANG_REPL_EXECUTABLE})
    endif()

    add_custom_target(venus-interactive
        COMMAND ${REPL_COMMAND}
                -Xcc=-std=c++26
                -Xcc=-DVENUS_INTERPRETER
                -Xcc=-include
                -Xcc=${CMAKE_CURRENT_SOURCE_DIR}/single_include/venus.hpp
                -Xcc=-include
                -Xcc=${CMAKE_CURRENT_SOURCE_DIR}/repl/pretty_print.hpp
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        USES_TERMINAL
    )
else()
    message(WARNING "Clang-repl not found - target 'venus-interactive' will not be available")
endif()