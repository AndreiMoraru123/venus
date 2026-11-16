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

option(VENUS_BUILD_MODULE "Build the venus module and run tests and demos against it" ON)

if(VENUS_BUILD_MODULE)
    # CMake's reason import std is unavailable here, empty when it works
    if(CMAKE_CXX_COMPILER_IMPORT_STD_ERROR_MESSAGE)
        message(FATAL_ERROR
            "The venus module needs import std: ${CMAKE_CXX_COMPILER_IMPORT_STD_ERROR_MESSAGE}. "
            "Configure with -DVENUS_BUILD_MODULE=OFF to build against the headers.")
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

if(BUILD_DEMO AND NOT VENUS_BUILD_MODULE)
    message(STATUS "Skipping the demos, which import venus")
elseif(BUILD_DEMO AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/demo")
    add_subdirectory(demo)
endif()

if (BUILD_SINGLE_HEADER AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/scripts")
    add_subdirectory(scripts)
endif()

find_program(RLWRAP_EXECUTABLE rlwrap)
cmake_path(GET CMAKE_CXX_COMPILER PARENT_PATH CXX_COMPILER_DIR)
find_program(CLANG_REPL_EXECUTABLE clang-repl HINTS ${CXX_COMPILER_DIR} NO_DEFAULT_PATH)
find_program(CLANG_REPL_EXECUTABLE clang-repl)

if(CLANG_REPL_EXECUTABLE)
    if (RLWRAP_EXECUTABLE)
        set(REPL_COMMAND ${RLWRAP_EXECUTABLE} --always-readline ${CLANG_REPL_EXECUTABLE})
    else()
        set(REPL_COMMAND ${CLANG_REPL_EXECUTABLE})
    endif()

    separate_arguments(REPL_FLAGS NATIVE_COMMAND "${CMAKE_CXX_FLAGS}")
    list(APPEND REPL_FLAGS -std=c++26 -DVENUS_INTERPRETER)

    if(VENUS_BUILD_MODULE AND EXISTS "${CMAKE_CXX_STDLIB_MODULES_JSON}")
        # clang-repl cannot read CMake's module map, so it gets its own BMIs
        file(READ ${CMAKE_CXX_STDLIB_MODULES_JSON} modules_json)
        string(JSON std_source GET ${modules_json} modules 0 source-path)
        cmake_path(GET CMAKE_CXX_STDLIB_MODULES_JSON PARENT_PATH modules_json_dir)
        cmake_path(ABSOLUTE_PATH std_source BASE_DIRECTORY ${modules_json_dir})
        set(PCM_DIR ${CMAKE_CURRENT_BINARY_DIR}/repl)
        add_custom_command(
            OUTPUT ${PCM_DIR}/std.pcm ${PCM_DIR}/venus.pcm
            COMMAND ${CMAKE_CXX_COMPILER} ${REPL_FLAGS} -Wno-reserved-module-identifier
                    --precompile ${std_source} -o ${PCM_DIR}/std.pcm
            COMMAND ${CMAKE_CXX_COMPILER} ${REPL_FLAGS} -I${CMAKE_CURRENT_SOURCE_DIR}/core
                    -fmodule-file=std=${PCM_DIR}/std.pcm
                    --precompile ${CMAKE_CURRENT_SOURCE_DIR}/module/venus.cppm -o ${PCM_DIR}/venus.pcm
            DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/module/venus.cppm ${VENUS_HEADERS}
        )
        set(REPL_INPUTS -fmodule-file=std=${PCM_DIR}/std.pcm -fmodule-file=venus=${PCM_DIR}/venus.pcm)
        set(REPL_DEPENDS ${PCM_DIR}/venus.pcm)
    else()
        set(REPL_INPUTS
            -include ${CMAKE_CURRENT_SOURCE_DIR}/single_include/venus.hpp
            -include ${CMAKE_CURRENT_SOURCE_DIR}/repl/pretty_print.hpp)
    endif()

    list(TRANSFORM REPL_FLAGS PREPEND "-Xcc=")
    list(TRANSFORM REPL_INPUTS PREPEND "-Xcc=")
    add_custom_target(venus-interactive
        COMMAND ${REPL_COMMAND} ${REPL_FLAGS} ${REPL_INPUTS}
        DEPENDS ${REPL_DEPENDS}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        USES_TERMINAL
    )
else()
    message(WARNING "Clang-repl not found - target 'venus-interactive' will not be available")
endif()
