cmake_minimum_required(VERSION 3.25.1)
project(venus LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
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
                -Xcc=-std=c++23
                -Xcc=-I${CMAKE_CURRENT_SOURCE_DIR}/core
                -Xcc=-I${CMAKE_CURRENT_SOURCE_DIR}
                -Xcc=-DVENUS_INTERPRETER
                -Xcc=-include
                -Xcc=${CMAKE_CURRENT_SOURCE_DIR}/single_include/venus.hpp
                -Xcc=-include
                -Xcc=${CMAKE_CURRENT_SOURCE_DIR}/repl/pretty_print.hpp
        DEPENDS venus
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        USES_TERMINAL
    )
else()
    message(WARNING "Clang-repl not found - target 'venus-interactive' will not be available")
endif()