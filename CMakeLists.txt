cmake_minimum_required(VERSION 3.25.1)
project(venus LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

enable_testing()
include(CTest)
include(FetchContent)

FetchContent_Declare(
    mdspan
    GIT_REPOSITORY https://github.com/kokkos/mdspan
    GIT_TAG stable
)

FetchContent_MakeAvailable(mdspan)

FetchContent_Declare(
    range-v3
    GIT_REPOSITORY https://github.com/ericniebler/range-v3.git
    GIT_TAG 0.12.0
)

FetchContent_MakeAvailable(range-v3)

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

target_link_libraries(venus INTERFACE
    mdspan
    range-v3::range-v3
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

if(BUILD_DEMO AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/demo")
    add_subdirectory(demo)
endif()

find_program(CLING_EXECUTABLE cling PATHS /opt/cling/bin)

if(CLING_EXECUTABLE)
    message(STATUS "Found Cling: ${CLING_EXECUTABLE}")
    
    add_custom_target(cling-interactive
        COMMAND ${CLING_EXECUTABLE} -std=c++2a 
                -DVENUS_INTERPRETER
                -I ${CMAKE_BINARY_DIR}/_deps/mdspan-src/include
                -I ${CMAKE_BINARY_DIR}/_deps/range-v3-src/include
        DEPENDS venus
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMENT "Starting interactive Cling session with Venus dependencies"
        USES_TERMINAL
    )
else()
    message(WARNING "Cling not found - target 'cling-interactive' will not be available")
endif()
