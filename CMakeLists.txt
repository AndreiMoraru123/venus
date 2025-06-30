cmake_minimum_required(VERSION 3.25.1)
project(venus LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

enable_testing()
include(CTest)
include(FetchContent)

FetchContent_Declare(
    flux
    GIT_REPOSITORY https://github.com/tcbrindle/flux.git
    GIT_TAG main
)

FetchContent_Declare(
    mdspan
    GIT_REPOSITORY https://github.com/kokkos/mdspan
    GIT_TAG stable
)

FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v3.8.0
)

FetchContent_MakeAvailable(flux mdspan Catch2)

add_library(venus INTERFACE)
target_include_directories(venus INTERFACE
     $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
     $<INSTALL_INTERFACE:include>
     $<BUILD_INTERFACE:${mdspan_SOURCE_DIR}/include>)

target_link_libraries(venus INTERFACE flux::flux)
add_subdirectory(test)

add_executable(demo main.cpp)
target_link_libraries(demo PRIVATE venus)
target_include_directories(demo PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${mdspan_SOURCE_DIR}/include
)

