cmake_minimum_required(VERSION 3.31.3)
project(venus LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(FetchContent)

FetchContent_Declare(
    flux
    GIT_REPOSITORY https://github.com/tcbrindle/flux.git
    GIT_TAG main
)

FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v3.8.0
)

FetchContent_MakeAvailable(flux Catch2)

add_executable(tests test_sequential.cpp)
target_link_libraries(tests
    PRIVATE
    flux::flux
    Catch2::Catch2WithMain
)
target_include_directories(tests PUBLIC ${flux_SOURCE_DIR}/include)