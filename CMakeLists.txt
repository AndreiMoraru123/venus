cmake_minimum_required(VERSION 3.31.3)
project(vypr LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(FetchContent)

FetchContent_Declare(
    flux
    GIT_REPOSITORY https://github.com/tcbrindle/flux.git
    GIT_TAG main
)

FetchContent_MakeAvailable(flux)

add_executable(main main.cpp)
target_link_libraries(main PUBLIC flux::flux)
target_include_directories(main PUBLIC ${flux_SOURCE_DIR}/include)