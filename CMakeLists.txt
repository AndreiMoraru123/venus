cmake_minimum_required(VERSION 3.31.3)
project(venus LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

add_executable(main main.cxx)

target_sources(main PRIVATE FILE_SET cxx_modules TYPE CXX_MODULES FILES vector.ixx)
target_sources(main PRIVATE vector.ixx)
