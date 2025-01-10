cmake_minimum_required(VERSION 3.31.3)
project(cpp23_test LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

add_executable(main main.cpp)
