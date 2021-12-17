cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# set the project name
project(MainNew)

include_directories(..)
add_executable(MainNew test_extra.cpp)