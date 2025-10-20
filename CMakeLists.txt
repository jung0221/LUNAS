cmake_minimum_required(VERSION 3.15)
project(ctsegmentation LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(BUILD_SHARED_LIBS "Build shared libraries" OFF)

find_package(ZLIB)
find_package(OpenMP QUIET)

add_subdirectory(lib)
