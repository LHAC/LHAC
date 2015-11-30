cmake_minimum_required(VERSION 3.0)
project(LHAC)

#To include self-defined cmake modules
SET(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

include_directories(include)
add_subdirectory(src bin)
