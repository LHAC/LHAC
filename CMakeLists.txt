cmake_minimum_required(VERSION 3.0)
project(LHAC)

# MESSAGE(STATUS "This is BINARY dir " ${LHAC_BINARY_DIR}) 
# MESSAGE(STATUS "This is SOURCE dir " ${LHAC_SOURCE_DIR})

#There are lots of scripts with cmake
#for finding external libraries. 
#see /usr/local/share/cmake-2.6/Modules/Find*.cmake for more examples
# find_package(GLUT)
# find_package(OpenGL)
# MESSAGE(STATUS "blas vendor: " ${BLA_VENDOR})

# set(CMAKE_CXX_FLAGS "-Wc++11-extensions")
# add_subdirectory(w01-cpp)

#To include self-defined cmake modules
SET(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

include_directories(include)
add_subdirectory(src bin)
