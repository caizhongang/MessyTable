cmake_minimum_required(VERSION 2.8.12)
project(wrapper)

set(CMAKE_CXX_FLAGS "-std=c++11")
#set(PYBIND11_PYTHON_VERSION "3.5")
#set(PYTHON_EXECUTABLE "/home/alex/sensetime/bin/python3.5")
set(PYTHON_EXECUTABLE "/home/alex/anaconda3/envs/env/bin/python")

find_package (Eigen3 REQUIRED NO_MODULE)
include_directories(${EIGEN3_INCLUDE_DIRS})

include_directories(src)
#include_directories(/usr/include/python3.5)
#include_directories(/usr/include/python3.6)
include_directories(/home/alex/anaconda3/envs/env/include/python3.7m)

add_subdirectory(pybind11)
pybind11_add_module(mytest tests/mytest.cpp)
pybind11_add_module(KMSolver src/KMSolver.cpp src/hungarian.cpp)
# do not build hungarian as a library outside and link
# instead, use pybind11 to build it together with the module
