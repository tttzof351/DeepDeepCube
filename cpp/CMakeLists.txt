# set(CMAKE_CXX_COMPILER "/opt/homebrew/Cellar/gcc/13.2.0/bin/g++-13")

cmake_minimum_required(VERSION 3.15)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(a_star CXX)

find_package(pybind11 REQUIRED)
# find_package(OpenMP REQUIRED)

pybind11_add_module(a_star src/main.cpp)

# include_directories(/opt/homebrew/include)

target_include_directories(
    a_star 
    PUBLIC 
    ${Python3_INCLUDE_DIRS}
    "/opt/homebrew/opt/libomp/include"
)

target_link_libraries(
    a_star PUBLIC 
    ${Python3_LIBRARIES}
    "/opt/homebrew/opt/libomp/lib/libomp.dylib"
    # OpenMP::OpenMP
    pybind11::module
)

# include_directories(/opt/homebrew/include)
# add_executable(${PROJECT_NAME} src/utils.h src/main.cpp)
# target_link_libraries(${PROJECT_NAME} OpenMP::OpenMP_CXX)

# cmake_minimum_required(VERSION 3.4...3.18)
# project(a_star)

# add_subdirectory(pybind11)
# pybind11_add_module(a_star src/main.cpp)

# # EXAMPLE_VERSION_INFO is defined by setup.py and passed into the C++ code as a
# # define (VERSION_INFO) here.
# target_compile_definitions(
#     a_star
#     PRIVATE VERSION_INFO=1
# )