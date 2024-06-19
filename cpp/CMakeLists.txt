cmake_minimum_required(VERSION 3.15)
project(a_star CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall -fPIC ${TORCH_CXX_FLAGS} -Xpreprocessor -fopenmp=libomp")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

find_package(pybind11 REQUIRED)
find_package(OpenMP REQUIRED)

# message(STATUS "Using Clang compiler: ${CMAKE_CXX_COMPILER}")

# Check if OpenMP is supported by the compiler
# find_package(OpenMP REQUIRED)
# if(OpenMP_FOUND)
#     message(STATUS "OpenMP found: ${OpenMP_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS} -stdlib=libstdc++")
# else()
#     message(FATAL_ERROR "OpenMP not found")
# endif()


pybind11_add_module(a_star src/main.cpp)

target_include_directories(
    a_star 
    PUBLIC 
    ${Python3_INCLUDE_DIRS}
    # "/opt/homebrew/opt/libomp/include"
)

target_link_libraries(
    a_star PUBLIC 
    ${Python3_LIBRARIES}
    # "/opt/homebrew/opt/libomp/lib/libomp.dylib"
    OpenMP::OpenMP_CXX
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