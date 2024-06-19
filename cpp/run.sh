#!/bin/bash

set -e
set -x

BASEDIR=$(dirname "$0")
pushd "$BASEDIR"

rm -rf build

conan install . --output-folder=build --build=missing --settings=build_type=Release
cd build
CXX=/opt/homebrew/Cellar/gcc/13.2.0/bin/g++-13 cmake .. -DCMAKE_TOOLCHAIN_FILE=./build/Release/generators/conan_toolchain.cmake -DCMAKE_BUILD_TYPE=Release 
    # -DOpenMP_C_FLAGS=-fopenmp=lomp \
    # -DOpenMP_CXX_FLAGS=-fopenmp=lomp \
    # -DOpenMP_C_LIB_NAMES="libomp" \
    # -DOpenMP_CXX_LIB_NAMES="libomp" \
    # -DOpenMP_libomp_LIBRARY="/opt/homebrew/opt/libomp/lib/libomp.dylib" \
    # -DOpenMP_CXX_FLAGS="-Xpreprocessor -fopenmp /opt/homebrew/opt/libomp/lib/libomp.dylib -I/opt/homebrew/opt/libomp/include" \
    # -DOpenMP_CXX_LIB_NAMES="libomp" \
    # -DOpenMP_C_FLAGS="-Xpreprocessor -fopenmp /opt/homebrew/opt/libomp/lib/libomp.dylib -I/opt/homebrew/opt/libomp/include"

cmake --build .
cd ..
python test_a_star.py
# ./a_star
