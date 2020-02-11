#!/bin/bash

DIR=`dirname "${BASH_SOURCE[0]}"`
rm -Rf "${DIR}"/build
mkdir "${DIR}"/build
pushd "${DIR}"/build
cmake -DCMAKE_CXX_FLAGS="-Wall" -DCMAKE_BUILD_TYPE=Release ..
make -j
popd

