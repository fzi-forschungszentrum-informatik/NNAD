cmake_minimum_required(VERSION 3.8)
project(DatasetOp)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/cmake)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(JsonCpp REQUIRED)
find_package(Boost COMPONENTS filesystem system REQUIRED)
find_package(OpenCV COMPONENTS highgui imgproc core imgcodecs REQUIRED)
find_package(Threads REQUIRED)

include_directories(${JSONCPP_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})
add_library(dataset
    src/bdd100k_dataset.cc
    src/cityscapes_dataset.cc
    src/flyingchairs_dataset.cc
    src/kitti_dataset.cc
    src/sintel_dataset.cc
    src/webvision_dataset.cc
    src/folder_dataset.cc
    src/file_dataset.cc
    src/dataset.cc
    src/augment.cc
    src/bbutils.cc
    src/utils.cc)
target_link_libraries(dataset ${JSONCPP_LIBRARIES} ${Boost_LIBRARIES} ${OpenCV_LIBRARIES} Threads::Threads)

find_package(Tensorflow REQUIRED)
find_package(yaml-cpp REQUIRED)

include_directories(${TENSORFLOW_INCLUDE_DIRS} ${YAML_CPP_INCLUDE_DIR})
add_library(dataset_op SHARED src/dataset_op.cc)
target_link_libraries(dataset_op dataset ${TENSORFLOW_LIBRARIES} ${YAML_CPP_LIBRARIES})

find_package(pybind11 REQUIRED)
pybind11_add_module(bbutils src/bbutils_binding.cc)
target_link_libraries(bbutils PUBLIC dataset)

