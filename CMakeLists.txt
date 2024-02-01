cmake_minimum_required(VERSION 3.10)
project(centerpoint)

add_definitions(-std=c++11)
add_definitions(-w)
# option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
find_package(CUDA REQUIRED)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Release)


include_directories(/usr/local/cuda-11.1/targets/x86_64-linux/include)
link_directories(/usr/local/cuda-11.1/targets/x86_64-linux/lib)

include_directories(/home/zichen/TensorRT-8.4.1.5/include/)
link_directories(/home/zichen/TensorRT-8.4.1.5/lib/)
file(GLOB_RECURSE SOURCE_FILES
  centerpoint/*.cu
  centerpoint/*.cc
)

cuda_add_executable(${PROJECT_NAME} demo.cc ${SOURCE_FILES})

target_link_libraries(${PROJECT_NAME} nvinfer cudart nvonnxparser nvinfer_plugin yaml-cpp)
add_definitions(-O2 -pthread)