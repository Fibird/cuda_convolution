project(conv2D_test)
cmake_minimum_required(VERSION 3.5)

find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED) 


set(MAIN_PROJECT_SOURCE_DIR ${PROJECT_SOURCE_DIR}/../..)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${MAIN_PROJECT_SOURCE_DIR}/bin)

set(GPU_CONVOLUTION_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/../convolution)

include_directories(
${OpenCV_INCLUDE_DIRS}
${GPU_CONVOLUTION_INCLUDE_DIRS}
${CPU_CONVOLUTION_INCLUDE_DIRS}
)

add_executable(cpu_convolution_test cpu_convolution_test.cpp)

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_60,code=sm_60)
cuda_add_executable(gpu_convolution_test gpu_convolution_test.cu)

set(GPU_CONVOLUTION_LIBS ${MAIN_PROJECT_SOURCE_DIR}/bin/lib/convolution/libgpu_convolution.a)
set(CPU_CONVOLUTION_LIBS ${MAIN_PROJECT_SOURCE_DIR}/bin/lib/convolution/libcpu_convolution.a)

target_link_libraries(gpu_convolution_test ${GPU_CONVOLUTION_LIBS} ${OpenCV_LIBS})
target_link_libraries(cpu_convolution_test ${CPU_CONVOLUTION_LIBS} ${OpenCV_LIBS})
