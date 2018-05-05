find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

set(WAVE_FORMAT_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/src/waveformat)
message(${WAVE_FORMAT_INCLUDE_DIRS})

include_directories(
${WAVE_FORMAT_INCLUDE_DIRS}
${OpenCV_INCLUDE_DIRS}
)

set(WAVE_FORMAT_LIBS ${PROJECT_SOURCE_DIRS}/bin/lib/libwaveformat.so)

add_library(cpu_convolution STATIC cpu_convolution.cpp)

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_60,code=sm_60)
cuda_add_library(gpu_convolution STATIC gpu_convolution.cu)

target_link_libraries(gpu_convolution ${OpenCV_LIBS} ${WAVE_FORMAT_LIBS})
target_link_libraries(cpu_convolution ${OpenCV_LIBS} ${WAVE_FORMAT_LIBS})
