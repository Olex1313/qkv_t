cmake_minimum_required(VERSION 3.16)
project(bench_kernels CXX)
set(CMAKE_CXX_STANDARD 20)

set(MNN_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../contrib/MNN-simiyutin" CACHE PATH "MNN source root")

add_executable(benchKernels benchKernels.cpp)

find_package(OpenCL QUIET)
if(OpenCL_FOUND)
    target_compile_definitions(benchKernels PRIVATE BENCH_OCL=1)
    target_link_libraries(benchKernels OpenCL::OpenCL)
else()
    message(STATUS "OpenCL not found — OCL bench disabled")
endif()

# Vulkan
find_package(Vulkan QUIET)
if(Vulkan_FOUND)
    target_compile_definitions(benchKernels PRIVATE BENCH_VK=1)
    target_link_libraries(benchKernels Vulkan::Vulkan)
else()
    message(STATUS "Vulkan not found — VK bench disabled")
endif()

target_compile_definitions(benchKernels PRIVATE
    OCL_KERNEL_PATH="${MNN_ROOT}/../../opencl-flash/qkvt/kernels/flash_attn_v2_mnn.cl"
    VK_SPV_DIR="${MNN_ROOT}/source/backend/vulkan/buffer/compiler/.cache/shader/cache/"
)

find_program(GLSLANG glslangValidator
    HINTS "$ENV{VULKAN_SDK}/bin" "$ENV{HOME}/VulkanSDK/1.4.341.1/macOS/bin")
if(GLSLANG)
    target_compile_definitions(benchKernels PRIVATE GLSLANG_PATH="${GLSLANG}")
    message(STATUS "glslangValidator: ${GLSLANG}")
else()
    message(WARNING "glslangValidator not found — Vulkan shader compile will use PATH")
endif()
