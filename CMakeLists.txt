cmake_minimum_required(VERSION 3.20)

project(ThermalDriftEdgeAI LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(ONNXRUNTIME_ROOT
    "${CMAKE_SOURCE_DIR}/third_party/onnxruntime-win-x64-1.31.0")

if(NOT EXISTS "${ONNXRUNTIME_ROOT}/include/onnxruntime_cxx_api.h")
    message(FATAL_ERROR "ONNX Runtime C++ headers not found")
endif()

if(NOT EXISTS "${ONNXRUNTIME_ROOT}/lib/onnxruntime.lib")
    message(FATAL_ERROR "ONNX Runtime import library not found")
endif()

add_executable(edge_ai_cpp cpp/main.cpp)

target_include_directories(edge_ai_cpp PRIVATE
    "${ONNXRUNTIME_ROOT}/include")

target_link_libraries(edge_ai_cpp PRIVATE
    "${ONNXRUNTIME_ROOT}/lib/onnxruntime.lib")

foreach(RUNTIME_DLL
    onnxruntime.dll
    onnxruntime_providers_shared.dll)
    if(EXISTS "${ONNXRUNTIME_ROOT}/lib/${RUNTIME_DLL}")
        add_custom_command(TARGET edge_ai_cpp POST_BUILD
            COMMAND "${CMAKE_COMMAND}" -E copy_if_different
                "${ONNXRUNTIME_ROOT}/lib/${RUNTIME_DLL}"
                "$<TARGET_FILE_DIR:edge_ai_cpp>"
            VERBATIM)
    endif()
endforeach()
