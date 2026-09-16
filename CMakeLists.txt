cmake_minimum_required(VERSION 3.29)

# Default to clang/clang++ unless a compiler was chosen explicitly (e.g. via a
# preset or -DCMAKE_CXX_COMPILER=...). Must happen before project().
if(NOT DEFINED CMAKE_C_COMPILER AND NOT DEFINED ENV{CC})
  set(CMAKE_C_COMPILER clang)
endif()
if(NOT DEFINED CMAKE_CXX_COMPILER AND NOT DEFINED ENV{CXX})
  set(CMAKE_CXX_COMPILER clang++)
endif()

project(mlir-interpreter LANGUAGES C CXX)

if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  message(WARNING "This project is meant to be built with clang++, "
                  "but the C++ compiler is ${CMAKE_CXX_COMPILER_ID}.")
endif()

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# LLVM + MLIR from the third_party/llvm-project submodule.
include(cmake/LLVMSubmodule.cmake)

add_subdirectory(lib)
add_subdirectory(tools)

# Expose compile_commands.json at the repo root for clangd.
if(PROJECT_IS_TOP_LEVEL)
  file(CREATE_LINK
    "${CMAKE_BINARY_DIR}/compile_commands.json"
    "${CMAKE_SOURCE_DIR}/compile_commands.json"
    SYMBOLIC)
endif()
