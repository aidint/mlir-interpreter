# Builds LLVM and MLIR from the third_party/llvm-project submodule as part of
# this CMake project (no installed LLVM/MLIR is used).
#
# LLVM is added with EXCLUDE_FROM_ALL, so only the libraries (and tablegen
# tools) our targets actually link against get built.
#
# Provides the INTERFACE target `mlir-submodule`, which carries the include
# directories and compile flags needed to use MLIR headers. Link it alongside
# the MLIR libraries you need, e.g.:
#
#   target_link_libraries(foo PRIVATE mlir-submodule MLIRIR MLIRParser)

set(LLVM_PROJECT_SOURCE_DIR "${PROJECT_SOURCE_DIR}/third_party/llvm-project")
set(LLVM_PROJECT_BINARY_DIR "${PROJECT_BINARY_DIR}/third_party/llvm-project")

if(NOT EXISTS "${LLVM_PROJECT_SOURCE_DIR}/llvm/CMakeLists.txt")
  message(FATAL_ERROR
    "llvm-project submodule is missing. Run:\n"
    "  git submodule update --init --depth 1 third_party/llvm-project")
endif()

# Emscripten cross-builds run TableGen from the native release build and build
# LLVM without target backends or threads. Must come before the defaults below,
# which don't override existing entries.
if(EMSCRIPTEN)
  set(LLVM_NATIVE_TOOL_DIR
    "${PROJECT_SOURCE_DIR}/build/release/third_party/llvm-project/llvm/bin"
    CACHE PATH "Directory with native llvm-min-tblgen and mlir-tblgen")
  foreach(tool llvm-min-tblgen mlir-tblgen)
    if(NOT EXISTS "${LLVM_NATIVE_TOOL_DIR}/${tool}")
      message(FATAL_ERROR
        "${tool} is missing from LLVM_NATIVE_TOOL_DIR. Build it natively:\n"
        "  cmake --preset release\n"
        "  cmake --build --preset release")
    endif()
  endforeach()
  set(LLVM_HOST_TRIPLE "wasm32-unknown-emscripten" CACHE STRING "")
  set(LLVM_TARGETS_TO_BUILD "" CACHE STRING "")
  set(LLVM_ENABLE_THREADS OFF CACHE BOOL "")
  set(LLVM_ENABLE_PIC OFF CACHE BOOL "")
  set(LLVM_ENABLE_BACKTRACES OFF CACHE BOOL "")
  set(LLVM_ENABLE_CRASH_OVERRIDES OFF CACHE BOOL "")
  set(LLVM_ENABLE_UNWIND_TABLES OFF CACHE BOOL "")
endif()

# Defaults for the LLVM build. These are plain cache entries (not FORCE), so
# they can still be overridden from the command line with -D.
set(LLVM_ENABLE_PROJECTS "mlir" CACHE STRING "")
set(LLVM_TARGETS_TO_BUILD "host" CACHE STRING "")
set(LLVM_ENABLE_ASSERTIONS ON CACHE BOOL "")
set(LLVM_ENABLE_RTTI OFF CACHE BOOL "")
set(LLVM_INCLUDE_TESTS OFF CACHE BOOL "")
set(LLVM_INCLUDE_EXAMPLES OFF CACHE BOOL "")
set(LLVM_INCLUDE_BENCHMARKS OFF CACHE BOOL "")
set(LLVM_INCLUDE_DOCS OFF CACHE BOOL "")
set(LLVM_BUILD_TOOLS OFF CACHE BOOL "")
set(LLVM_BUILD_UTILS OFF CACHE BOOL "")
set(LLVM_ENABLE_BINDINGS OFF CACHE BOOL "")
set(LLVM_ENABLE_ZLIB OFF CACHE STRING "")
set(LLVM_ENABLE_ZSTD OFF CACHE STRING "")
set(LLVM_ENABLE_LIBXML2 OFF CACHE STRING "")
set(LLVM_ENABLE_LIBEDIT OFF CACHE BOOL "")
set(LLVM_ENABLE_TERMINFO OFF CACHE BOOL "")
set(MLIR_INCLUDE_TESTS OFF CACHE BOOL "")
set(MLIR_INCLUDE_INTEGRATION_TESTS OFF CACHE BOOL "")
set(MLIR_ENABLE_BINDINGS_PYTHON OFF CACHE BOOL "")

# LLVM uses precompiled headers by default. clangd can't consume PCH files
# (and fails outright if clangd's version differs from the compiler's), and
# they defeat ccache, so turn them off.
set(CMAKE_DISABLE_PRECOMPILE_HEADERS ON CACHE BOOL "")

# Silence a CMake >= 4.3 policy warning triggered inside LLVM's own CMake code.
# OLD is the behavior LLVM expects.
set(CMAKE_POLICY_DEFAULT_CMP0219 OLD)

add_subdirectory(
  "${LLVM_PROJECT_SOURCE_DIR}/llvm"
  "${LLVM_PROJECT_BINARY_DIR}/llvm"
  EXCLUDE_FROM_ALL)

# LLVM/MLIR set their include directories and flags with directory-scoped
# commands, which don't propagate back up to us. Re-export what consumers need.
add_library(mlir-submodule INTERFACE)
target_include_directories(mlir-submodule SYSTEM INTERFACE
  "${LLVM_PROJECT_SOURCE_DIR}/llvm/include"
  "${LLVM_PROJECT_BINARY_DIR}/llvm/include"
  "${LLVM_PROJECT_SOURCE_DIR}/mlir/include"
  "${LLVM_PROJECT_BINARY_DIR}/llvm/tools/mlir/include")
if(NOT LLVM_ENABLE_RTTI)
  target_compile_options(mlir-submodule INTERFACE -fno-rtti)
endif()
if(LLVM_ENABLE_ASSERTIONS)
  # Keep NDEBUG consistent with how LLVM/MLIR were compiled.
  target_compile_options(mlir-submodule INTERFACE -UNDEBUG)
endif()
