add_executable(engine-lifetime-test EngineLifetime.cpp)

target_link_libraries(engine-lifetime-test PRIVATE
  Interpreter
  InterpreterArith
  InterpreterBuiltin
  MLIRArithDialect
  MLIRFuncDialect
  MLIRIR
  MLIRParser
  MLIRSupport
  LLVMSupport)

add_executable(eval-cache-test EvalCache.cpp)

target_link_libraries(eval-cache-test PRIVATE
  Interpreter
  InterpreterBuiltin
  MLIRArithDialect
  MLIRIR
  MLIRSupport
  LLVMSupport)

# Configures a lit suite for the tests in `source_dir`. Every suite shares
# `lit.cfg.py`; only its name and roots differ.
function(configure_lit_suite name source_dir binary_dir)
  set(LIT_SUITE_NAME ${name})
  set(LIT_SOURCE_DIR ${source_dir})
  set(LIT_BINARY_DIR ${binary_dir})
  configure_file(lit.site.cfg.py.in ${binary_dir}/lit.site.cfg.py @ONLY)
endfunction()

# LLVM sets its output directories in its own scope, so read them from targets.
# `llvm-lit` is generated next to `FileCheck`.
get_target_property(LLVM_TOOLS_DIR FileCheck RUNTIME_OUTPUT_DIRECTORY)
set(LLVM_LIT_OUTPUT_DIR ${LLVM_TOOLS_DIR})
get_target_property(interpreter_dir interpreter BINARY_DIR)
set(INTERPRETER_TOOLS_DIRS "${interpreter_dir};${CMAKE_CURRENT_BINARY_DIR}")

configure_lit_suite(interpreter
  ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR})
configure_lit_suite(interpreter-examples
  ${PROJECT_SOURCE_DIR}/examples ${PROJECT_BINARY_DIR}/examples)

add_lit_testsuite(check-interpreter "Running the interpreter tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  ${PROJECT_BINARY_DIR}/examples
  DEPENDS FileCheck count not interpreter engine-lifetime-test eval-cache-test)
