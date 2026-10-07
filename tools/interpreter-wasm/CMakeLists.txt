# Builds the shared runner as an Emscripten module, `interpreter.{js,wasm}`.
# `showcase.sh` packages it with the page in `web/`.
add_executable(interpreter-wasm InterpreterWasm.cpp)

set_target_properties(interpreter-wasm PROPERTIES OUTPUT_NAME interpreter)

target_link_libraries(interpreter-wasm PRIVATE InterpreterRunner)

# The module is single-threaded and is instantiated by a factory, so a worker
# can load it once and call the narrow C exports for every run. Node is a
# supported environment so the parity check can load the same files.
target_link_options(interpreter-wasm PRIVATE
  -sMODULARIZE
  -sEXPORT_NAME=createInterpreter
  -sENVIRONMENT=web,worker,node
  -sALLOW_MEMORY_GROWTH
  -sSTACK_SIZE=4MB
  -sEXPORTED_FUNCTIONS=_malloc,_free
  -sEXPORTED_RUNTIME_METHODS=stringToNewUTF8,UTF8ToString)
