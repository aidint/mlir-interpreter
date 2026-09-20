add_library(InterpreterBuiltin BuiltinAttrEval.cpp BuiltinEvalValues.cpp)

target_link_libraries(InterpreterBuiltin PUBLIC
  InterpreterInterfaces)
