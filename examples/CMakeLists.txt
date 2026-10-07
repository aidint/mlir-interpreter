# Checks that the interpreter prints `<name>.expected` for `<name>.mlir`. Extra
# arguments are passed to the interpreter.
function(add_example name)
  add_test(NAME example-${name}
    COMMAND ${CMAKE_COMMAND}
      -DTOOL=$<TARGET_FILE:interpreter>
      -DINPUT=${CMAKE_CURRENT_SOURCE_DIR}/${name}.mlir
      -DEXPECTED=${CMAKE_CURRENT_SOURCE_DIR}/${name}.expected
      "-DARGS=${ARGN}"
      -P ${PROJECT_SOURCE_DIR}/cmake/CheckExample.cmake)
endfunction()

add_example(01-basic)
add_example(02-equal-values)
add_example(03-swapped-operands)
add_example(04-key-changes)
add_example(05-zero-product --budget=2)
add_example(06-unknown-and-budget --budget=3)
