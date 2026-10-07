# Runs TOOL with ARGS on INPUT and fails unless it succeeds and prints exactly
# the contents of EXPECTED.
execute_process(COMMAND ${TOOL} ${ARGS} ${INPUT}
  OUTPUT_VARIABLE actual
  RESULT_VARIABLE result)
if(NOT result EQUAL 0)
  message(FATAL_ERROR "${TOOL} failed on ${INPUT}: ${result}")
endif()
file(READ ${EXPECTED} expected)
if(NOT actual STREQUAL expected)
  message(FATAL_ERROR "report differs from ${EXPECTED}:\n${actual}")
endif()
