find_package(Python3 REQUIRED COMPONENTS Interpreter)
add_test(NAME pgorca_ml_orca_test
    COMMAND ${Python3_EXECUTABLE} -B -m unittest discover -s ml_orca/tests -t .)
set_tests_properties(pgorca_ml_orca_test PROPERTIES
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    LABELS "ml-orca")
