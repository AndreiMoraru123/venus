include(Catch)

add_executable(test_sequential test_sequential.cpp)
target_link_libraries(test_sequential
    PRIVATE
    venus
    Catch2::Catch2WithMain
)

add_executable(test_var_type_dict test_var_type_dict.cpp)
target_link_libraries(test_var_type_dict
    PRIVATE
    venus
    Catch2::Catch2WithMain
)

catch_discover_tests(test_sequential)
catch_discover_tests(test_var_type_dict)

add_executable(all_tests test_sequential.cpp test_var_type_dict.cpp)
target_link_libraries(all_tests
    PRIVATE
    venus
    Catch2::Catch2WithMain
)
catch_discover_tests(all_tests)
