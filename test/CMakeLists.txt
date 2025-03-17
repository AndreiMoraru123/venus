include(Catch)

file(GLOB_RECURSE TEST_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

add_executable(all_tests ${TEST_SOURCES})
target_link_libraries(all_tests
    PRIVATE
    venus
    Catch2::Catch2WithMain
)

catch_discover_tests(all_tests)
