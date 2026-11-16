include(CTest)
include(Catch)

file(GLOB_RECURSE TEST_SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

add_executable(all_tests ${TEST_SOURCES})
target_link_libraries(all_tests
    PRIVATE
    Catch2::Catch2WithMain
)
venus_link_tests_and_demos(all_tests)

catch_discover_tests(all_tests)
