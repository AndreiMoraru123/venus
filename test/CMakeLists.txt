include(CTest)
include(Catch)

file(GLOB_RECURSE TEST_SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

add_executable(all_tests ${TEST_SOURCES})
target_link_libraries(all_tests
    PRIVATE
    Catch2::Catch2WithMain
)

# The tests include venus headers, which become import venus with the module
if(VENUS_BUILD_MODULE)
    target_link_libraries(all_tests PRIVATE venus_module)
    target_include_directories(all_tests BEFORE PRIVATE ${VENUS_IMPORT_HEADERS_DIR})
else()
    target_link_libraries(all_tests PRIVATE venus)
endif()

catch_discover_tests(all_tests)
