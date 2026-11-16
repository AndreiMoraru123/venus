include(CTest)
include(Catch)

file(GLOB_RECURSE TEST_SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

add_executable(all_tests ${TEST_SOURCES})
target_link_libraries(all_tests
    PRIVATE
    venus
    Catch2::Catch2WithMain
)

if(VENUS_BUILD_MODULE)
    target_link_libraries(all_tests PRIVATE venus_module)
    target_compile_definitions(all_tests PRIVATE VENUS_IMPORT)
    set_target_properties(all_tests PROPERTIES CXX_SCAN_FOR_MODULES ON)
endif()

catch_discover_tests(all_tests)
