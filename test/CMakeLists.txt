include(CTest)
include(Catch)

file(GLOB_RECURSE TEST_SOURCES CONFIGURE_DEPENDS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
# Each file in module/ imports venus itself and is its own executable
list(FILTER TEST_SOURCES EXCLUDE REGEX "^module/")

add_executable(all_tests ${TEST_SOURCES})
target_link_libraries(all_tests
    PRIVATE
    Catch2::Catch2WithMain
)

if(VENUS_BUILD_MODULE)
    target_link_libraries(all_tests PRIVATE venus_module)
    target_compile_definitions(all_tests PRIVATE VENUS_IMPORT)

    add_executable(module_print_tensor module/test_print_tensor.cpp module/format_tensors.cpp)
    target_link_libraries(module_print_tensor PRIVATE venus_module Catch2::Catch2WithMain)
    catch_discover_tests(module_print_tensor)
else()
    target_link_libraries(all_tests PRIVATE venus)
endif()

catch_discover_tests(all_tests)
