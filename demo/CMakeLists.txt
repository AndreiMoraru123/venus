file(GLOB_RECURSE DEMO_SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

foreach(demo_source ${DEMO_SOURCES})
    get_filename_component(demo_name ${demo_source} NAME_WE)
    set(target_name "demo_${demo_name}")
    add_executable(${target_name} ${demo_source})
    target_link_libraries(${target_name} PRIVATE venus)
    target_include_directories(${target_name} PRIVATE
        ${CMAKE_SOURCE_DIR}
        ${mdspan_SOURCE_DIR}/include
    )
    set_target_properties(${target_name} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/demos"
    )
endforeach()