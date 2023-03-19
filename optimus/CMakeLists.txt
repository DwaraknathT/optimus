add_subdirectory(utils)
add_subdirectory(kernels)
add_subdirectory(bindings)

add_library(tensor SHARED tensor.cc)
target_link_libraries(tensor PUBLIC -lcudart -lcurand -lcudart)
set_property(TARGET tensor PROPERTY POSITION_INDEPENDENT_CODE  ON)