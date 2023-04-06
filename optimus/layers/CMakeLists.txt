add_library(dense_layer SHARED dense.cc)
set_property(TARGET dense_layer PROPERTY POSITION_INDEPENDENT_CODE  ON)
target_link_libraries(dense_layer PUBLIC memanager affine_transforms)