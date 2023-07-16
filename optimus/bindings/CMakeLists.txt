pybind11_add_module(pyoptimus MODULE ops.cu)
target_link_libraries(pyoptimus PUBLIC 
                      gemm 
                      cuda_utils
                      log_utils memanager -lcudart)