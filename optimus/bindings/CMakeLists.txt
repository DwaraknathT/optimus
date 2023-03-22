pybind11_add_module(pyoptimus MODULE ops.cu)
target_link_libraries(pyoptimus PUBLIC 
                      gemm_kernels 
                      log_utils memanager -lcudart)