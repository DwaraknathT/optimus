add_executable(test test.cu)
target_link_libraries(test PUBLIC memanager gemm_kernels)
target_link_libraries(test PUBLIC -lcublas -lcublasLt -lcudart -lcurand)