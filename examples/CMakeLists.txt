add_executable(test test.cu)
target_link_libraries(test PUBLIC memanager gemm cuda_utils)
target_link_libraries(test PUBLIC -lcublas -lcublasLt -lcudart -lcurand)