add_executable(matmul_profile matmul_profile.cu)
target_link_libraries(matmul_profile PUBLIC gemm cuda_utils)
target_link_libraries(matmul_profile PUBLIC -lcublas -lcublasLt -lcudart -lcurand)