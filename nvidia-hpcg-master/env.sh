#!/bin/bash

spack load cuda
spack load nvhpc
spack load intel-oneapi-mpi

# --- 1. 根据文档要求，设置 *_PATH 变量 ---
# 使用 spack location -i <package> 动态获取路径，避免硬编码
export MPI_PATH=$(spack location -i intel-oneapi-mpi)
export CUDA_PATH=$(spack location -i cuda)
export I_MPI_PMI_LIBRARY=/slurm/libpmi2.so.0.0.0

# 对于 nvhpc 内部的库，需要拼接路径
NVHPC_ROOT=$(spack location -i nvhpc)
export MATHLIBS_PATH=$NVHPC_ROOT/Linux_x86_64/25.1/math_libs
export NCCL_PATH=$NVHPC_ROOT/Linux_x86_64/25.1/comm_libs/nccl
# 注意: 文档提到了 NVPL_SPARSE_PATH, 它也在 nvhpc 包里
export NVPL_SPARSE_PATH=$NVHPC_ROOT/Linux_x86_64/25.1/math_libs/lib/nvpl_sparse

# --- 2. 设置系统级的 PATH 和 LD_LIBRARY_PATH ---
# 这对于让命令行能找到 mpicxx 等命令至关重要
export PATH=$MPI_PATH/bin:$CUDA_PATH/bin:$PATH
export LD_LIBRARY_PATH=$MPI_PATH/lib:$CUDA_PATH/lib64:$MATHLIBS_PATH/lib64:$NCCL_PATH/lib:$LD_LIBRARY_PATH

echo "--- 环境设置完成 ---"
