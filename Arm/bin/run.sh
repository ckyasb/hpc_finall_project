#!/bin/bash

# --- 步骤 1: 检查用户输入参数 ---
# 检查传入的参数数量是否正确 (必须等于2个)
if [ "$#" -ne 2 ]; then
    echo "错误：参数数量不正确。"
    echo ""
    echo "用法: ./run_test.sh <问题规模> <运行时长(秒)>"
    echo "----------------------------------------------------"
    echo "示例 1 (快速调优): ./run_test.sh 256 120"
    echo "示例 2 (正式测试): ./run_test.sh 256 1800"
    exit 1
fi

# --- 步骤 2: 将参数赋值给可读的变量 ---
PROBLEM_SIZE=$1
RUNTIME_SECONDS=$2

# --- 步骤 3: 设置最优的环境变量 ---
# 每个MPI进程使用32个OpenMP线程 (4个进程 x 32线程 = 128核心)
export OMP_NUM_THREADS=32
# 开启OpenMP线程绑定，'close'表示线程会绑定到相邻的核心上，以提高缓存效率
export OMP_PROC_BIND=close
export OMP_PLACES=threads

# --- 步骤 4: 打印当前配置，供用户确认 ---
echo "================================================="
echo "准备运行HPCG测试..."
echo "MPI 进程数              : 4"
echo "每个进程的OpenMP线程数    : $OMP_NUM_THREADS"
echo "问题规模 (nx ny nz)       : $PROBLEM_SIZE x $PROBLEM_SIZE x $PROBLEM_SIZE"
echo "预计运行时长 (秒)         : $RUNTIME_SECONDS"
echo "================================================="

# --- 步骤 5: 执行核心运行命令 ---
# --map-by numa: 将4个进程分别精确地映射到4个NUMA节点上，这是性能的关键！
# --bind-to core: 将进程内的线程绑定到核心上
# --nx, --ny, --nz, --rt: 从命令行参数读取问题规模和时长
mpirun -np 4 --map-by numa --bind-to core ./xhpcg --nx=$PROBLEM_SIZE --ny=$PROBLEM_SIZE --nz=$PROBLEM_SIZE --rt=$RUNTIME_SECONDS

echo "================================================="
echo "测试运行结束。"
echo "请检查最新生成的 HPCG-Benchmark...txt 文件获取结果。"
echo "================================================="
