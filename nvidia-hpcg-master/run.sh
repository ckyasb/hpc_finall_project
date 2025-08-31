#!/bin/bash

# ==============================================================================
# SLURM 作业提交脚本 for HPCG 性能测试 (v2 - 集群优化版)
#
# 使用方法:
# 1. 将此脚本保存在您的项目根目录 (nvidia-hpcg-master/)。
# 2. 提交作业: sbatch submit_hpcg.slurm
# 3. 查看作业状态: squeue -u $(whoami)
# 4. 查看输出日志: cat slurm-<job_id>.out
# ==============================================================================

# --- SLURM 作业参数设置 ---

#SBATCH --job-name=HPCG-Tune      # 作业名称，方便识别
#SBATCH --output=slurm-%j.out     # 标准输出和错误将写入此文件 (%j 会被替换为作业ID)

# --- 关键修改：根据您的集群推荐配置添加 --partition ---
#SBATCH --partition=V100          # 指定使用 V100 GPU 所在的分区

# --- 资源请求：已根据您的要求配置为2个GPU ---
#SBATCH --nodes=1                 # 我们只需要1个计算节点
#SBATCH --ntasks-per-node=2       # 在这个节点上启动2个MPI进程 (因为有2块GPU)
#SBATCH --cpus-per-task=1        # 为每个MPI进程分配64个CPU核心 (匹配您的硬件配置)
#SBATCH --gres=gpu:2              # 请求2块GPU资源
#SBATCH --time=01:00:00           # 预计作业运行时间 (1小时)，请根据run.sh中的测试数量调整

echo "=========================================================="
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Running on nodes: $SLURM_NODELIST"
echo "Number of Tasks: $SLURM_NTASKS"
echo "Tasks per Node: $SLURM_NTASKS_PER_NODE"
echo "CPUs per Task: $SLURM_CPUS_PER_TASK"
echo "Job started at: $(date)"
echo "=========================================================="
echo ""


# --- 执行环境和测试 ---

# 1. 加载您的计算环境
#    我们的 env.sh 脚本(位于根目录)已经包含了加载CUDA, MPI等所有依赖的逻辑，
#    这比单独的 module load cuda 更完整、更适合我们这个项目。
echo "正在加载计算环境..."
source ./env.sh
echo "环境加载完成。"
echo ""

# 2. 切换到可执行文件所在的 bin 目录
echo "正在进入 bin 目录..."
cd ./bin
echo "当前目录: $(pwd)"
echo ""

# 3. 执行我们的自动化测试脚本 run.sh (位于bin目录)
echo "开始执行 run.sh 自动化测试脚本..."
./run.sh

echo ""
echo "=========================================================="
echo "run.sh 脚本执行完毕。"
echo "Job finished at: $(date)"
echo "=========================================================="


