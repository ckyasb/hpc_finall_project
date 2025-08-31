#!/bin/bash

# ==============================================================================
# HPCG 进阶性能测试自动化脚本 (v13 - 采用SLURM原生集成)
#
# 该脚本会自动检测 SLURM 环境。如果检测到，它将使用 srun 的原生MPI集成
# (--mpi=pmi2) 和任务CPU分配 (--cpus-per-task) 功能，
# 这是在SLURM环境中运行MPI程序的最佳实践。
# ==============================================================================

# --- 用户配置区 ---
# 请根据您找到的最佳问题规模修改以下参数
NX=256
NY=256
NZ=544
RT=1800 # 每次测试的运行时间（秒）

# 其他 hpcg.sh 脚本参数
P2P_MODE=4
BENCH_MODE=1

# --- 待扫描的参数列表 ---
# 要测试的MPI进程数 (例如，1个GPU vs 2个GPU)
declare -a N_PROCS_LIST=("2")

# 根据 lscpu 输出 (2个Socket, 每个Socket 64核), 我们将测试更宽的核心数范围
declare -a OMP_THREADS_LIST=("2")

# 为2个GPU定义要测试的进程网格布局 ("NPX NPY NPZ")
declare -a GRIDS_FOR_2_PROCS=("1 1 2")
# 为1个GPU定义进程网格布局 (只有一个进程，所以只有一种布局)
declare -a GRIDS_FOR_1_PROC=("1 1 1")

# L2缓存压缩选项 (0=禁用, 1=启用)
declare -a L2CMP_OPTIONS=("1")

# GPU切片大小 (GSS) 选项
declare -a GSS_OPTIONS=("4096")

# 定义存放结果的目录 (相对于 bin/ 目录)
RESULT_DIR="./result"
# --- 配置区结束 ---


# --- 脚本主逻辑 ---
mkdir -p ${RESULT_DIR}

# --- 新增：智能启动器与MPI检测 (v13) ---
echo "正在检测运行环境..."
# 1. 检测调度器
if [ -n "$SLURM_JOB_ID" ]; then
    echo "检测到 SLURM 环境。将使用 'srun' 启动器并采用原生MPI集成。"
    LAUNCHER="srun"
    
    # --- 关键优化 (v13)：采用SLURM原生MPI支持 ---
    # --mpi=pmi2: 这是官方推荐的、最可靠的方式，让SLURM和MPI进行交互。
    MPI_FLAG="--mpi=pmi2"
    
    # 保留这些环境变量作为兼容性保障
    export I_MPI_HYDRA_BOOTSTRAP=slurm
    export I_MPI_PMI_PROVIDER=pmi2
    if [ -z "$I_MPI_PMI_LIBRARY" ]; then 
        PMI_LIB_PATH=$(find /usr -name "libpmi2.so" 2>/dev/null | head -n 1)
        if [ -n "$PMI_LIB_PATH" ]; then
            export I_MPI_PMI_LIBRARY=$PMI_LIB_PATH
        fi
    fi
    
    # 构建 --export 参数，确保将关键变量传递给计算节点
    # 注意：我们不再需要传递 OMP_NUM_THREADS，因为它将由 --cpus-per-task 控制
    EXPORT_VARS_BASE="ALL,I_MPI_HYDRA_BOOTSTRAP=slurm,I_MPI_PMI_PROVIDER=pmi2"
    if [ -n "$I_MPI_PMI_LIBRARY" ]; then
        EXPORT_VARS_BASE="${EXPORT_VARS_BASE},I_MPI_PMI_LIBRARY=${I_MPI_PMI_LIBRARY}"
    fi
    EXPORT_COMMAND="--export=${EXPORT_VARS_BASE}"
    # --- 优化结束 ---

else
    echo "未检测到 SLURM 环境。将使用 'mpirun' 启动器。"
    LAUNCHER="mpirun"
    MPI_ENV_PREFIX="" 

    # 2. 检测MPI实现 (仅在非SLURM环境下)
    MPI_VERSION_INFO=$($LAUNCHER --version)
    if [[ $MPI_VERSION_INFO == *"Intel(R) MPI Library"* ]]; then
        echo "检测到 Intel MPI。"
        MPI_ENV_FLAG="-env"
        MPI_BIND_FLAG="-env I_MPI_PIN_DOMAIN socket"
        MPI_ENV_COMMAND="${MPI_ENV_FLAG} OMP_NUM_THREADS"
    elif [[ $MPI_VERSION_INFO == *"Open MPI"* ]]; then
        echo "检测到 Open MPI。"
        MPI_ENV_FLAG="--genv"
        MPI_BIND_FLAG="-bind-to socket"
        MPI_ENV_COMMAND="${MPI_ENV_FLAG} OMP_NUM_THREADS="
    else
        echo "警告：未能识别MPI版本，将默认使用Open MPI的语法。"
        MPI_ENV_FLAG="--genv"
        MPI_BIND_FLAG="-bind-to socket"
        MPI_ENV_COMMAND="${MPI_ENV_FLAG} OMP_NUM_THREADS="
    fi
fi
echo "----------------------------------------------------------"
# --- 检测结束 ---


echo "开始HPCG进阶性能扫描 (v13)..."
echo "结果将保存在: ${RESULT_DIR}"
echo "=========================================================="

for n_procs in "${N_PROCS_LIST[@]}"; do
    if [ "$n_procs" -eq 1 ]; then GRIDS=("${GRIDS_FOR_1_PROC[@]}"); GPU_AFFINITY="0"; else GRIDS=("${GRIDS_FOR_2_PROCS[@]}"); GPU_AFFINITY="0:1"; fi
    for n_threads in "${OMP_THREADS_LIST[@]}"; do
        for grid in "${GRIDS[@]}"; do
            read -r NPX NPY NPZ <<< "$grid"; GRID_NAME="${NPX}x${NPY}x${NPZ}"
            for l2cmp in "${L2CMP_OPTIONS[@]}"; do
                for gss in "${GSS_OPTIONS[@]}"; do
                    echo "正在测试: NProc=${n_procs}, OMP_Threads=${n_threads}, Grid=${GRID_NAME}, L2Cmp=${l2cmp}, GSS=${gss}"
                    TIMESTAMP=$(date +%Y%m%d_%HM%S)
                    LOG_FILE="${RESULT_DIR}/hpcg_n${n_procs}_t${n_threads}_g${GRID_NAME}_l2c${l2cmp}_gss${gss}_${TIMESTAMP}.log"

                    # --- 优化点 (v13): 根据启动器构建不同的命令 ---
                    if [ "$LAUNCHER" == "srun" ]; then
                        # 使用 --cpus-per-task 来控制核心数，这是SLURM的原生方式
                        CPU_TASK_FLAG="--cpus-per-task=${n_threads}"
                        COMMAND="${LAUNCHER} -n ${n_procs} ${MPI_FLAG} ${CPU_TASK_FLAG} ${EXPORT_COMMAND} ./hpcg.sh \
                        --nx ${NX} --ny ${NY} --nz ${NZ} --rt ${RT} \
                        --p2p ${P2P_MODE} --gpu-affinity ${GPU_AFFINITY} --b ${BENCH_MODE} \
                        --npx ${NPX} --npy ${NPY} --npz ${NPZ} \
                        --l2cmp ${l2cmp} \
                        --gss ${gss}"
			bash -lc 'echo RANK=$SLURM_PROCID ; nvidia-smi --query-compute-apps=pid,process_name,gpu_uuid,used_memory --format=csv'
                    else
                        # 保持旧的 mpirun 逻辑
                        OMP_ENV_COMMAND=$(echo "${MPI_ENV_COMMAND}${n_threads}")
                        COMMAND="${LAUNCHER} -n ${n_procs} ${MPI_BIND_FLAG} ${OMP_ENV_COMMAND} ./hpcg.sh \
                        --nx ${NX} --ny ${NY} --nz ${NZ} --rt ${RT} \
                        --p2p ${P2P_MODE} --gpu-affinity ${GPU_AFFINITY} --b ${BENCH_MODE} \
                        --npx ${NPX} --npy ${NPY} --npz ${NPZ} \
                        --l2cmp ${l2cmp} \
                        --gss ${gss}"
			bash -lc 'echo RANK=$SLURM_PROCID ; nvidia-smi --query-compute-apps=pid,process_name,gpu_uuid,used_memory --format=csv'
                    fi
                    
                    # --- 执行与记录 ---
                    echo "################################################################" > "${LOG_FILE}"
                    echo "### Test executed at: $(date)" >> "${LOG_FILE}"; echo "### Command:" >> "${LOG_FILE}"
                    echo "${COMMAND}" >> "${LOG_FILE}"; echo "################################################################" >> "${LOG_FILE}"; echo "" >> "${LOG_FILE}"
                    eval ${COMMAND} >> "${LOG_FILE}" 2>&1
                    echo "测试完成。日志已保存至: ${LOG_FILE}"; echo "----------------------------------------------------------"
                done
            done
        done
    done
done

echo "=========================================================="
echo "所有进阶测试已完成！"


