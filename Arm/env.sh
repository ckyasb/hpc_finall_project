#!/bin/bash

# =================================================================
# HPCG 多版本编译环境设置脚本 (env.sh)
#
# 功能:
#   根据用户选择，加载不同的MPI库和数学库到当前终端环境。
#   支持 Spack 安装的软件包和手动安装的软件包。
#
# 用法:
#   source env.sh <MPI库选择> <数学库选择>
#
#   必须使用 'source' 或 '.' 命令来执行，否则环境变量不会生效！
# =================================================================

# --- 步骤 1: 检查用户输入参数 ---
# 检查传入的参数数量是否正确 (必须等于2个)
if [ "$#" -ne 2 ]; then
    echo "错误：参数数量不正确。"
    echo ""
    echo "用法: source env.sh <MPI库选择> <数学库选择>"
    echo "----------------------------------------------------"
    echo "MPI库选择 (第一个参数):"
    echo "  - openmpi"
    echo "  - mpich"
    echo ""
    echo "数学库选择 (第二个参数):"
    echo "  - armpl    (Spack 安装的 Arm Performance Libraries)"
    echo "  - boostkit (手动安装在 ~/kml 的 BoostKit Math Library)"
    echo ""
    echo "示例: source env.sh openmpi boostkit"
    return 1 # 使用 return 而非 exit，因为脚本是被 source 的
fi

# --- 步骤 2: 将参数赋值给可读的变量 ---
MPI_CHOICE=$1
MATH_LIB_CHOICE=$2

echo "--- 开始配置环境 ---"

# --- 步骤 3: 根据选择加载MPI库 ---
case $MPI_CHOICE in
  openmpi)
    echo "正在加载 Spack 安装的 OpenMPI..."
    spack load openmpi
    ;;
  mpich)
    echo "正在加载 Spack 安装的 MPICH..."
    spack load mpich
    ;;
  *)
    echo "错误: 无效的MPI库选择 '$MPI_CHOICE'。请选择 'openmpi' 或 'mpich'。"
    return 1
    ;;
esac

# --- 步骤 4: 根据选择加载数学库 ---
case $MATH_LIB_CHOICE in
  armpl)
    echo "正在加载 Spack 安装的 armpl-gcc..."
    spack load armpl-gcc
    ;;
  boostkit)
    # 对于手动安装的库，我们需要手动设置环境变量
    KML_DIR=~/kml
    echo "正在从手动安装路径 '$KML_DIR' 加载 BoostKit Math Library (KML)..."

    # 检查目录是否存在，增加脚本的健壮性
    if [ -d "$KML_DIR/lib" ] || [ -d "$KML_DIR/lib64" ]; then
        # 将库文件路径添加到链接器和运行时库路径中
        # Prepend (放在前面) 以确保它被优先找到
        export LD_LIBRARY_PATH=$KML_DIR/lib:$KML_DIR/lib64:$LD_LIBRARY_PATH
        export LIBRARY_PATH=$KML_DIR/lib:$KML_DIR/lib64:$LIBRARY_PATH
        # 将头文件路径添加到编译器搜索路径中
        export CPATH=$KML_DIR/include:$CPATH
        echo "BoostKit 路径已成功添加到环境变量。"
    else
        echo "警告: 在 '$KML_DIR' 中找不到 'lib' 或 'lib64' 目录。请确认您的安装路径。"
    fi
    ;;
  *)
    echo "错误: 无效的数学库选择 '$MATH_LIB_CHOICE'。请选择 'armpl' 或 'boostkit'。"
    return 1
    ;;
esac

echo ""
echo "✅ 环境设置完成！"
echo "您现在可以开始编译了。"

