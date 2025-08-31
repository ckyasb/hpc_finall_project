#!/bin/bash

# 设置-e, 如果任何命令失败，脚本将立即退出
set -e

# 初始化变量以存储最佳结果
max_gflops=0
best_file=""

# --- 文件处理逻辑 ---
# 如果用户提供了文件名作为参数，则使用这些文件。
# 如果没有提供参数，则自动搜索当前目录下的所有文件。
files_to_check=()
if [ "$#" -gt 0 ]; then
    echo "将处理您通过参数指定的 ${#} 个文件..."
    files_to_check=("$@")
else
    echo "未提供文件名作为参数, 将自动搜索当前目录下的所有文件..."
    # 使用 find 命令可以更好地控制文件类型，并避免目录被包含进来
    # -maxdepth 1: 只搜索当前目录, 不进入子目录
    # -type f: 只选择文件
    # -print0 和 read -d '' 组合可以安全地处理包含空格或特殊字符的文件名
    while IFS= read -r -d '' file; do
        files_to_check+=("$file")
    done < <(find . -maxdepth 1 -type f -print0)
fi

# 检查是否有文件可供处理
if [ ${#files_to_check[@]} -eq 0 ]; then
    echo "错误: 未找到任何文件进行处理."
    echo "用法: $0 [HPCG输出文件1.log] [HPCG输出文件2.log] ..."
    echo "      (如果没有提供文件, 脚本会自动搜索当前目录)"
    exit 1
fi

echo "--------------------------------------------------------"
echo "正在 ${#files_to_check[@]} 个文件中搜索最佳的 HPCG 性能结果..."
echo "--------------------------------------------------------"

# 循环遍历所有待处理的文件
for file in "${files_to_check[@]}"; do
    # 检查文件是否存在且可读
    if [ -r "$file" ]; then
        # 使用 grep 查找结果行, 并用 awk 提取 GFLOP/s 值
        # 该行格式为: Final Summary::HPCG result is VALID with a GFLOP/s rating of=242.269
        # 使用 tail -n 1 确保即使一个文件中有多个结果行, 也只取最后一个
        current_gflops=$(grep "Final Summary::HPCG result is VALID" "$file" | tail -n 1 | awk -F'=' '{print $NF}')

        # 检查是否成功提取到了数值
        if [ -n "$current_gflops" ]; then
            echo "在文件 '$file' 中找到结果: ${current_gflops} GFLOP/s"
            
            # 使用 awk 进行浮点数比较, 判断当前结果是否更优
            is_better=$(awk -v current="$current_gflops" -v max="$max_gflops" 'BEGIN {print (current > max)}')

            if [ "$is_better" -eq 1 ]; then
                max_gflops=$current_gflops
                best_file=$file
            fi
        fi
    else
        echo "警告: 无法读取文件 '$file'. 已跳过."
    fi
done

echo "--------------------------------------------------------"

# 打印最终的最佳结果
if [ -n "$best_file" ]; then
    echo "🏆 找到的最佳结果:"
    echo "   所在文件: $best_file"
    echo "   GFLOP/s:  $max_gflops"
else
    echo "❌ 在所有被检查的文件中均未找到有效的 HPCG 结果行."
fi


