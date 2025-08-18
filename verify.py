import numpy as np

# --- Winograd 变换矩阵 ---
G = np.array([
    [1.0, 0.0, 0.0],
    [0.5, 0.5, 0.5],
    [0.5, -0.5, 0.5],
    [0.0, 0.0, 1.0]
], dtype=np.float32)

B_T = np.array([
    [1.0, 0.0, -1.0, 0.0],
    [0.0, 1.0, 1.0, 0.0],
    [0.0, -1.0, 1.0, 0.0],
    [0.0, 1.0, 0.0, -1.0]
], dtype=np.float32)

A_T = np.array([
    [1.0, 1.0, 1.0, 0.0],
    [0.0, 1.0, -1.0, -1.0]
], dtype=np.float32)

def print_matrix(matrix, title):
    """Helper function to print a matrix."""
    print(f"--- {title} ---")
    with np.printoptions(precision=4, suppress=True, linewidth=120):
        print(matrix)
    print("")

def main():
    """
    用 NumPy 计算 Winograd 变换的中间结果，用于与 CUDA 代码的输出进行比对。
    请将 CUDA 程序打印的 "For Python Verification" 部分的矩阵粘贴到下方。
    """
    
    # --- 在下方粘贴你的输入数据 ---
    # 这是从 CUDA 程序输出中复制的第一个 4x4 输入图块
    d = np.array([
        [ 2.3789,  2.8335,  1.1280,  9.8239],
        [ 9.5881,  5.7032,  3.7523,  4.1118],
        [ 5.5492,  1.7149,  0.3340,  9.5042],
        [ 1.6413,  4.9083,  9.3621,  3.0414]
    ], dtype=np.float32)

    # 这是从 CUDA 程序输出中复制的第一个 3x3 滤波器
    g = np.array([
        [ 8.0163,  9.8833,  3.3335],
        [ 9.1433,  4.9322,  3.7242],
        [ 2.7293,  2.0830,  3.2925]
    ], dtype=np.float32)
    # --- 数据粘贴结束 ---

    # 1. 计算滤波器变换: U = G @ g @ G.T
    U = G @ g @ G.T
    
    # 2. 计算输入变换: V = B_T @ d @ B_T.T
    V = B_T @ d @ B_T.T
    
    # 3. 计算逐元素乘积 (对应 16 次 GEMM 的结果)
    # M_elementwise 包含了 16 个值，对应于 CUDA 中 M_padded 矩阵在
    # (k=0, p=0) 位置的 16 个 gemm_idx 的结果。
    M_elementwise = U * V
    
    # 4. 计算输出变换: Y = A_T @ M @ A_T.T
    Y = A_T @ M_elementwise @ A_T.T

    # 打印结果
    print("--- Python (NumPy) Verification Results ---")
    # U[0] 对应 U 矩阵的左上角 (k=0, c=0)
    # 在 CUDA 中，这 16 个值被分散到 U_padded[0...15] 的 (k=0,c=0) 位置
    print_matrix(U, "U = G @ g @ G.T")
    
    # V[0] 对应 V 矩阵的左上角 (p=0, c=0)
    print_matrix(V, "V = B_T @ d @ B_T.T")

    # M[0] 对应 M 矩阵的左上角 (k=0, p=0)
    print_matrix(M_elementwise, "M = U * V (element-wise)")

    print_matrix(Y, "Y = A_T @ M @ A")


if __name__ == "__main__":
    main()

