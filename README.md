### winograd

##### 初级版本

- 多内核分离 (Multi-Kernel Separation)
将复杂的Winograd算法分解为变换、矩阵乘法 (GEMM) 和反变换等多个独立的CUDA内核阶段，便于针对性优化。

- 调用cuBLAS库 (cuBLAS Integration)
对于计算最密集的矩阵乘法（GEMM）步骤，我们直接调用NVIDIA官方高度优化的数学库cuBLAS来执行，以获得最佳性能和稳定性。

- Grid-Stride循环 (Grid-Stride Loop)
在数据变换内核中，我们采用此模式来大幅减少内核启动的总次数。每个线程在内核内部以循环方式处理多个数据块，从根本上解决了海量内核启动带来的调度开销问题。

- shared memoryTiling / 协作抓取 (Advanced Tiling / Cooperative Fetching)
在最耗时的输入变换阶段，我们让一个线程块（Thread Block）作为一个团队，协作地将一块更大的、包含所有重叠区域的数据片（Patch）一次性加载到高速的**共享内存（Shared Memory）**中，以最大化数据复用，显著降低对慢速全局内存的访问压力。

- 启发式策略 (Heuristics)
程序会智能地根据输入尺寸（特别是输入通道数C）选择最优的计算路径：

对于大矩阵：使用上述的多内核+cuBLAS+Tiling的复杂流水线。

对于小矩阵：自动切换到一个更轻量级的融合内核（Fused Kernel），以避免Winograd算法在高固定开销下得不偿失。

##### 失败的优化

- tensor core：因为tensor core处理的数据是FP16，但是我们现在处理的数据是FP32，最终结果是精度不够，修改了main中精度校对为e-3就可以通过test，，初始的e-4则不行
- Double Buffering：不知道什么情况，性能下降了

##### 测试结果

使用环境：hpc101-cuda  

=== Running Naive Convolution (Baseline on a single GPU) ===
Layer  0: 1.485 ms (1801.65 GFLOPS)
Layer  1: 14.450 ms (1975.56 GFLOPS)
Layer  2: 24.942 ms (2289.15 GFLOPS)
Layer  3: 49.970 ms (2285.16 GFLOPS)
Layer  4: 101.873 ms (2241.81 GFLOPS)
Layer  5: 38.489 ms (2259.70 GFLOPS)
Layer  6: 75.237 ms (2311.97 GFLOPS)
Layer  7: 149.686 ms (2324.15 GFLOPS)
Layer  8: 302.475 ms (2300.30 GFLOPS)
Layer  9: 111.135 ms (2130.37 GFLOPS)
Layer 10: 1.028 ms (2066.20 GFLOPS)
Layer 11: 19.589 ms (2313.39 GFLOPS)
Layer 12: 9.503 ms (2288.11 GFLOPS)
Layer 13: 18.636 ms (2333.41 GFLOPS)
Layer 14: 5.575 ms (2193.96 GFLOPS)
Layer 15: 2.071 ms (2051.04 GFLOPS)
Layer 16: 5.469 ms (2070.62 GFLOPS)
Layer 17: 5.722 ms (1899.93 GFLOPS)
Baseline Total: 937.336 ms (2265.56 GFLOPS)

=== Running Winograd Convolution (2 GPUs with NCCL) ===
Layer  0: 0.830 ms (3225.72 GFLOPS)
Layer  1: 3.924 ms (7276.01 GFLOPS)
Layer  2: 4.780 ms (11945.38 GFLOPS)
Layer  3: 7.006 ms (16298.60 GFLOPS)
Layer  4: 9.417 ms (24252.09 GFLOPS)
Layer  5: 3.083 ms (28209.60 GFLOPS)
Layer  6: 4.661 ms (37318.20 GFLOPS)
Layer  7: 8.575 ms (40568.41 GFLOPS)
Layer  8: 13.058 ms (53284.89 GFLOPS)
Layer  9: 5.077 ms (46632.18 GFLOPS)
Layer 10: 0.637 ms (3336.26 GFLOPS)
Layer 11: 7.568 ms (5988.39 GFLOPS)
Layer 12: 1.906 ms (11405.05 GFLOPS)
Layer 13: 2.869 ms (15159.68 GFLOPS)
Layer 14: 0.807 ms (15152.20 GFLOPS)
Layer 15: 0.552 ms (7699.13 GFLOPS)
Layer 16: 0.819 ms (13832.89 GFLOPS)
Layer 17: 0.602 ms (18070.56 GFLOPS)
Winograd Total (2 GPUs): 76.169 ms (27879.99 GFLOPS)

=== Correctness Check ===
Layer  0: CORRECT (Speedup: 1.79x)  
Layer  1: CORRECT (Speedup: 3.68x)  
Layer  2: CORRECT (Speedup: 5.22x)  
Layer  3: CORRECT (Speedup: 7.13x)  
Layer  4: CORRECT (Speedup: 10.82x)  
Layer  5: CORRECT (Speedup: 12.48x)  
Layer  6: CORRECT (Speedup: 16.14x)  
Layer  7: CORRECT (Speedup: 17.46x)  
Layer  8: CORRECT (Speedup: 23.16x)  
Layer  9: CORRECT (Speedup: 21.89x)  
Layer 10: CORRECT (Speedup: 1.61x)  
Layer 11: CORRECT (Speedup: 2.59x)  
Layer 12: CORRECT (Speedup: 4.98x)  
Layer 13: CORRECT (Speedup: 6.50x)  
Layer 14: CORRECT (Speedup: 6.91x)  
Layer 15: CORRECT (Speedup: 3.75x)  
Layer 16: CORRECT (Speedup: 6.68x)  
Layer 17: CORRECT (Speedup: 9.51x)  

=== Final Results ===
All layers passed correctness check!
Baseline Total (1 GPU): 937.33 ms (2265.56 GFLOPS)  
Winograd Total (2 GPUs): 76.16 ms (27879.99 GFLOPS) 
Overall Speedup: 12.31x