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

=== Running Naive Convolution ===
Layer  0: 1.49 ms (1800.83 GFLOPS)
Layer  1: 14.45 ms (1975.23 GFLOPS)
Layer  2: 25.15 ms (2270.47 GFLOPS)
Layer  3: 48.68 ms (2345.58 GFLOPS)
Layer  4: 98.53 ms (2317.90 GFLOPS)
Layer  5: 37.02 ms (2349.14 GFLOPS)
Layer  6: 74.14 ms (2346.34 GFLOPS)
Layer  7: 148.14 ms (2348.47 GFLOPS)
Layer  8: 296.61 ms (2345.75 GFLOPS)
Layer  9: 110.89 ms (2135.11 GFLOPS)
Layer 10: 1.01 ms (2094.01 GFLOPS)
Layer 11: 19.37 ms (2339.48 GFLOPS)
Layer 12: 9.33 ms (2330.72 GFLOPS)
Layer 13: 18.58 ms (2340.96 GFLOPS)
Layer 14: 5.57 ms (2196.38 GFLOPS)
Layer 15: 2.07 ms (2056.46 GFLOPS)
Layer 16: 5.46 ms (2073.86 GFLOPS)
Layer 17: 5.72 ms (1901.52 GFLOPS)
Baseline Total: 922.20 ms (2302.75 GFLOPS)

=== Running Winograd Convolution ===
Layer  0: 1.36 ms (1970.55 GFLOPS)
Layer  1: 10.23 ms (2789.61 GFLOPS)
Layer  2: 14.64 ms (3900.80 GFLOPS)
Layer  3: 20.50 ms (5570.11 GFLOPS)
Layer  4: 31.29 ms (7298.17 GFLOPS)
Layer  5: 9.48 ms (9174.85 GFLOPS)
Layer  6: 13.58 ms (12813.23 GFLOPS)
Layer  7: 23.99 ms (14502.63 GFLOPS)
Layer  8: 38.38 ms (18127.61 GFLOPS)
Layer  9: 13.28 ms (17828.43 GFLOPS)
Layer 10: 1.10 ms (1938.15 GFLOPS)
Layer 11: 15.40 ms (2943.61 GFLOPS)
Layer 12: 4.82 ms (4509.49 GFLOPS)
Layer 13: 8.91 ms (4882.80 GFLOPS)
Layer 14: 3.42 ms (3572.82 GFLOPS)
Layer 15: 2.09 ms (2028.96 GFLOPS)
Layer 16: 2.93 ms (3868.66 GFLOPS)
Layer 17: 2.32 ms (4691.49 GFLOPS)
Custom Total: 217.71 ms (9754.34 GFLOPS)

=== Correctness Check ===
Layer  0: CORRECT (Speedup: 1.09x)
Layer  1: CORRECT (Speedup: 1.41x)
Layer  2: CORRECT (Speedup: 1.72x)
Layer  3: CORRECT (Speedup: 2.37x)
Layer  4: CORRECT (Speedup: 3.15x)
Layer  5: CORRECT (Speedup: 3.91x)
Layer  6: CORRECT (Speedup: 5.46x)
Layer  7: CORRECT (Speedup: 6.18x)
Layer  8: CORRECT (Speedup: 7.73x)
Layer  9: CORRECT (Speedup: 8.35x)
Layer 10: CORRECT (Speedup: 0.93x)
Layer 11: CORRECT (Speedup: 1.26x)
Layer 12: CORRECT (Speedup: 1.93x)
Layer 13: CORRECT (Speedup: 2.09x)
Layer 14: CORRECT (Speedup: 1.63x)
Layer 15: CORRECT (Speedup: 0.99x)
Layer 16: CORRECT (Speedup: 1.87x)
Layer 17: CORRECT (Speedup: 2.47x)

=== Final Results ===
Results are correct!
Naive:    922.20 ms (2302.75 GFLOPS)
Winograd: 217.71 ms (9754.34 GFLOPS)
Overall speedup: 4.24x