### 环境初始化

再项目根目录中，创建了env.sh来完成环境从初始化  

总而言之我们一共有  
两个数学库：armpl-gcc kml  
两个mpi：openmpi，mpich  

具体使用：  
组合A: 加载 OpenMPI 和 BoostKit (kml)  
source ./env.sh openmpi boostkit  
组合B: 加载 MPICH 和 ArmPL  
source ./env.sh mpich armpl  
组合C: 加载 OpenMPI 和 ArmPL  
source ./env.sh openmpi armpl  
……  

### 运行

在Aem/bin目录下运行./run.sh "问题规模" “总时长”  
  
示例  
./run.sh 64 60
单个进程上处理的问题结果为64*64*64，运行总时长上限为60s
运行结果会输出到当前目录

### 测试结果说明

文件名包含了具体的mpi进程数和具体的节点名  
文件中我们可以看到问题的规模配置  
kp03前四个版本：openmpi，armpl  
kp03后两个版本：openmpi，kml  
kp06两个版本：mpich，kml  
个让人感觉差别不大

### MPI测试结果

随着进程数的增加GFLOPS也随之增加，这是一个可以预见的结果，尽管破环了NUMA但是当前所有进程都会执行一个相同规模的问题，加速内存的访问。

### 编译选项

-Ofast: 极限速度优化。这是比 -O3 更高级别的优化，它会启用所有 -O3 的优化，并额外开启一些以微小精度为代价换取更高速度的浮点数运算优化。

-march=native: 为鲲鹏CPU量身定制。此标志告诉编译器，生成的代码只需在当前这台机器上运行，从而允许编译器使用鲲鹏处理器支持的所有最新、最快的指令集（如 SVE）。

-flto: 链接时优化 (LTO)。这是一个非常强大的“全局优化”技术。在程序的最后链接阶段，编译器会重新审视所有代码，进行跨文件的深度优化，能带来显著的性能提升。

-fopenmp: 启用OpenMP多线程。这是让程序能够在单个MPI进程内使用多个CPU核心（例如32个线程）的关键。

-ftree-loop-distribution: 高级循环优化。帮助编译器更好地重排循环代码，以提升缓存命中率和矢量化效率。

-fno-math-errno: 微小的数学函数优化。告诉编译器在执行标准数学函数时无需设置系统错误码，能节省一些额外的指令开销。
全部启动启动启动