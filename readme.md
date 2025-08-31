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