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

### 测试结果说明

文件名包含了具体的mpi进程数和具体的节点名
文件中我们可以看到问题的规模配置
kp03前四个版本：openmpi，armpl
kp03后两个版本：openmpi，kml
kp06两个版本：mpich，kml