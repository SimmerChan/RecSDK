# HybridTorchrec模型迁移样例
## 迁移说明
本样例以DLRM模型为例,适配hybrid_torchrec框架并在NPU上进行训练。  
模型参考的开源链接为:https://github.com/facebookresearch/dlrm/tree/main/torchrec_dlrm/,  
克隆源码并固定版本为:Commits on Jun 7 , 2024，提交的SHA-1 hash值（提交ID）：b631a99 

## 构建镜像
### 前提
- 物理机上已经安装好对应CANN版本的驱动和固件
- 物理机上已经安装docker，并且docker网络可用
- 准备基础OS镜像：可从AscendHub上拉取[CentOS7.6.1810](https://www.hiascend.com/developer/ascendhub/detail/9353d9619c2a44db87845bce546c17bd)镜像或在物理机上使用命令`docker pull debian:12`从Dockerhub上拉取debian12镜像

### 准备依赖
新建`build_images`目录存放依赖。建议用户手动下载依赖，使用wget方式下载文件需确保下载文件名与Dockerfile中的文件名一致。
- gmp-6.1.0  
下载地址：https://mirrors.huaweicloud.com/gnu/gmp/gmp-6.1.0.tar.bz2
- mpfr-3.1.4  
下载地址：https://mirrors.huaweicloud.com/gnu/mpfr/mpfr-3.1.4.tar.bz2
- mpc-1.0.3  
下载地址：https://mirrors.huaweicloud.com/gnu/mpc/mpc-1.0.3.tar.gz
- isl-0.18  
下载地址：https://gcc.gnu.org/pub/gcc/infrastructure/isl-0.18.tar.bz2
- gcc-10.2.0  
下载地址：https://mirrors.huaweicloud.com/gnu/gcc/gcc-10.2.0/gcc-10.2.0.tar.gz
- cmake-3.20.6  
下载地址：https://cmake.org/files/v3.20/cmake-3.20.6.tar.gz
- python3.11.0  
下载地址：https://repo.huaweicloud.com:8443/artifactory/python-local/3.11.0/Python-3.11.0.tar.xz
- automake-1.16.5  
下载地址：https://ftp.gnu.org/gnu/automake/automake-1.16.5.tar.gz
- libtool-2.4.7  
下载地址：https://ftp.gnu.org/gnu/libtool/libtool-2.4.7.tar.gz
- libevent-2.1.12  
下载地址：https://github.com/libevent/libevent/releases/download/release-2.1.12-stable/libevent-2.1.12-stable.tar.gz
- libunwind-1.8.1  
下载地址：https://github.com/libunwind/libunwind/releases/download/v1.8.1/libunwind-1.8.1.tar.gz
- double-conversion-3.1.4  
下载地址：https://github.com/google/double-conversion/archive/refs/tags/v3.1.4.tar.gz
- gflags-2.2.2  
下载地址：https://github.com/gflags/gflags/archive/refs/tags/v2.2.2.tar.gz
- glog-0.4.0  
下载地址：https://github.com/google/glog/archive/refs/tags/v0.4.0.tar.gz
- fmt-9.1.0  
下载地址：https://github.com/fmtlib/fmt/archive/refs/tags/9.1.0.tar.gz
- gtest-1.10.0  
下载地址：https://github.com/google/googletest/archive/refs/tags/release-1.10.0.tar.gz
- boost_1_87_0  
下载地址：https://archives.boost.io/release/1.87.0/source/boost_1_87_0.tar.gz
- folly-2020.12.28.00  
下载地址：https://github.com/facebook/folly/archive/v2020.12.28.00.tar.gz

### 构建镜像
Step1：基础镜像为centos，执行操作一；基础镜像为debian，执行操作二。  
操作一： 在物理机上运行如下命令下载`CentOS-Base.repo`文件
```shell
wget -O ./CentOS-Base.repo https://mirrors.huaweicloud.com/repository/conf/CentOS-7-anon.repo --no-check-certificate
```
操作二： 在物理机上运行如下命令创建`sources.list`文件
```shell
cat << EOF > ./sources.list
deb http://mirrors.tuna.tsinghua.edu.cn/debian bookworm main contrib non-free
deb-src http://mirrors.tuna.tsinghua.edu.cn/debian bookworm main contrib non-free
deb http://mirrors.tuna.tsinghua.edu.cn/debian-security bookworm-security main contrib non-free
deb-src http://mirrors.tuna.tsinghua.edu.cn/debian-security bookworm-security main contrib non-free
deb http://mirrors.tuna.tsinghua.edu.cn/debian bookworm-updates main contrib non-free
deb-src http://mirrors.tuna.tsinghua.edu.cn/debian bookworm-updates main contrib non-free
EOF
```

Step2：以centos:7.6.1810或debian:12作为基础镜像构建镜像时，需要准备较多的依赖；用户请按照`准备依赖`章节内容下载相关文件。

Step3：在`build_images`目录下准备CANN包。用户可以从[昇腾社区](https://www.hiascend.com/developer/download/community/result?module=pt+cann&product=4&model=26)下载**8.0.RC3.beta1**版本的toolkit包与kernels包。用户根据实际情况选择是否使用配套的[固件驱动](https://www.hiascend.com/hardware/firmware-drivers/community?product=1&model=30&cann=8.0.RC3.beta1&driver=1.0.27.alpha)。**注意：软件包格式选择run**。

安装CANN包需要两个文件，分别是
- version.info（驱动版本文件）
- ascend_install.info（固件驱动安装文件）

可将物理机上相应的文件拷贝到`build_images`目录。物理机安装驱动与固件后，version.info默认安装路径为/usr/local/Ascend/driver/version.info；
ascend_install.info默认安装路径为/etc/ascend_install.info。

Step4：下载3个配套软件压缩包到`build_images`目录。

- Ascend-mindxsdk-mxrec-add-ons-*.tar.gz
- Ascend-mindxsdk-hybrid-torchrec-0.5.0-linux-*.tar.gz
- Ascend-mindxsdk-torchrec-0.5.0-npu-linux-*.tar.gz

运行如下命令下载`torch2.1.0`与`fbgemm_gpu0.5.0`的whl到`build_images`目录。
```shell
# torch2.1.0
wget https://download.pytorch.org/whl/cpu/torch-2.1.0%2Bcpu-cp311-cp311-linux_x86_64.whl
# fbgemm_gpu0.5.0
wget https://download.pytorch.org/whl/cpu/fbgemm_gpu-0.5.0%2Bcpu-cp311-cp311-manylinux2014_x86_64.whl
```

Step5：将`Dockerfile`,`CentOS-Base.repo`或`sources.list`文件移动到`build_images`目录中，并在目录内运行下面命令构建镜像。构建镜像的步骤在Dockerfile中有详细的说明，用户根据实际情况修改Dockerfile文件。制作镜像时需确保服务器能访问外网，否则需要配置代理。
```dockerfile
# 服务器能访问外网
docker build -t hybrid_torchrec_base:v1.0-x86 -f Dockerfile .
# 服务器配置代理访问外网
docker build -t hybrid_torchrec_base:v1.0-x86 -f Dockerfile --build-arg http_proxy=http://your_proxy --build-arg https_proxy=https://your_proxy .
```
**注意：**   
1. 以debian为基础镜像时，在安装folly库时会报错，用户可参考FAQ章节第一条内容修改folly-2020.12.28.00/folly/fibers/FiberManager.cpp，并将修改后的文件放在`build_images`目录下。
2. 运行命令前注意修改Dockerfile中的基础镜像名。
3. `build_images`目录下不能有相关依赖的解压文件，否则运行上述docker命令会报错。

## 启动容器
创建启动脚本run_docker.sh，参考如下：
```shell
#!/bin/bash
container_name=$1
image_name=$2
docker run \
-u root \
-it \
--name ${container_name} \
--net=host \
--shm-size="300g" \
--privileged \
-v /etc/localtime:/etc/localtime \
-e ASCEND_VISIBLE_DEVICES=0-7 \
-v /etc/ascend_install.info:/etc/ascend_install.info \
-v /home:/home \
-v /root/ascend:/root/ascend \
-v /root/.ssh:/root/.ssh \
-v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
${image_name} \
/bin/bash
```
执行如下命令新建容器：
```shell
bash run_docker.sh 容器名 {镜像名称}:{版本名称}
```

进入容器后，执行：
```shell
sudo su
```
切换为root用户

用户可参考以下命令对开源dlrm模型进行修改。注意压缩包`Ascend-mindxsdk-mxrec-add-ons-*.tar.gz`解压后的目录名称，根据实际情况修改下面命令。
```shell
tar -zxvf Ascend-mindxsdk-mxrec-add-ons-*.tar.gz
git clone -b main https://github.com/facebookresearch/dlrm.git
cd dlrm && git checkout b631a99 
cp -f ../mindxsdk-mxrec-add-ons/torchrec/torchrec_dlrm.patch ./
git apply torchrec_dlrm.patch
```

设置环境变量

```shell
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

### FAQ
1.基础镜像为debian时，在安装folly库时会遇到错误[error: no matching function for call to 'constexpr_max(long int, int)'
  348 | constexpr size_t kAltStackSize = folly::constexpr_max(SIGSTKSZ, 32 * 1024);](https://github.com/facebook/folly/issues/1539)。
可按照[PR](https://github.com/facebook/folly/commit/7df2d7e5098119c1562422ac9571e70f032adb50?diff=split#diff-7a7fc034ae3376ae1d877c9809405c48510e13de5d8c5100671c0594261e1a63L1-R409)修改folly/fibers/FiberManager.cpp文件。

2.使用debian镜像创建容器跑训练任务时，可能会遇到`The IPv6 network addresses of (xxx) cannot be retrieved`错误，其中括号内为主机名。
解决方案：编辑/etc/hosts文件，添加服务器IP与主机名（如 127.0.0.1 node1）。

3.CANN软件提供进程级环境变量设置脚本，供用户在进程中引用，以自动完成环境变量设置。用户进程结束后自动失效。设置CANN的相关环境变量，可在程序启动的Shell脚本执行，也可在命令行执行。CANN安装路径：root用户默认安装路径为“/usr/local/Ascend”。


## 数据集预处理
进入[开源模型官网](https://github.com/facebookresearch/dlrm/blob/main/torchrec_dlrm/README.MD)，官网提供两种方式跑通demo：
1. 下载原始数据处理后，提前进行mutil-hot的合成，产生4T的数据
2. 下载原始数据处理后，在训练的过程中生成mutil-hot数据，使用690gb数据集
由于1需要的条件苛刻，大部分机器很难满足条件，本次演示使用2中的条件。无host瓶颈的情况下，对性能影响较小。需要修改模型脚本代码，让host生成的数据在pin_memory上。

### 处理方法
Step 1: Download and uncompressing the [Criteo 1TB Click Logs dataset](https://ailab.criteo.com/download-criteo-1tb-click-logs-dataset/).  

Step 2: Run the 1TB Criteo Preprocess script.

```shell
# 需要1~2天才能处理完成。
bash ./scripts/process_Criteo_1TB_Click_Logs_dataset.sh \
./criteo_1tb/raw_input_dataset_dir \
./criteo_1tb/temp_intermediate_files_dir \
./criteo_1tb/numpy_contiguous_shuffled_output_dataset_dir
```

### torchrec_dlrm脚本侧的修改
`multi_hot.py`与`data/dlrm_dataloader.py`详细修改见`torchrec_dlrm.patch`文件。

## 运行命令
运行如下命令，会在`mindxsdk-mxrec-add-ons/torchrec/build`目录下生成`libparallel_hashmap.so`。在跑dlrm模型时设置该环境变量将使用folly库的hashmap，获得性能收益。使用Dockerfile制作镜像，会在`/usr/local/python3.11.0/lib/python3.11/site-packages/use_libso/`路径下存放编译好的`libparallel_hashmap.so`。
```shell
tar -zxvf Ascend-mindxsdk-mxrec-add-ons-*.tar.gz
mv mindxsdk-mxrec-add-ons* mindxsdk-mxrec-add-ons
cd mindxsdk-mxrec-add-ons/torchrec/parallel_hashmap
bash bash.sh
```

运行如下脚本启动训练任务。`$insert_your_path_here`为数据集路径。

```shell
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  # 指定哪些Device对当前进程可见
export PARALLEL_HASH_MAP_SO="/usr/local/python3.11.0/lib/python3.11/site-packages/use_libso/libparallel_hashmap.so"
export OMP_NUM_THREADS=12
export PREPROCESSED_DATASET=$insert_your_path_here
export TOTAL_TRAINING_SAMPLES=4195197692 ;
export GLOBAL_BATCH_SIZE=16384;
export WORLD_SIZE=8;
torchx run -s local_cwd dist.ddp -j 1x8 --script dlrm_main.py -- \
    --embedding_dim 128 \
    --dense_arch_layer_sizes 512,256,128 \
    --over_arch_layer_sizes 1024,1024,512,256,1 \
    --in_memory_binary_criteo_path $PREPROCESSED_DATASET \
    --num_embeddings_per_feature 40000000,39060,17295,7424,20265,3,7122,1543,63,40000000,3067956,405282,10,2209,11938,155,4,976,14,40000000,40000000,40000000,590152,12973,108,36 \
    --validation_freq_within_epoch $((TOTAL_TRAINING_SAMPLES / (GLOBAL_BATCH_SIZE * 20))) \
    --epochs 1 \
    --pin_memory \
    --mmap_mode \
    --batch_size $((GLOBAL_BATCH_SIZE / WORLD_SIZE)) \
    --interaction_type=dcn \
    --dcn_num_layers=3 \
    --dcn_low_rank_dim=512 \
    --adagrad \
    --learning_rate 0.005 \
    --multi_hot_distribution_type uniform \
    --multi_hot_sizes=3,2,1,2,6,1,1,1,1,7,3,8,1,6,9,5,1,1,1,12,100,27,10,3,1,1 2>&1 | tee "temp.log"
```
## 精度、性能对比

| Device Type | Number of GPUs/NPUs |Collective Size of Embedding Tables (GiB)|Local Batch Size|Global Batch Size|Learning Rate|Interaction Type|Optimizer| AUROC Over Test Set After 1 Epoch | Training speed                        | Time to Train 1 Epoch |Unique Flags|
|-------------|---------------------| --- | --- | --- | --- | --- | --- |-----------------------------------|---------------------------------------|-----------------------| --- |
| GPU         | 8                   |104.54|2,048|16,384|0.006|DCN v2|Adagrad| 0.7973                            | ~55.0 batches/s == ~901,120 samples/s | 1h20m21s              |`--batch_size 2048 --learning_rate 0.006 --adagrad --interaction_type=dcn` |
| NPU         | 8                   |104.54|2,048|16,384|0.006|DCN v2|Adagrad| 0.7975                            | ~59.0 batches/s == ~966,656 samples/s | 1h12m03s              |`--batch_size 2048 --learning_rate 0.006 --adagrad --interaction_type=dcn`|

说明：NPU测试结果为X86环境上的测试结果，模型测试时参数设置与GPU对齐,GPU具体测试数据参考: https://github.com/facebookresearch/dlrm/tree/main/torchrec_dlrm/ 。