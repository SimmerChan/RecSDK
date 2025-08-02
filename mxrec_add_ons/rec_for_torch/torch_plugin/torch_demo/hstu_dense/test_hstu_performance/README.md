# 测试hstu npu, gpu性能对比
npu tag: https://gitee.com/ascend/RecSDK.git branch_v7.1.0-RC1
gpu tag: https://github.com/NVIDIA/recsys-examples.git v25.05
# 1. 准备一台GPU, 一台NPU 服务器
# 2. GPU服务器下载编译 recsys-example

```
git clone https://github.com/NVIDIA/recsys-examples.git
cd recsys-examples/corelib/hstu
git checkout v25.05
make install
```
配置服务器端的sftp server服务

# 3. 配置NFS共享路径

# 4. NPU算子编译

# 5. 配置config.py 填写所有x的位置
```
# Absolute directory path of the recsys-example project recsys-examples/corelib/hstu (replace with actual path)
RECSYS_DIR="x"

# Network File System (NFS) mount directory path (replace with actual path)
NFS_DIR="x"

# IP address of the GPU server (replace with actual IP)
GPU_IP="x"

# Login username for the GPU server (replace with actual username)  
GPU_USER="x"

# Absolute path to Python3 interpreter (replace with actual path, e.g. /usr/bin/python3)
PYTHON3="x"
```
# 6. 上传test_npu_performance 到NPU 共享目录NFS_DIR

# 7. 登录NPU服务器，执行性能测试命令
```
cd test_npu_performance
pip install -r requirements.txt
python3 test_benchmark.py
```
输入参数在benchmark.csv, 结果保存在result.csv中
