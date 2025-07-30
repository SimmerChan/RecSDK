# 1. 准备一台GPU, 一台NPU 服务器
# 2. GPU服务器下载编译 recsys-example

```
git clone https://github.com/NVIDIA/recsys-examples.git
cd recsys-examples/corelib/hstu
git checkout v25.05
make install
```

# 3. 配置共享路径

# 4. 配置config.py 填写所有x的位置
```
RECSYS_DIR="x"
NFS_DIR="x"

GPU_IP="x"
GPU_USER="x"   
GPU_PASSWORD="x"

PYTHON3="x"
```
# 5. 上传test_npu_performance 到NPU 共享目录NFS_DIR

# 6. 登录NPU服务器，执行性能测试命令
```
cd test_npu_performance; python3 test_benchmark.py
```
输入参数在benchmark.csv, 结果保存在result.csv中
