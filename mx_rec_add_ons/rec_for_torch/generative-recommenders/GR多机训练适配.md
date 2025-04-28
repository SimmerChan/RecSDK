# Generative Recommendations多机训练适配样例

## 适配说明
本样例基于Generative Recommendations已完成NPU侧训练迁移的模型代码,对其进行NPU多机训练的代码适配。（NPU迁移代码请参考本目录下README.md）

## 多机训练代码修改

拷贝原有目录下`train.py`， 更名为`train_multinode.py`

### 修改train_multinode,py

#### 1.重复数据集（可选）
原代码第76行，增加用于重复数据集的类`RepeatDataset`
```python
from torch.utils.data import Dataset

class RepeatDataset(Dataset):
    def __init__(self, dataset, num_repeats):
        self.dataset = dataset
        self.num_repeats = num_repeats
        
    def __len__(self):
        return len(self.dataset) * self.num_repeats
    
    def __getitem__(self, idx):
        return self.dataset[idx % len(self.dataset)]
```

原代码第158行，将 `train_data_sampler, train_data_loader=create_data_loader`这段代码修改为：

```python
train_dataset = RepeatDataset(dataset.train_dataset, num_repeats=300)
train_data_sampler, train_data_loader = create_data_loader(
    train_dataset,
    batch_size=local_batch_size,
    world_size=world_size,
    rank=rank,
    shuffle=True,
    drop_last=world_size>1,
)
```

原代码第166行，将`eval_data_sampler, eval_data_loader=create_data_loader`这段代码修改为：
```python
eval_dataset = RepeatDataset(dataset.eval_dataset, num_repeats=100)
eval_data_sampler, eval_data_loader = create_data_loader(
    eval_dataset,
    batch_size=eval_batch_size,
    world_size=world_size,
    rank=rank,
    shuffle=True,
    drop_last=world_size>1,
)
```
#### 2.修改setup函数

原代码第90行，修改`setup`函数：

```python
def setup(world_size:int) -> None:
    # initialize the process group
    addr = os.getenv("MASTER_ADDR")
    port = os.getenv("MASTER_PORT")
    rank_id = int(os.environ["RANK"])
    logging.info(f"tcp://{addr}:{port}, rank={rank_id}")
    dist.init_process_group("hccl", init_method=f"tcp://{addr}:{port}", rank=rank_id, 
                            world_size=world_size)
```

原代码第149行，调用`setup`函数的地方, 修改为：
```python
    setup(world_size)
```

#### 3.增加模型参数量打印、profiling采集(可选)

原代码第310行，在训练代码：`for epoch in range(num_epochs)`前增加模型参数量打印：

```python
    for name, param in model.named_parameters():
        logging.info(f"Layer:{name} | Size: {param.size()} | Number of parameters: {param.numel()}")
    
    # 总参数量计算：
    num_parameters= sum(p.numel() for p in model.parameters())
    logging.info(f"Total parameters: {num_parameters}")

```

在参数量打印后面加入prof设置：
```python
    experimental_config = torch_npu.profiler._ExperimentalConfig(
        export_type = torch_npu.profiler.ExportType.Text,
        profiler_level = torch_npu.profiler.ProfilerLevel.Level1,
        aic_metrics = torch_npu.profiler.AiCMetrics.PipeUtilization,
    )

    prof = torch_npu.profiler.profile(
        activities = [
            torch_npu.profiler.ProfilerActivity.CPU,
            torch_npu.profiler.ProfilerActivity.NPU,
        ],
        schedule = torch_npu.profiler.schedule(wait=0, warmup=10, active=1, repeat=1, skip_first=100),
        on_trace_ready = torch_npu.profiler.tensorboard_trace_handler("./profiler"),
        record_shapes = False,
        profile_memory = True,
        experimental_config = experimental_config)
    
    prof.start()
    
```
在原代码第431行，在`opt.step()`后添加`prof.step()`, 如下所示：
```python
        opt.step()
        prof.step()
        batch_id+=1
    prof.stop()
```

#### 4.修改main函数

原代码第556行，修改`main`函数：
```python
def main(argv):
    world_size = int(os.environ["RANK_SIZE"])
    mp_train_fn(int(os.environ["LOCAL_RANK"]), world_size, FLAGS.master_port, FLAGS.gin_config_file)
```

### 增加多级训练脚本run_multinode.sh

```shell
rank_size=16
export RANK_SIZE="$rank_size"
export HCCL_WHITELIST_DISABLE=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_SOCKET_IFNAME=enp189s0f0

export ASCEND_GLOBAL_LOG_LEVEL=3
export ASCEND_GLOBAL_EVENT_ENABLE=0
export ASCEND_SLOG_PRINT_TO_STDOUT=1
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
torchrun --nproc_per_node=$((rank_size/2)) --nnodes=2 --node_rank=0 --master_addr="61.47.2.188" --master_port=6006 train_multinode.py --gin_config_file=configs/ml-20m/hstu-sampled-softmax-n128-large-final.gin --master_port=1998 | tee temp.log
#torchrun --nproc_per_node=$((rank_size/2)) --nnodes=2 --node_rank=1 --master_addr="61.47.2.188" --master_port=6006 train_multinode.py --gin_config_file=configs/ml-20m/hstu-sampled-softmax-n128-large-final.gin --master_port=1998 | tee temp.log

```

## 多机训练代码运行

```shell
bash run_multimode.sh
```

## FAQ

### 1. 多机训练失败，plog报错为：Transport init error, call trace: hcclRet -> 13

使用的CANN版本、HDK版本不匹配，更换匹配的版本即可。注意：多机训练时，不同机器上需要使用相同版本的CANN、HDK