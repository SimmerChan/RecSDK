## **模型转换工具使用说明**


-----------------
### **工具简介**： 

将基于mxRec+NPU训练保存下来的NPU格式的稀疏表，转换为可被GPU、CPU加载使用的格式。

### **环境依赖**

该工具在tf1/tf2环境上进行测试，两种环境配置如下，供用户参考：

> **tf1**
<br>
tensorflow == 1.15.0 / 1.15.4<br>
numpy == 1.21.6 <br>
python == 3.7.5 <br>


> **tf2**
<br>
tensorflow == 2.6.5 <br>
numpy == 1.19.5 <br>
python == 3.7.5 <br>

<br>

### **使用说明**：

<br>

**使用范例：**

示例如下：<br>
`python3 model_convert.py --input_path=./saved_model --output_path=./saved-model-out`

打屏日志输出 `convert model success.` 代表模型成功转换完成。

**参数解释：**

`input_path`:   类型：`str`。 NPU格式的模型保存路径。

`output_path`:  类型：`str`。 转换后CPU/GPU 格式输出的路径。该参数可以用户自行设置，若该模型输出路径不存在，会新建目录。

<br>

**加载说明：**

由于tf1\tf2接口更迭，故本工具在编码时tf1\tf2采取的接口不同。<br>在加载的时候根据tf版本情况采用不同的加载接口。示例如下:

- tf1
<br>
```python
restore_table = tf.contrib.lookup.MutableHashTable(
                    key_dtype=tf.int64, 
                    value_dtype=tf.float32,  
                    default_value=initialize_value,  
                    name=args.table_name,  
                    checkpoint=True)  

with tf.Session() as sess:  
    saver = tf.train.Saver()  
    saver.restore(sess, args.path + "/model.ckpt-0"）  
    lookup_embedding = restore_table.lookup(key)  
```

- tf2
```python
restore_table = tf.lookup.experimental.MutableHashTable(
            key_dtype=tf.int64, 
            value_dtype=tf.float32,  
            default_value=np.zeros((240,)),  
            name="deep_sparse_table")  
 
restore_table1 = tf.lookup.experimental.MutableHashTable(
            key_dtype=tf.int64, 
            value_dtype=tf.float32,  
            default_value=np.zeros((37,)),  
            name="wide_sparse_table")  

# 这里restore table的顺序需与保存的顺序保持一致，保存的顺序在模型转换的时候会输出  

checkpoint = tf.train.Checkpoint(table_list=[restore_table,restore_table1])  
manager = tf.train.CheckpointManager(checkpoint, directory=args.path, max_to_keep=3)  
checkpoint.restore(manager.latest_checkpoint)  
 
lookup_embedding = restore_table1.lookup(key)  
```