**使用说明：**

1.导出网络结构到excel：protobuf.pb.xlsx 中去

python3.5 dump_tf.py --pb=protobuf.pb 

python3.5 dump_tf.py --type=0  --pb=protobuf.pb 

python3.5 dump_tf.py --type=1  --pbtxt=protobuf.pbtxt 


2.搜索protobuf.pbtxt网络中算子Add的前N层级联关系

python3.5 dump_tf.py --type=1  --pbtxt=protobuf.pbtxt --optype=Add  --optype_before_n_layers N

3.搜索protobuf网络中算子Add的后N层级联关系

python3.5 dump_tf.py --type=0  --pb=protobuf.pb --optype=Add  --optype_after_n_layers N

4.freeze ckpt中的frozen pb模型

python3.5 dump_tf.py --type=2  --ckpt=../character-level-convolutional-networks-for+Text-Classification-Using-CNN/ckpt_npu

-------------------------------------------------------------------------------------------------------------------------------

注：ckpt中目录要求如下：

[root@bms-aiserver-113 dump_op_shape]# ll ../character-level-convolutional-networks-for+Text-Classification-Using-CNN/ckpt_npu

total 90140

-rw------- 1 root root       77 Mar 29 21:08 checkpoint

-rw------- 1 root root   429491 Mar 29 21:08 graph.pbtxt

-rw------- 1 root root 91621428 Mar 29 21:08 model.ckpt.data-00000-of-00001

-rw------- 1 root root     1524 Mar 29 21:08 model.ckpt.index

-rw------- 1 root root   240627 Mar 29 21:08 model.ckpt.meta

此外dndump的可选环境变量1: INVALID_OUTNODE_LOC_RATIO （默认是0.3，表示找到的输出节点位置和所有节点个数的比例，小于该比例的输出节点是是伪输出节点，丢弃之） 

dndump的可选环境变量2: INVALID_PB_SIZE_RATIO  （默认是0.2，表示找到的输出节点冻结的图size和最大的冻结图size的比例，少于该比例的冻结图是伪图，丢弃之） 

dndump的可选环境变量3: FREEZEOUTNODES  （默认是空，表示指定freeze的输出节点，eg："Argmax,Softmax" ）

-------------------------------------------------------------------------------------------------------------------------------






**usage:** 

dump_tf.py [-h] [--type TYPE] [--pb PB] [--pbtxt PBTXT] [--ckpt CKPT]

                  [--optype OPTYPE] [--optype_location {A,S,M,E}]
                  
                  [--optype_before_n_layers OPTYPE_BEFORE_N_LAYERS]
                  
                  [--optype_after_n_layers OPTYPE_AFTER_N_LAYERS]
                  
                  [--debug DEBUG] [--find_output_names FIND_OUTPUT_NAMES]
                  
                  [--find_intput_names FIND_INTPUT_NAMES]

optional arguments:

  -h, --help            show this help message and exit
  
  --type TYPE           0:pb 1:pbtxt 2:freeze ckpt, default:0
  
  --pb PB               tf pb path
  
  --pbtxt PBTXT         tf pbtxt path
  
  --ckpt CKPT           ckpt path
  
  --optype OPTYPE       search bottom and top optype of optype from Network
  
  --optype_location {A,S,M,E}
  
                        Search for operators before or after the search
                        
                        operator with the search operator as ends or starts.
                        
  --optype_before_n_layers OPTYPE_BEFORE_N_LAYERS
  
                        search n layers of optype from Network
                        
  --optype_after_n_layers OPTYPE_AFTER_N_LAYERS
  
                        search n layers of optype from Network
                        
  --debug DEBUG         debug on or off
  
  --find_output_names FIND_OUTPUT_NAMES
  
                        find outputnodes_names
                        
  --find_intput_names FIND_INTPUT_NAMES
  
                        find intputnodes_names

