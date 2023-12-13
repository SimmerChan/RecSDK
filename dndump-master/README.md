**dump_tf：**

使用说明：

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

注：ckpt中目录要求如下：

[root@bms-aiserver-113 dump_op_shape]# ll ../character-level-convolutional-networks-for+Text-Classification-Using-CNN/ckpt_npu

total 90140

-rw------- 1 root root       77 Mar 29 21:08 checkpoint

-rw------- 1 root root   429491 Mar 29 21:08 graph.pbtxt

-rw------- 1 root root 91621428 Mar 29 21:08 model.ckpt.data-00000-of-00001

-rw------- 1 root root     1524 Mar 29 21:08 model.ckpt.index

-rw------- 1 root root   240627 Mar 29 21:08 model.ckpt.meta





usage: dump_tf.py [-h] [--type TYPE] [--pb PB] [--pbtxt PBTXT] [--ckpt CKPT]

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






**dump_caffe：**

使用说明：

使用方法1.导出网络结构到excel 中去

python3 dump_caffe.py --prototxt=models/ResNet-50-deploy.prototxt


使用方法2.搜索prototxt网络中算子ReLU的级联关系

python3 dump_caffe.py --prototxt=models/ResNet-50-deploy.prototxt --optype=ReLU

----------------multi_op(ReLU)-----------------------

['ReLU']--->['Convolution'] : 33

['Eltwise']--->['ReLU'] : 16

['Convolution']--->['ReLU'] : 33

['Convolution']--->['ReLU']--->['Convolution'] : 33

['ReLU']--->['Eltwise'] : 16

['Eltwise']--->['ReLU']--->['Eltwise'] : 16


【结果查看】：dump_caffe.py同级目录下有一个excel文件 eg:"ResNet-50-deploy.prototxt.xlsx"


