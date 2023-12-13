使用说明：
1.导出网络结构到excel 中去
python3 dump_caffe.py --prototxt=models/ResNet-50-deploy.prototxt


2.搜索prototxt网络中算子ReLU的级联关系
python3 dump_caffe.py --prototxt=models/ResNet-50-deploy.prototxt --optype=ReLU

----------------multi_op(ReLU)-----------------------
['ReLU']--->['Convolution'] : 33
['Eltwise']--->['ReLU'] : 16
['Convolution']--->['ReLU'] : 33
['Convolution']--->['ReLU']--->['Convolution'] : 33
['ReLU']--->['Eltwise'] : 16
['Eltwise']--->['ReLU']--->['Eltwise'] : 16



