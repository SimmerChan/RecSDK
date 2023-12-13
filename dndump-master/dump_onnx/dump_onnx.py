import onnx
from onnx import helper, shape_inference
from onnx import TensorProto
import argparse
import multiprocessing
import pandas as pd
import os
from tools.dump_util import *

args = get_args()
print(args)
if args.onnx is None:
    print('Error: --onnx is None!')
    exit(0)

original_model = onnx.load(args.onnx)
modelname = args.onnx.split(
    '/')[-1][0:20] if args.onnx is not None else "default-network"
modelpath = os.path.dirname(args.onnx) if args.onnx is not None else "./"
modelpath = "./" if modelpath == '' else modelpath

#original_model = helper.make_model(graph, producer_name='onnx-examples')

# Check the model and print Y's shape information
onnx.checker.check_model(original_model)
#print('Before shape inference, the shape info of Y is:\n{}'.format(original_model.graph.value_info))

# Apply shape inference on the model
inferred_model = shape_inference.infer_shapes(original_model)

# Check the model and print Y's shape information
onnx.checker.check_model(inferred_model)
#print('After shape inference, the shape info of Y is:\n{}'.format(inferred_model.graph.value_info))

#onnxoptimizer
inferred_model=optimize(inferred_model) if args.onnxoptimizer else inferred_model

import pdb
pdb.set_trace() if args.debug else None
operator_shapes_dict = {}
for i in range(len(inferred_model.graph.input)):
    operator_shapes_dict[inferred_model.graph.input[i].name] = [
        -1 if dim.dim_value==0 else dim.dim_value 
        for dim in (inferred_model.graph.input[i].type.tensor_type.shape.dim)
    ]
for i in range(len(inferred_model.graph.initializer)):
    operator_shapes_dict[inferred_model.graph.initializer[i].
                         name] = inferred_model.graph.initializer[i].dims
for i in range(len(inferred_model.graph.sparse_initializer)):
    operator_shapes_dict[inferred_model.graph.sparse_initializer[
        i].name] = inferred_model.graph.sparse_initializer[i].dims
for i in range(len(inferred_model.graph.value_info)):
    if 'shape' in str(inferred_model.graph.value_info[i]):
        #unknow dims:
        #Ref: https://github.com/onnx/onnx/blob/master/docs/IR.md
        #The use of an empty string (as a dimension variable) to denote an unknown dimension not related to any other dimension. 
        #This was discarded in favor of using a Dimension with neither dim_value nor dim_param set.
        operator_shapes_dict[inferred_model.graph.value_info[i].name] = [
        -1 if dim.dim_value==0 else dim.dim_value for dim in (
            inferred_model.graph.value_info[i].type.tensor_type.shape.dim)
        ]
    else:
        #unknow rank:
        #Ref: https://github.com/onnx/onnx/blob/master/docs/IR.md
        #A tensor of unknown rank is represented using a TypeProto::Tensor object with no shape, which is legal.
        operator_shapes_dict[inferred_model.graph.value_info[i].name] = [-2]
for i in range(len(inferred_model.graph.output)):
    operator_shapes_dict[inferred_model.graph.output[i].name] = [
        -1 if dim.dim_value==0 else dim.dim_value 
        for dim in (inferred_model.graph.output[i].type.tensor_type.shape.dim)
    ]

operator_types_dict = {}
for i in range(len(inferred_model.graph.input)):
    operator_types_dict[inferred_model.graph.input[i].name] = (
        inferred_model.graph.input[i].type.tensor_type.elem_type)
for i in range(len(inferred_model.graph.initializer)):
    operator_types_dict[inferred_model.graph.initializer[i].
                        name] = inferred_model.graph.initializer[i].data_type
for i in range(len(inferred_model.graph.sparse_initializer)):
    operator_types_dict[inferred_model.graph.sparse_initializer[
        i].name] = inferred_model.graph.sparse_initializer[i].data_type
for i in range(len(inferred_model.graph.value_info)):
    operator_types_dict[
        inferred_model.graph.value_info[i].
        name] = inferred_model.graph.value_info[i].type.tensor_type.elem_type
for i in range(len(inferred_model.graph.output)):
    operator_types_dict[inferred_model.graph.output[i].name] = (
        inferred_model.graph.output[i].type.tensor_type.elem_type)

attribute_dict = {}
for i in range(len(inferred_model.graph.node)):
    node_name = inferred_model.graph.node[i].name if not inferred_model.graph.node[i].name == '' else \
        inferred_model.graph.node[i].op_type + '_' + inferred_model.graph.node[i].output[0]
    if not inferred_model.graph.node[i].op_type in ['Constant','ConstantOfShape']:
        attribute_dict[node_name] = inferred_model.graph.node[i].attribute
    else:
        attribute_dict[node_name] = []

all_ops = []
print(
    '|Nodename|Optype|InputLayer|OutputLayer|InputShape|OutputShape|InputDataType|OutputDataType|Attribute|'
) if args.debug else None
for i in range(len(inferred_model.graph.input)):
    all_ops.append({
        "name":
        inferred_model.graph.input[i].name,
        "optype":
        'Input',
        "inputLayer": [],
        "outputLayer": [inferred_model.graph.input[i].name],
        "input_shape": [],
        "output_shape":
        [operator_shapes_dict[inferred_model.graph.input[i].name]],
        "input_type": [],
        "output_type":
        [DataType[inferred_model.graph.input[i].type.tensor_type.elem_type]],
        "attribute":
        "[]"
    })
for node in inferred_model.graph.node:
    input_shapes_list = []
    output_shapes_list = []
    input_types_list = []
    output_types_list = []

    for i in node.input:
        if i in operator_shapes_dict.keys():
            input_shapes_list.append(operator_shapes_dict[i])
        else:
            input_shapes_list.append([])
    for i in node.output:
        if i in operator_shapes_dict.keys():
            output_shapes_list.append(operator_shapes_dict[i])
        else:
            output_shapes_list.append([])

    for i in node.input:
        if i in operator_types_dict.keys():
            input_types_list.append(DataType[operator_types_dict[i]])
        else:
            input_types_list.append(DataType[0])
    for i in node.output:
        if i in operator_types_dict.keys():
            output_types_list.append(DataType[operator_types_dict[i]])
        else:
            output_types_list.append(DataType[0])
    all_ops.append({
        "name":
        node.name if not node.name == '' else node.op_type + '_' +
        node.output[0],
        "optype":
        node.op_type,
        "inputLayer":
        node.input,
        "outputLayer":
        node.output,
        "input_shape":
        input_shapes_list,
        "output_shape":
        output_shapes_list,
        "input_type":
        input_types_list,
        "output_type":
        output_types_list,
        "attribute":
        str(attribute_dict[node.name]) if not node.name == '' else str(
                attribute_dict[node.op_type + '_' + node.output[0]])
    })
for i in range(len(inferred_model.graph.output)):
    all_ops.append({
        "name":
        inferred_model.graph.output[i].name,
        "optype":
        'Output',
        "inputLayer": [inferred_model.graph.output[i].name],
        "outputLayer": [inferred_model.graph.output[i].name],
        "input_shape":
        [operator_shapes_dict[inferred_model.graph.output[i].name]],
        "output_shape":
        [operator_shapes_dict[inferred_model.graph.output[i].name]],
        "input_type":
        [DataType[operator_types_dict[inferred_model.graph.output[i].name]]],
        "output_type":
        [DataType[operator_types_dict[inferred_model.graph.output[i].name]]],
        "attribute":
        "[]"
    })

for i in all_ops:
    print(i) if args.debug else None
    continue

writer = pd.ExcelWriter(modelpath + '/' + modelname + '.xlsx')
df = pd.DataFrame(
    data={
        'name': [i["name"] for i in all_ops],
        'optype': [i["optype"] for i in all_ops],
        'inputLayer': [i["inputLayer"] for i in all_ops],
        'outputLayer': [i["outputLayer"] for i in all_ops],
        'input_shape': [i["input_shape"] for i in all_ops],
        'output_shape': [i["output_shape"] for i in all_ops],
        'input_type': [i["input_type"] for i in all_ops],
        'output_type': [i["output_type"] for i in all_ops],
        'attribute': [i["attribute"] for i in all_ops]
    })
df.to_excel(writer, 'modelname')
writer.save()

print('[Result-excel]: pls check the result excel',
      modelpath + '/' + modelname + '.xlsx')
