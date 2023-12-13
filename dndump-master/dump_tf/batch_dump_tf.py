from tensorflow.python.platform import gfile
from tensorflow.python.tools import freeze_graph
import pdb
import tensorflow as tf
import os
import sys
import xlsxwriter
import datetime
from google.protobuf import text_format
import argparse
import networkx as nx
import json
import time
from tools.dump_util import *
from tools.backend_tf import *
import multiprocessing

args = get_args()
print(args)

pdb.set_trace() if args.debug else None

model_pb = None
model_meta_pb = None
model_pb_list = []
if (args.type == 3):
    for root, dirs, files in os.walk(args.batch_ckpt, topdown=False):
        for name in files:
            if "graph.pbtxt" == name:
                model_pb_list.append(root + "/graph.pbtxt")
else:
    print("args.type is invalid! not support ", args.type)
    exit(0)

##########################################Main-start#######################################

for model_pb in model_pb_list:
    optype = args.optype
    netname = model_pb.split(
        '/')[-1][0:20] if model_pb is not None else "default-netname"
    timestamp = str(time.time())
    workbook = xlsxwriter.Workbook(netname + "_" + timestamp + '.xlsx')
    worksheet = workbook.add_worksheet(netname)
    worksheet_op = workbook.add_worksheet("op")
    worksheet_multi = workbook.add_worksheet("multi_op")
    netname = netname + "_" + timestamp

    init_worksheet(worksheet, worksheet_op, worksheet_multi)
    G_LIST = []
    if model_pb is not None:
        with gfile.FastGFile(model_pb, 'rb') as f:
            pdb.set_trace() if args.debug else None
            graph_def = tf.GraphDef()
            if (args.type == 0 or args.type == 4):
                graph_def.ParseFromString(f.read())
            else:
                text_format.Merge(f.read(), graph_def)
            #tf.train.import_meta_graph('my_graph.meta')
            #sess.graph.as_default()
            try:
                tf.import_graph_def(graph_def, name='')
                print("---------tf.import_graph_def [", model_pb,
                      "] Success!----------")
            except Exception as e:
                print("tf.import_graph_def [", model_pb, "] failed!")
                print(e)
                continue
            all_ops = tf.get_default_graph().get_operations()
            #tf.train.write_graph(graph_def, './', 'protobuf.pbtxt', as_text=True)
    else:
        #graph_def = tf.GraphDef()
        pdb.set_trace() if args.debug else None
        try:
            tf.train.import_meta_graph(model_meta_pb)
            graph_def = tf.get_default_graph().as_graph_def()
            tf.train.write_graph(graph_def,
                                 args.ckpt,
                                 'graph.pbtxt',
                                 as_text=True)
            print("---------tf.train.import_meta_graph [", model_meta_pb,
                  "] Success!----------")
        except Exception as e:
            print("tf.train.import_meta_graph [", model_meta_pb, "] failed!")
            print(e)
            continue
        model_pb = args.ckpt.rstrip('/') + "/graph.pbtxt"
        all_ops = tf.get_default_graph().get_operations()

    ops = []
    print("all_ops:", all_ops) if args.debug == 2 else None
    for index, op in enumerate(all_ops):
        op_info = {
            "name": "",
            "optype": "",
            "control_inputLayer": [],
            "inputLayer": [],
            "outputLayer": [],
            "input_shape": [],
            "output_shape": []
        }
        op_info["name"] = op.node_def.name
        op_info["optype"] = op.type
        for input_tensor in op.inputs:
            op_info["input_shape"].append(input_tensor.get_shape())
            if args.debug:
                op_info["inputLayer"].append(input_tensor.name)
            else:
                op_info["inputLayer"].append(input_tensor.name[:-2])
        for control_input_tensor in op.control_inputs:
            op_info["control_inputLayer"].append(control_input_tensor.name)
        for out_tensor in op.outputs:
            op_info["output_shape"].append(out_tensor.get_shape())
            if args.debug:
                op_info["outputLayer"].append(out_tensor.name)
            else:
                op_info["outputLayer"].append(out_tensor.name[:-2])
        print(
            str(op_info["name"]) + "---------" +
            str(op_info["output_shape"])) if args.debug == 2 else None
        ops.append(op_info)

        if args.type == 4 and op_info["optype"] == "Placeholder":
            inputs = inputs + op.node_def.name + ':0;' if args.inputs == 'NUll' else args.inputs
            input_types = input_types + str(
                op.node_def.attr['dtype'].type
            ) + ';' if args.input_types == 'NUll' else args.input_types
            if args.input_types == 'NUll':
                if not len(op.node_def.attr['shape'].shape.dim) == 0:
                    for d in op.node_def.attr['shape'].shape.dim:
                        input_shapes = input_shapes + str(d.size) + ','
                    input_shapes = input_shapes[0:-1] + ';'
                else:
                    input_shapes = input_shapes + '-1;'
            else:
                input_shapes = args.input_shapes

        if args.type == 4 and index == len(all_ops) - 1:
            if args.outputs == 'NUll':
                outputs = outputs + op.node_def.name + ':0;'
            else:
                outputs = args.outputs

    with open(netname + ".txt", "w") as f_txt:
        for l in ops:
            LOG = l["name"] + "   " + str(tensorshape_list(
                l["input_shape"])) + "   " + str(
                    tensorshape_list(l["output_shape"])) + "\n"
            f_txt.write(LOG) if args.debug else None

    i = 0
    all_op = {}
    #for i in range(len(graph_def.node)):
    for i in range(len(ops)):
        #Layer
        col = 'A'
        row = col + str(i + 2)
        data = [ops[i]["name"]]
        worksheet.write_row(row, data)

        #Type
        col = chr(ord(col) + 1)
        row = col + str(i + 2)
        data = [ops[i]["optype"]]
        worksheet.write_row(row, data)

        #inputLayer
        col = chr(ord(col) + 1)

        row = col + str(i + 2)

        data = [str(ops[i]['inputLayer'])]
        worksheet.write_row(row, data)

        #outputLayer
        col = chr(ord(col) + 1)
        row = col + str(i + 2)
        data = [str(ops[i]['outputLayer'])]
        worksheet.write_row(row, data)

        #input_shape
        col = chr(ord(col) + 1)
        row = col + str(i + 2)
        data = [str(tensorshape_list(ops[i]['input_shape']))]
        worksheet.write_row(row, data)

        #output_shape
        col = chr(ord(col) + 1)
        row = col + str(i + 2)
        data = [str(tensorshape_list(ops[i]['output_shape']))]
        worksheet.write_row(row, data)
        #attr
        for j in graph_def.node[i].attr.keys():
            col = chr(ord(col) + 1)
            row = col + str(i + 2)
            data = 'key : ' + j + ' \nvalue : { ' + str(
                graph_def.node[i].attr[j]) + '}'
            if graph_def.node[i].attr[j].tensor.tensor_content != b'':
                length = len(graph_def.node[i].attr[j].tensor.tensor_content)
                data = del_field(data, 'tensor_content', length)
            data = [data]
            worksheet.write_row(row, data)

        #pdb.set_trace()
        #find all_op
        if (all_op.get(ops[i]["optype"]) is not None):
            all_op[ops[i]["optype"]] += 1
        else:
            all_op.update({ops[i]["optype"]: 1})
        #find networkx
        if (optype != "NULL" and optype != "ALL"
                and optype == ops[i]["optype"]):
            G1 = nx.DiGraph()
            G2 = nx.DiGraph()
            find_output_op_new(i, ops, ops[i]["name"], optype, G1,
                               args.optype_after_n_layers, 0)
            find_input_op_new(i, ops, ops[i]["name"], optype, G2,
                              args.optype_before_n_layers)
            G3 = nx.compose(G1, G2)
            G1.graph = str(i) + "_" + optype + "_S" + "_0_" + str(
                args.optype_after_n_layers)
            G2.graph = str(i) + "_" + optype + "_E" + "_" + str(
                args.optype_before_n_layers) + "_0"
            G3.graph = str(i) + "_" + optype + "_M" + "_" + str(
                args.optype_before_n_layers) + "_" + str(
                    args.optype_after_n_layers)
            if args.optype_location == "A":
                G_LIST.append(G1)
                G_LIST.append(G2)
                G_LIST.append(G3)
            elif args.optype_location == "M":
                G_LIST.append(G3)
            elif args.optype_location == "S":
                G_LIST.append(G1)
            elif args.optype_location == "E":
                G_LIST.append(G2)

    print('----------------------------' + netname +
          '----------------------------------')

    if (optype == "ALL"):
        for key in all_op.keys():
            for i in range(len(ops)):
                if key == ops[i]['optype']:
                    G1 = nx.DiGraph()
                    G2 = nx.DiGraph()
                    find_output_op_new(i, ops, ops[i]["name"],
                                       ops[i]['optype'], G1,
                                       args.optype_after_n_layers, 0)
                    find_input_op_new(i, ops, ops[i]["name"], ops[i]['optype'],
                                      G2, args.optype_before_n_layers)
                    G3 = nx.compose(G1, G2)
                    G1.graph = str(i) + "_" + key + "_S" + "_0_" + str(
                        args.optype_after_n_layers)
                    G2.graph = str(i) + "_" + key + "_E" + "_" + str(
                        args.optype_before_n_layers) + "_0"
                    G3.graph = str(i) + "_" + key + "_M" + "_" + str(
                        args.optype_before_n_layers) + "_" + str(
                            args.optype_after_n_layers)
                    G_LIST.append(G1)
                    G_LIST.append(G2)
                    G_LIST.append(G3)

    ##########################Result###################################
    #print result: all_op  networkx
    print('\n' + netname + ' : ' + str(sum(list(all_op.values()))))
    print('----------------all_op------------------------')
    #for i in all_op.keys():
    for index, i in enumerate(list(all_op.keys())):
        print(i + ' : ' + str(all_op[i]))
        worksheet_op.write_row('A' + str(index + 2), [i])
        worksheet_op.write_row('B' + str(index + 2), [str(all_op[i])])
    print('\n')
    statistic_op(all_op)

    print('----------------networkx------------------------')
    if not os.path.exists("./output_json"):
        os.mkdir("./output_json")
    if not os.path.exists("./output_json/" + netname):
        os.mkdir("./output_json/" + netname)
    for i, G in enumerate(G_LIST):
        #print(G.number_of_nodes()) if args.debug else 0
        #print(G.number_of_edges()) if args.debug else 0
        print(list(G.nodes)) if args.debug else 0
        print(list(G.edges)) if args.debug else 0
        print(G.nodes.data()) if args.debug else 0
        #worksheet_multi.write_row('A'+str(i+2), [str(list(G.edges))])
        js_file = "./output_json/" + netname + "/" + G.graph + ".json"
        with open(js_file, "w" if args.debug else "a") as f:
            json.dump(nx.node_link_data(G), f)
            print("Dump subgraph [" + str(i) + "] to " + js_file +
                  " success!") if args.debug else 0
    _ = statistic_graph(G_LIST)
    for index, j in enumerate(list(_.keys())):
        if not j == "NA":
            print(j, _[j])
            worksheet_multi.write_row('A' + str(index + 1), [j])
            worksheet_multi.write_row('B' + str(index + 1), [str(_[j])])
    print("\nFind operator [", optype, "]'s location [", args.optype_location,
          "]: ", sum(list(_.values())))
    workbook.close()
    print(
        '\n-------------------------Result----------------------------------')
    print("\nExcel result report is in [" + netname + ".xlsx], pls check it!")
    print('\n-------------------------' + netname +
          '--------------------------------------')
    tf.reset_default_graph()

    if args.type == 2:
        find_outputnodes_name(args, ops, model_meta_pb)

    if args.type == 4:
        pdb.set_trace() if args.debug else None
        print("------------------Start inference--------------------------")
        BTF = BackendTensorflow(args,
                                model_pb,
                                inputs=inputs.rstrip(';'),
                                input_shapes=input_shapes.rstrip(';'),
                                input_types=input_types.rstrip(';'),
                                outputs=outputs.rstrip(';'))
        try:
            BTF.load()
            for i in range(args.iter):
                output = BTF.predict()
                print("------------------iter" + str(i) +
                      "--------------------------")
                for index, res in enumerate(BTF.outputs.split(";")):
                    print(
                        "Inference result-[" + res + "] :", output
                        if not isinstance(output, list) else output[index])
                    if isinstance(output, list):
                        print(
                            "Inference result-[" + res + "] shape:",
                            "0" if not isinstance(output[index], np.ndarray)
                            else output[index].shape)
                    else:
                        print(
                            "Inference result-[" + res + "] shape:",
                            "0" if not isinstance(output, np.ndarray) else
                            output.shape)

            tf.reset_default_graph()
        except Exception as e:
            print("[ERROR]:Inference: inference pb failed! pb:", model_pb)
            print(e)
        print("------------------End inference--------------------------")

print(
    '\n-------------------------Summary-Result----------------------------------'
)
result_summary(args, optype)
