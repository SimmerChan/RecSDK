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
import re
import multiprocessing
# from queue import Queue
import threading
import numpy as np
from tools.backend_tf import *

try:
    from npu_bridge.hccl import hccl_ops
    from npu_bridge.estimator.npu.npu_optimizer import NPUDistributedOptimizer
    from npu_bridge.estimator.npu.npu_loss_scale_optimizer import NPULossScaleOptimizer
    from npu_bridge.estimator.npu.npu_loss_scale_manager import ExponentialUpdateLossScaleManager
    from npu_bridge.estimator.npu.npu_loss_scale_manager import FixedLossScaleManager
except:
    print("[WARNING]:No npu_bridge!!!")

try:
    import horovod.tensorflow as hvd
except:
    print("[WARNING]:No horovod!!!")

BLACK_TYPE_LIST = [
    'Const',
    'SparseApplyMomentum',
    'ApplyMomentum',
    'Merge',
    'IteratorToStringHandle',
    'Adam',
    'IsVariableInitialized',
    'VarIsInitializedOp',
    'AssignSub',
    'Assign',
    'Shape',
    'Less',
    'Greater',
    'NextIteration',
    'GeneratorDataset',
    'ZerosLike',
    'Switch',
    'AssignAdd',
    'ReadVariableOp',
    'Exit',
    # 'Identity',
    'NotEqual',
    'Equal',
    'LogicalAnd',
    "LogicalOr",
    "LogicalNot",
    "DrawBoundingBoxes",
    "DrawBoundingBoxesV2",
    'ApplyAdam'
]
BLACK_NAME_PREFIX_LIST = [
    'global_step/',
    'save/',
    'save_',
    'init_ops/',
    'gradients/',
    'Adam/',
    # 'model/optimizer/gradients/',
    'ApplyAdam/'
]
BLACK_NAME_INTER_LIST = [
    '/gradients/', '/BasicLSTMCellZeroState/', '/Regularizer/'
]
DEL_TYPE_LIST = [
    "Assert", "Print", "PrintV2", "TensorSummaryV2", "TensorSummary",
    "ScalarSummary", "HistogramSummary", "ImageSummary", "AudioSummaryV2",
    "AudioSummary", "MergeSummary", "Timestamp"
]

WHITE_TYPE_LIST = [
    'Argmax',
    'Softmax',
]
WHITE_NAME_LIST = []
FORCE_OUTPUT_NAME_LIST = []
if os.getenv('FREEZEOUTNODES') is not None:
    if not os.getenv('FREEZEOUTNODES') == '':
        FORCE_OUTPUT_NAME_LIST = os.getenv('FREEZEOUTNODES').split(",")

INVALID_PB_SIZE_RATIO = 0.2
if os.getenv('INVALID_PB_SIZE_RATIO') is not None:
    try:
        INVALID_PB_SIZE_RATIO = float(os.getenv('INVALID_PB_SIZE_RATIO'))
    except:
        INVALID_PB_SIZE_RATIO = 0.2

INVALID_OUTNODE_LOC_RATIO = 0.1
if os.getenv('INVALID_OUTNODE_LOC_RATIO') is not None:
    try:
        INVALID_OUTNODE_LOC_RATIO = float(
            os.getenv('INVALID_OUTNODE_LOC_RATIO'))
    except:
        INVALID_OUTNODE_LOC_RATIO = 0.1

G_dic = {"NA": 0}
op_dic = {}

inputs = ''
input_shapes = ''
input_types = ''
outputs = ''


#####################################config##########################################
def get_args():
    """Parse commandline."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--type",
        type=int,
        default=0,
        choices=[0, 1, 2, 3, 4],
        help=
        "0:dump pb 1:dump pbtxt 2:dump & freeze ckpt 3:batch dump & freeze ckpt 4:dump & inference pb, choices=[0,1,2,3,4],default:0"
    )
    parser.add_argument("--pb", help="tf pb path. --type=0, --type=4")
    parser.add_argument("--pbtxt", help="tf pbtxt path . --type=1")
    parser.add_argument("--ckpt", help="ckpt path. --type=2")
    parser.add_argument("--batch_ckpt", help="batch_ckpt path. --type=3")
    parser.add_argument(
        "--optype",
        default="NULL",
        help="(Optional)search bottom and top optype of optype from Network,")
    parser.add_argument("--optype_location", default="A", choices=["A", "S", "M", "E"], \
                        help="(Optional) A:All,S:Start,M:Middle,E:End; Search for operators before or after the search operator with the search operator as ends or starts.")
    parser.add_argument(
        "--optype_before_n_layers",
        default=1,
        type=int,
        help="(Optional) search n layers of optype from Network")
    parser.add_argument(
        "--optype_after_n_layers",
        default=1,
        type=int,
        help="(Optional) search n layers of optype from Network")
    parser.add_argument("--debug",
                        default=0,
                        type=int,
                        help="(Optional) debug on or off")
    parser.add_argument("--inference",
                        action="store_true",
                        help="(Optional) enable inference on")
    parser.add_argument(
        "--inputs",
        default='NUll',
        help=
        "(Optional) inputs node list, semicolon separated. eg:placeholder:0;placeholder_1:0"
    )
    parser.add_argument(
        "--input_shapes",
        default='NUll',
        help=
        "(Optional) inputs node list, semicolon separated. eg:16,128;16,128")
    parser.add_argument(
        "--input_types",
        default='NUll',
        help=
        "(Optional) inputs node list, semicolon separated. eg:float32;float32")
    parser.add_argument(
        "--outputs",
        default='NUll',
        help=
        "(Optional) outputs node list, semicolon separated. eg:Argmax:0;Softmax:0"
    )
    parser.add_argument("--iter",
                        default=10,
                        type=int,
                        help="(Optional) outputs node list")
    parser.add_argument("--processes",
                        default=int(multiprocessing.cpu_count() / 3),
                        type=int,
                        help="(Optional) multi processes freeze pb")
    #parser.add_argument("--find_output_names", default=0, type=int, help="(Optional) find outputnodes_names")
    #parser.add_argument("--find_intput_names", default=0, type=int, help="(Optional) find intputnodes_names")
    args = parser.parse_args()
    return args


def init_worksheet(worksheet, worksheet_op, worksheet_multi):
    COL = {
        'Layer': 'A',
        'Type': 'B',
        'inputLayer': 'C',
        'outputLayer': 'D',
        'input_shape': 'E',
        'output_shape': 'F',
        'attr1': 'G',
        'attr2': 'H',
        'attr3': 'I',
        'attr4': 'J',
        'attr5': 'K'
    }
    for key in list(COL.keys()):
        row = COL[key] + '1'
        data = [key]
        worksheet.write_row(row, data)

    COL_op = {'op': 'A', 'count': 'B'}
    for key in list(COL_op.keys()):
        row = COL_op[key] + '1'
        data = [key]
        worksheet_op.write_row(row, data)

    COL_multi = {'multi_op': 'A', 'count': 'B'}
    for key in list(COL_multi.keys()):
        row = COL_multi[key] + '1'
        data = [key]
        worksheet_multi.write_row(row, data)


def replace_GPU(file, old_str, new_str):
    with open(file, "r", encoding="utf-8") as f1, open("%s.bak" % file,
                                                       "w",
                                                       encoding="utf-8") as f2:
        for line in f1:
            f2.write(re.sub(old_str, new_str, line))
    # os.remove(file)
    os.rename(file, "%s.old" % file)
    os.rename("%s.bak" % file, file)


# Add 20201228
def find_optype_by_name_new(ops,
                            name,
                            optype,
                            find_str=None,
                            find_type=0,
                            G=None):
    i_list = []
    name_list = []
    optype_list = []
    for i in range(len(ops)):
        if find_type == 1 and find_str in ops[i]["inputLayer"]:
            optype_list.append(ops[i]['optype'])
            name_list.append(ops[i]["name"])
            i_list.append(i)
            G.add_node(ops[i]["name"], optype=ops[i]['optype'])
            G.add_edge(name, ops[i]["name"])
        if find_type == 0 and find_str in ops[i]["outputLayer"]:
            optype_list.append(ops[i]['optype'])
            name_list.append(ops[i]["name"])
            i_list.append(i)
            G.add_node(ops[i]["name"], optype=ops[i]['optype'])
            G.add_edge(ops[i]["name"], name)
    return i_list, name_list, optype_list


def find_output_op_new(i, ops, name, optype, G=None, n_layers=0, m_layers=0):
    if G is not None:
        n_layers = n_layers - 1
        m_layers = m_layers + 1
        if not G.has_node(name):
            G.add_node(name, optype=optype)
        # pdb.set_trace()
        if n_layers != -1:
            for _ in ops[i]["outputLayer"]:
                i_list, name_list, optype_list = find_optype_by_name_new(
                    ops, name, optype, _, 1, G)
                for index in range(len(i_list)):
                    find_output_op_new(i_list[index], ops, name_list[index],
                                       optype_list[index], G, n_layers,
                                       m_layers)
                    find_input_op_new(i_list[index], ops, name_list[index],
                                      optype_list[index], G, m_layers)


def find_input_op_new(i, ops, name, optype, G=None, n_layers=0):
    if G is not None:
        n_layers = n_layers - 1
        if not G.has_node(name):
            G.add_node(name, optype=optype)
        if n_layers != -1:
            for _ in ops[i]["inputLayer"]:
                i_list, name_list, optype_list = find_optype_by_name_new(
                    ops, name, optype, _, 0, G)
                for index in range(len(i_list)):
                    find_input_op_new(i_list[index], ops, name_list[index],
                                      optype_list[index], G, n_layers)


# Add 20201228


def replace_none(l):
    tmp = []
    # pdb.set_trace()
    for i in l:
        if i is None:
            tmp.append(-1)
        else:
            tmp.append(i)
    return tmp


def del_field(data, field, length):
    start_pos = data.find(field)
    end_pos = data.find('\t\n', start_pos)
    data = data[0:start_pos] + data[end_pos:]
    return data


def tensorshape_list(tensorshape):
    list = []
    try:
        # scalar
        if tensorshape == []:
            return list
        # Unknown shape & Fully-known shape & Partially-known shape
        for i in tensorshape:
            if i == tf.TensorShape(None):
                list.append([-2])
            else:
                list.append(replace_none(i.as_list()))
        return list
    except ValueError:
        print('tensorshape:', tensorshape)
        raise ValueError("tensorshape:")


def del_invalid_pb(path, suffix='frozen_model_'):
    pb_list = [
        os.path.join(root, file) for root, dirs, files in os.walk(path)
        for file in files if file.startswith(suffix)
    ]
    pb_size = [os.path.getsize(pb_file) for pb_file in pb_list]
    for index, size in enumerate(pb_size):
        if size < max(pb_size) * INVALID_PB_SIZE_RATIO:
            os.remove(pb_list[index])
            print("[ERROR]", pb_list[index], "file", size,
                  " is too small,del it")


def statistic_graph(G_LIST):
    _ = {"NA": 0}
    for i, G in enumerate(G_LIST):
        G_list = []
        tmp_list = list(G.edges)
        for j, edge in enumerate(tmp_list):
            G_tmp_list = []
            G_tmp_list.append(G.nodes().get(edge[0])['optype'])
            G_tmp_list.append(G.nodes().get(edge[1])['optype'])
            G_list.append(G_tmp_list)
        if str(G_list) in G_dic.keys():
            G_dic[str(G_list)] += 1
        else:
            G_dic[str(G_list)] = 1

        if str(G_list) in _.keys():
            _[str(G_list)] += 1
        else:
            _[str(G_list)] = 1
    return _


def statistic_op(all_op):
    for i in all_op.keys():
        if i in op_dic.keys():
            op_dic[i] = op_dic[i] + all_op[i]
        else:
            op_dic[i] = all_op[i]


def freeze_pb_process(args, ops, model_meta_pb, qitem):
    if (args.type == 2):
        try:
            pdb.set_trace() if args.debug else None
            tf.reset_default_graph()
            print('[OK]:dump_tf.py: start freeze pb ---', qitem.nodename)
            '''
            ret=freeze_graph.freeze_graph(args.ckpt.rstrip('/')+'/graph.pbtxt',
                    '', 
                    False,  
                    model_meta_pb[0:-5], 
                    qitem.nodename,'save/restore_all', 'save/Const:0', 
                    args.ckpt.rstrip('/')+'/frozen_model_'+qitem.nodename.replace('/','-')+'.pb',False,'')
            '''
            ret = freeze_graph.freeze_graph(
                '',
                '',
                True,
                model_meta_pb[0:-5],
                qitem.nodename,
                'save/restore_all',
                'save/Const:0',
                args.ckpt.rstrip('/') + '/frozen_model_' +
                qitem.nodename.replace('/', '-') + '.pb',
                False,
                '',
                variable_names_whitelist='',
                variable_names_blacklist='',
                input_meta_graph=model_meta_pb,
                input_saved_model_dir=None)

            print('[OK]:dump_tf.py: graph_def nodes number is:',
                  len(tf.get_default_graph().as_graph_def().node))
            if ret is not None:
                print(
                    "[OK]:dump_tf.py:frozen pb: ",
                    args.ckpt.rstrip('/') + '/frozen_model_' +
                    qitem.nodename.replace('/', '-') + '.pb', " success!")
            else:
                print(
                    "[WARNING]:dump_tf.py: invalid graph_def, so no need frozen it: ",
                    args.ckpt.rstrip('/') + '/frozen_model_' +
                    qitem.nodename.replace('/', '-') + '.pb')
        except Exception as e:
            print("[ERROR]:dump_tf.py: freeze pb failed! nodename:",
                  qitem.nodename)
            print(e)


class Item:
    """An item that we queue for processing by the thread pool."""
    def __init__(self, index, nodename=None):
        self.index = index
        self.nodename = nodename
        self.start = time.time()


class MultiProcessRunner():
    def __init__(self, args, ops, model_meta_pb):
        self.tasks = multiprocessing.Queue(maxsize=1024)
        self.workers = []
        self.result_dict = {}
        for _ in range(args.processes):
            worker = multiprocessing.Process(target=self.handle_tasks,
                                             args=(
                                                 self.tasks,
                                                 args,
                                                 ops,
                                                 model_meta_pb,
                                             ))
            # worker = threading.Thread(target=self.handle_tasks, args=(self.tasks,args,ops,model_meta_pb,))
            worker.daemon = True
            self.workers.append(worker)
            worker.start()

    def handle_tasks(self, tasks_queue, args, ops, model_meta_pb):
        """Worker thread."""
        while True:
            qitem = tasks_queue.get()
            if qitem is None:
                # None in the queue indicates the parent want us to exit
                # tasks_queue.task_done()
                break
            print(qitem.index, qitem.nodename)
            freeze_pb_process(args, ops, model_meta_pb, qitem)
            # tasks_queue.task_done()

    def enqueue(self, index, nodename):
        self.tasks.put(Item(index, nodename))

    def finish(self):
        # exit all threads
        for _ in self.workers:
            self.tasks.put(None)
        for worker in self.workers:
            worker.join()


def create_placeholder_op(node_name, dtype=None, shape=None):
    """Creates a placeholder op."""
    output_node = node_def_pb2.NodeDef()
    output_node.op = 'Placeholder'
    output_node.name = node_name
    output_node.attr['dtype'].CopyFrom(dtype)
    output_node.attr['value'].CopyFrom(shape)
    return output_node


def find_inputnodes_name(args, ops, model_meta_pb):
    tmp_ops = ops.copy()
    MPR = MultiProcessRunner(args, ops, model_meta_pb)
    for index in range(len(tmp_ops) - 1, -1, -1):
        if tmp_ops[index]['optype'] in DEL_TYPE_LIST:
            tmp_ops[index]['valid'] = False
            del tmp_ops[index]
        else:
            tmp_ops[index]['valid'] = True
    for index, l in enumerate(tmp_ops):
        if index < len(tmp_ops) * INVALID_OUTNODE_LOC_RATIO:
            continue


def inference_pb_process(args, model_pb, inputs, input_shapes, input_types,
                         outputs):
    pdb.set_trace() if args.debug else None
    print("------------------Start inference--------------------------")
    BTF = BackendTensorflow(args,
                            model_pb,
                            inputs=inputs.rstrip(';'),
                            input_shapes=input_shapes.rstrip(';'),
                            input_types=input_types.rstrip(';'),
                            outputs=outputs.rstrip(';'))
    try:
        tf.reset_default_graph()
        BTF.load()
        for i in range(args.iter):
            output = BTF.predict()
            print("------------------iter" + str(i) +
                  "--------------------------")
            for index, res in enumerate(BTF.outputs.split(";")):
                print(
                    "Inference result-[" + res + "] :",
                    output if not isinstance(output, list) else output[index])
                if isinstance(output, list):
                    print(
                        "Inference result-[" + res + "] shape:",
                        "0" if not isinstance(output[index], np.ndarray) else
                        output[index].shape)
                else:
                    print(
                        "Inference result-[" + res + "] shape:", "0" if
                        not isinstance(output, np.ndarray) else output.shape)
    except Exception as e:
        print("[ERROR]:Inference: inference pb failed! pb:", model_pb)
        print(e)
    print("------------------End inference--------------------------")


def find_outputnodes_name(args, ops, model_meta_pb):
    print(
        "----------------------------start freeze graph----------------------------------"
    )
    tmp_ops = ops.copy()
    tmp_outputnodes_list = ''
    replace_GPU(args.ckpt.rstrip('/') + '/graph.pbtxt', "GPU:0", "CPU:0")
    MPR = MultiProcessRunner(args, ops, model_meta_pb)
    for index in range(len(tmp_ops) - 1, -1, -1):
        if tmp_ops[index]['optype'] in DEL_TYPE_LIST:
            tmp_ops[index]['valid'] = False
            del tmp_ops[index]
        else:
            tmp_ops[index]['valid'] = True
    pdb.set_trace() if args.debug else None
    for index, l in enumerate(tmp_ops):
        if index < len(tmp_ops) * INVALID_OUTNODE_LOC_RATIO:
            continue
        for name in BLACK_NAME_PREFIX_LIST:
            if l['name'].startswith(name):
                l['valid'] = False
        for name in BLACK_NAME_INTER_LIST:
            if name in l['name']:
                l['valid'] = False
        if l['optype'] in BLACK_TYPE_LIST or len(l['outputLayer']) == 0:
            l['valid'] = False
        for n in range(len(l['outputLayer'])):
            if l['valid'] == False:
                break
            for ll in tmp_ops[index:]:
                for m in range(len(ll['inputLayer'])):
                    if l['outputLayer'][n] == ll['inputLayer'][m]:
                        l['valid'] = False
                for control_m in range(len(ll['control_inputLayer'])):
                    if l['outputLayer'][n] == ll['control_inputLayer'][
                            control_m]:
                        l['valid'] = False
                if l['valid'] == False:
                    break
        if l['name'] in WHITE_NAME_LIST or l['optype'] in WHITE_TYPE_LIST:
            l['valid'] = True
        if not len(FORCE_OUTPUT_NAME_LIST) == 0:
            l['valid'] = True if l['name'] in FORCE_OUTPUT_NAME_LIST else False
        if l['valid']:
            # if not find and l['valid'] and not len(l['outputLayer'])==0:
            print("[OK]:dump_tf.py:Find one outputnode:", l['name'],
                  l['optype'])
            if (args.type == 2):
                MPR.enqueue(index, l['name'])
    MPR.finish()
    del_invalid_pb(args.ckpt.rstrip('/'))
    '''
    tmp_outputnodes_list=tmp_outputnodes_list.rstrip(',')
    try:
        tf.reset_default_graph()
        freeze_graph.freeze_graph(args.ckpt.rstrip('/')+'/graph.pbtxt',
                    '', 
                    False,  
                    model_meta_pb[0:-5], 
                    tmp_outputnodes_list,'save/restore_all', 'save/Const:0', 
                    args.ckpt.rstrip('/')+'/frozen_model_All.pb',False,'')
        print("[OK]:dump_tf.py:frozen pb: ",args.ckpt.rstrip('/')+'/frozen_model_All.pb'," success!")
    except Exception as e:
        print("[ERROR]:dump_tf.py: freeze pb failed! nodename:",tmp_outputnodes_list)
        print(e)
    '''
    print(
        "----------------------------end freeze graph----------------------------------\n"
    )


def result_summary(args, optype):
    workbook = xlsxwriter.Workbook("Summary_Result_" + optype + ".xlsx")
    worksheet_summary = workbook.add_worksheet("Summary_op")
    worksheet_summary.write_row('A' + str(0 + 1), ["op"])
    worksheet_summary.write_row('B' + str(0 + 1), ["count"])
    worksheet_summary_multi = workbook.add_worksheet("Summary_multi_op")
    worksheet_summary_multi.write_row('A' + str(0 + 1), ["multi_op"])
    worksheet_summary_multi.write_row('B' + str(0 + 1), ["count"])
    for index, j in enumerate(list(G_dic.keys())):
        if not j == "NA":
            print(j, G_dic[j])
            worksheet_summary_multi.write_row('A' + str(index + 1), [j])
            worksheet_summary_multi.write_row('B' + str(index + 1),
                                              [str(G_dic[j])])
    for index, j in enumerate(list(op_dic.keys())):
        worksheet_summary.write_row('A' + str(index + 2), [j])
        worksheet_summary.write_row('B' + str(index + 2), [str(op_dic[j])])
    print("\nFind operator [", optype, "]'s location [", args.optype_location,
          "]: ", sum(list(G_dic.values())))
    workbook.close()
