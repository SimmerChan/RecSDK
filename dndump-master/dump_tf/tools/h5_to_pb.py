import sys
from tensorflow.keras.models import load_model
import tensorflow as tf
import os
import os.path as osp
from tensorflow.keras import backend as K
import argparse


def get_args():
    """Parse commandline."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5", default='my_model.h5', help="input: h5 file")
    parser.add_argument("--pb", default='my_model.pb', help="output: pb file")
    args = parser.parse_args()

    return args


args = get_args()
print(args)

model_pb = args.pb
model_h5 = args.h5


def freeze_session(session,
                   keep_var_names=None,
                   output_names=None,
                   clear_devices=True):
    from tensorflow.python.framework.graph_util import convert_variables_to_constants
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(
            set(v.op.name
                for v in tf.global_variables()).difference(keep_var_names
                                                           or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = convert_variables_to_constants(session, input_graph_def,
                                                      output_names,
                                                      freeze_var_names)
    return frozen_graph


input_fld = sys.path[0]
weight_file = model_h5
output_graph_name = model_pb

output_fld = input_fld + '/'
weight_file_path = osp.join(input_fld, weight_file)
K.set_learning_phase(0)
net_model = load_model(weight_file_path)
print('input is :', net_model.input.name)
print('output is:', net_model.output.name)
sess = K.get_session()
frozen_graph = freeze_session(K.get_session(),
                              output_names=[net_model.output.op.name])
from tensorflow.python.framework import graph_io
graph_io.write_graph(frozen_graph,
                     output_fld,
                     output_graph_name,
                     as_text=False)
print('saved the constant graph (ready for inference) at: ',
      osp.join(output_fld, output_graph_name))
