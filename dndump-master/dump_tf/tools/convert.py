import tensorflow as tf
from tensorflow.python.platform import gfile
from google.protobuf import text_format
import sys

model_pb = sys.argv[1]
convert = sys.argv[2]
netname = model_pb.split('/')[-1].split('.')[0]


def convert_pb_to_pbtxt(filename):
    with tf.gfile.GFile(filename, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')
        tf.train.write_graph(graph_def,
                             './',
                             str(netname) + '.pbtxt',
                             as_text=True)
    return


def convert_pbtxt_to_pb(filename):
    #     """Returns a `tf.GraphDef` proto representing the data in the given pbtxt file.
    #     Args:
    #       filename: The name of a file containing a GraphDef pbtxt (text-formatted
    #         `tf.GraphDef` protocol buffer data).
    #     """
    with tf.gfile.FastGFile(filename, 'r') as f:
        graph_def = tf.GraphDef()
        file_content = f.read()
        # Merges the human-readable string in `file_content` into `graph_def`.
        text_format.Merge(file_content, graph_def)
        tf.train.write_graph(graph_def,
                             './',
                             str(netname) + '.pb',
                             as_text=False)
    return


if convert == '0':
    convert_pb_to_pbtxt(model_pb)
else:
    convert_pbtxt_to_pb(model_pb)
