"""
tensorflow backend (https://github.com/tensorflow/tensorflow)
"""

# pylint: disable=unused-argument,missing-docstring,useless-super-delegation

import tensorflow as tf
from tensorflow.core.framework import graph_pb2

import sys, os
import time
import numpy as np

DataType = {
    "0": None,
    "1": np.float32,
    "2": np.float64,
    "3": np.int32,
    "4": np.uint8,
    "5": np.int16,
    "6": np.int8,
    "7": None,
    "8": np.complex64,
    "9": np.int64,
    "10": np.bool,
    "11": None,
    "12": None,
    "13": None,
    "14": None,
    "15": None,
    "16": None,
    "17": np.uint16,
    "18": np.complex128,
    "19": np.float16,
    "20": None,
    "21": None,
    "22": np.uint32,
    "23": np.uint64,
}


class BackendTensorflow():
    def __init__(self,
                 args,
                 model=None,
                 inputs=None,
                 input_shapes=None,
                 input_types=None,
                 outputs=None):
        self.outputs = outputs if outputs is not None else None
        self.inputs = inputs if inputs is not None else None
        self.input_shapes = input_shapes if input_shapes is not None else None
        self.input_types = input_types if input_types is not None else None
        self.model = model if model is not None else None

    def version(self):
        return tf.__version__ + "/" + tf.__git_version__

    def name(self):
        return "tensorflow"

    def image_format(self):
        # By default tensorflow uses NHWC (and the cpu implementation only does NHWC)
        return "NHWC"

    def load(self):
        # there is no input/output meta data i the graph so it need to come from config.
        if not self.inputs:
            raise ValueError("BackendTensorflow needs inputs")
        if not self.outputs:
            raise ValueError("BackendTensorflow needs outputs")

        graph_def = graph_pb2.GraphDef()
        with open(self.model, "rb") as f:
            graph_def.ParseFromString(f.read())
        g = tf.import_graph_def(graph_def, name='')
        self.sess = tf.Session(graph=g)
        self.create_random_feed()
        return self

    def create_random_feed(self):
        self.feed = {}
        inputs_list = self.inputs.split(";")
        input_shapes_list = self.input_shapes.split(";")
        input_types_list = self.input_types.split(";")
        for i in range(len(inputs_list)):
            shape = []
            for d in input_shapes_list[i].strip(",").split(","):
                shape.append(int(d) if not d == "-1" else 1)
            self.feed[inputs_list[i]] = np.random.randint(0, 10, shape).astype(
                DataType[input_types_list[i]])
        '''
        for name,shape in zip(self.inputs.split(";"),self.input_shapes.split(";")):
            shape=[int(i) for i in shape.strip(",").split(",")]
            self.feed[name]=np.random.randint(0, 10, shape)
        '''

    def predict(self, feed=None):
        # normal
        result = self.sess.run(self.outputs, feed_dict=self.feed)
        return result


'''
BTF=BackendTensorflow()
BTF.load(args)
for i in range(args.iter):
    output=BTF.predict()
    print("------------------iter"+str(i)+"--------------------------")
    for index,res in enumerate(BTF.outputs):
        print("Inference result-["+res+"] :",output[index])
        print("Inference result-["+res+"] shape:",output[index].shape)
'''
