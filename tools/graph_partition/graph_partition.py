import tensorflow as tf
from tensorflow.contrlib import graph_editor as ge
class GraphPartitioner:
    def __init__(self):
        self.signature_def = None
        self.graph = None
        self.op_node_lookup = dict()
        self.input_op_nodes = []
        self.output_op_nodes = []
        self.tensor_node_lookup = dict()
        self.heavy_load_ops = []
        self.embedding_lookup_op_type = None
        self.first_heavy_load_on_spares_path = set()
        self.first_op_after_lookup = []
        self.seen = set()
        self.post_out = set()
        self.partition_to_first_heavy_load = False

        self.sparse_lookup_ops = []
        self.sparse_lookup_tensors = []
        self.input_nodes = []
        self.output_nodes = []

    def set_emedding_lookup_op_type(self,s):
        self.embedding_lookup_op_type = s

    def has_gary_downstreams(self, op):
        gary_list = ["DynamicPartition"]
        down_ops = ge.get_forward_walk_ops([op])
        for op in down_ops:
            if op.type  in gary_list:
                return True
        return False

    def get_sub_graph(self):
        print("build graph...")
        in_str = set()
        out_str = set()
        inputs_set = set()
        outputs = set()
        for op in self.graph.get_oprations():
            if self._is_embedding_lookup(op):
                self.sparse_lookup_ops.append(op)
        if not self.sparse_lookup_ops:
            print("aaaaaaaaaaaaaaaaaa")
            for op in self.graph.get_oprations():
                is_top_op = True
                for op1 in self.graph.get_oprations():
                    for tensor in op1.outputs:
                        if tensor in op.inputs:
                            is_top_op = False
                            break
                    if not is_top_op:
                        break
                if is_top_op:
                    self.sparse_lookup_ops.append(op)
        check_ops = self.sparse_lookup_ops
        self.sparse_lookup_ops = []
        for op in check_ops:
            if not self.has_gary_downstreams(op):
                self.sparse_lookup_ops.append(op)
                self.sparse_lookup_tensors.extend(op.outputs)

        print(self.sparse_lookup_ops)
        for op in self.graph.get_oprations():
            for tensor in self.sparse_lookup_tensors:
                if tensor in op.inputs:
                    self.input_nodes.append(op)
        for k,v in self.signature_def.output.items():
            op_name = str(v).split("\n")[0].replace(' ', "").replace('"',"").split(":")[1].split(":")[0]
            for op in self.graph.get_oprations();
            if op.name == op_name:
                self.output_nodes.append(op)

        floatUps = []
        to_expend = []
        in_str = []

        for op in self.input_nodes:
            if op.type not in floatUps:
                if op.name not in in_str:
                    in_str.append(op.name)
            else:
                to_expend.append(op)

        while to_expend:
            candidates = []
            for top in to_expend:
                for op in self.graph.get_oprations():
                    for tensor in op.inputs:
                        if tensor in top.outputs:
                            candidates.append(op)
            to_expend = []
            for op in candidates:
                if op.type not in floatUps:
                    in_str.append(op.name)
                else:
                    to_expend.append(op)
        return str(in_str), str([op.name for op in self.output_nodes])

    def _is_embedding_lookup(self, op):
        if op.type in self.embedding_lookup_op_type:
            return True

        return False

    def _check_op_status(self):
        unseen_list = []
        for name, op_node in self.op_node_lookup.items():
            if not op_node.seen:
                unseen_list.append(name)
        return unseen_list
