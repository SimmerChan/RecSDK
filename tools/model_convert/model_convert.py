import argparse
import os
import re

import tensorflow as tf
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--input_path', type=str, required=True, help='path of the model file to be converted')
parser.add_argument('--output_path', type=str, required=True, help='output path of the converted model')

slice_prefix = "slice_"
data_suffix = ".data"
attribute_suffix = ".attribute"
min_file_size = 1
max_file_size = 1024 * 1024 * 1024 * 1024


class ModelConverter:
    def __init__(self, input_model_path, output_model_path):
        self._input_path = input_model_path
        self._output_path = output_model_path
        self._table_list = []
        self.table_info_dict = {}
        self.sparse_file_dir = None

        if not os.path.exists(self._input_path):
            raise FileNotFoundError(f"the input path {self._input_path} does not exists. please check it.")
        if not os.path.exists(self._output_path):
            os.makedirs(self._output_path)
        self._get_sparse_file_dir()
        self._build_table_info_dict()

    def convert(self):
        insert_op_list = []
        var_list = []
        hash_table_list = []
        # load old checkpoint and get var list
        if not os.path.exists(self._input_path):
            raise FileNotFoundError(f"the checkpoint path {self._input_path} does not exists.")
        ckpt = tf.train.load_checkpoint(self._input_path)
        var_names = ckpt.get_variable_to_shape_map().keys()
        var_values = [ckpt.get_tensor(name) for name in var_names]
        for i, name in enumerate(var_names):
            var = tf.Variable(var_values[i], name=name)
            var_list.append(var)

        # get key and embedding from file to insert hashtable
        for table_name, emb_size in self.table_info_dict.items():
            initialize_value = np.zeros((emb_size,))
            # create mutable hashtable
            if tf.__version__.startswith("2"):
                hash_table = tf.lookup.experimental.MutableHashTable(key_dtype=tf.int64, value_dtype=tf.float32,
                                                                     default_value=initialize_value, name=table_name)
            else:
                hash_table = tf.contrib.lookup.MutableHashTable(key_dtype=tf.int64, value_dtype=tf.float32,
                                                                default_value=initialize_value, name=table_name)

            key = get_key(self.sparse_file_dir, table_name)
            emb_data = get_embedding_array(self.sparse_file_dir, table_name)
            insert_op = hash_table.insert(tf.convert_to_tensor(key), tf.convert_to_tensor(emb_data))
            insert_op_list.append(insert_op)
            print("build save table:", table_name)
            hash_table_list.append(hash_table)
        if tf.__version__.startswith("2"):
            checkpoint = tf.train.Checkpoint(table_list=hash_table_list)
            manager = tf.train.CheckpointManager(checkpoint, directory=self._output_path, max_to_keep=5)
            manager.save()
        else:
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                sess.run(insert_op_list)
                saver = tf.train.Saver()
                saver.save(sess, self._output_path + "/model.ckpt-0")

    def _get_sparse_file_dir(self):
        latest_ckpt = self._get_latest_ckpt_name()
        latest_step = latest_ckpt.split("-")[-1]
        pattern = re.compile(r"^sparse-.*{}$".format(latest_step))
        for folder_name in os.listdir(self._input_path):
            if os.path.isdir(os.path.join(self._input_path, folder_name)) and pattern.match(folder_name):
                self.sparse_file_dir = os.path.join(self._input_path, folder_name)

    def _get_latest_ckpt_name(self):
        ckpt_path = os.path.join(self._input_path, "checkpoint")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"the input path you provided {ckpt_path} miss checkpoint file.please check it.")
        with open(ckpt_path, "r") as fin:
            # validate open file
            validate_read_file(ckpt_path)
            latest_ckpt = fin.readline().rstrip()
            latest_ckpt = latest_ckpt.split(":")[1].strip(' ').replace('"', '')
            latest_ckpt = latest_ckpt.split("/")[-1]
        return latest_ckpt

    def _build_table_info_dict(self):
        table_list = []
        if not os.path.exists(self.sparse_file_dir):
            raise FileNotFoundError(f"the sparse file path {self.sparse_file_dir} does not exists.")
        for _, table_name, _ in os.walk(self.sparse_file_dir):
            table_list.append(table_name)

        if not table_list:
            raise FileNotFoundError(f"under the sparse file path {self.sparse_file_dir}, no file exists.")

        self._table_list = table_list[0]
        for table_name in self._table_list:
            table_path = os.path.join(self.sparse_file_dir, table_name, "embedding")
            attribute_file = get_attribute_and_data_file(table_path)[0]
            with open(attribute_file, "r") as fin:
                validate_read_file(attribute_file)
                attributes = np.fromfile(attribute_file, dtype=np.uint64)
                data_shape = attributes[:2]
                self.table_info_dict[table_name] = data_shape[1]


def get_key(self, sparse_file_path, table_name):
    upper_dir = generate_upper_dir(sparse_file_path, table_name, "key")
    attribute_data_dir, target_data_dir = get_attribute_and_data_file(upper_dir)

    with open(attribute_data_dir, "r") as fin:
        validate_read_file(attribute_data_dir)
        attributes = np.fromfile(attribute_data_dir, dtype=np.uint64)
    data_shape = attributes[0]

    with open(target_data_dir, "r") as fin:
        validate_read_file(target_data_dir)
        key_data = np.fromfile(target_data_dir, dtype=np.int64)
    key_data = key_data.reshape(data_shape)
    return key_data


def get_embedding_array(self, sparse_file_path, table_name):
    upper_dir = generate_upper_dir(sparse_file_path, table_name, "embedding")
    attribute_data_dir, target_data_dir = get_attribute_and_data_file(upper_dir)
    with open(attribute_data_dir, "r") as fin:
        validate_read_file(attribute_data_dir)
        attributes = np.fromfile(attribute_data_dir, dtype=np.uint64)
        data_shape = attributes[:2]
    with open(target_data_dir, "r") as fin:
        validate_read_file(target_data_dir)
        emb_data = np.fromfile(target_data_dir, dtype=np.float32)

    emb_data = emb_data.reshape(data_shape)
    return emb_data


def get_attribute_and_data_file(table_path):
    if not os.path.exists(table_path):
        raise FileNotFoundError(f"the input table path {table_path} does not exists.")

    attribute_file_list = []
    data_file_list = []
    for file_name in os.listdir(table_path):
        if file_name.endswith(attribute_suffix):
            attribute_file_list.append(file_name)
        if file_name.endswith(data_suffix):
            data_file_list.append(file_name)
    if len(attribute_file_list) != 1:
        raise AssertionError(f"under the table path {table_path}, ther must only one attribute file. "
                             f"In fact, {len(attribute_file_list)} attribute file exists. ")
    if len(data_file_list) != 1:
        raise AssertionError(f"under the table path {table_path}, ther must only one data file. "
                             f"In fact, {len(data_file_list)} data file exists. ")
    attribute_file = os.path.join(table_path, attribute_file_list[0])
    data_file = os.path.join(table_path, data_file_list[0])
    return attribute_file, data_file


def generate_upper_dir(sparse_file, table_name, data_type):
    temp_dir = sparse_file
    return os.path.join(temp_dir, table_name, data_type)


def validate_read_file(read_path):
    if os.path.islink(read_path):
        raise ValueError(f"the path {read_path} to be read is soft link.")
    file_stat = tf.io.gfile.stat(read_path)
    if not min_file_size < file_stat.length <= max_file_size:
        raise ValueError(f"file size: {file_stat.length} is invalid, not in ({min_file_size}, {max_file_size}]")


if __name__ == "__main__":
    args = parser.parse_args()
    convert_instance = ModelConverter(input_model_path=args.input_path, output_model_path=args.output_path)
    convert_instance.convert()
    print("convert model success.")