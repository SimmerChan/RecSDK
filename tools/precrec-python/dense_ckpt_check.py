import copy
import logging
import os
import numpy as np
import tensorflow as tf
from utils import nested_dict_to_str


DENSE_ALLCLOSE_RTOL = 1e-10


class DenseModel:
    def __init__(self, data_dir):
        self.dense_path = os.path.join(data_dir, "02dump_model", "model-0")

        var_list = tf.train.list_variables(self.dense_path)
        self.var_name_list = [var_item[0] for var_item in var_list]
        self.var_dict = {}

        for var_name in self.var_name_list:
            tensor = tf.train.load_variable(self.dense_path, var_name)
            self.var_dict[var_name] = tensor

    def __eq__(self, other):
        if self.var_name_list != other.var_name_list:
            logging.error(f"Dense ckpt var items not equal!\n" \
                          f"Test var_name_list:{self.var_name_list}\n" \
                          f"Golden var_name_list:{other.var_name_list}\n")
            return False
        
        for var_name in self.var_name_list:
            test_var = self.var_dict[var_name]
            golden_var = other.var_dict[var_name]
            if test_var.shape != golden_var.shape:
                logging.error(f"[DenseModel]Test and Golden shape not equal!Variable name:{var_name}\n"
                                f"Test:{test_var.shape}\n"
                                f"Golden:{golden_var.shape}\n")
                return False
            
            if not np.allclose(test_var, golden_var, rtol=DENSE_ALLCLOSE_RTOL):
                logging.error(f"[DenseModel]Test and Golden value not equal!Variable name:{var_name}\n" 
                              f"Test var_name_list:\n{test_var}"
                              f"Golden var_name_list:\n{golden_var}\n")
                return False
        return True
        
