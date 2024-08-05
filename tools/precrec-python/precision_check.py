import logging
import sys
import os
import re

import numpy as np

from utils import init_logger, parse_input_param, validate_path
from dump_info_check import DumpInfo
from data_set_check import BatchDataSet
from sparse_ckpt_check import SparseModel
from dense_ckpt_check import DenseModel
from op_check import OpData


OUTPUT_REGEX_STR = r'^\d{8}_\d{6}$'
DUMP_INFO_FUNC_KEY = "DumpInfo"
BATCH_DATASET_FUNC_KEY = "BatchDataSet"
SPARSE_MODEL_FUNC_KEY = "SparseModel"
DENSE_MODEL_FUNC_KEY = "DenseModel"
OP_DATA_FUNC_KEY = "Opdata"

class PrecisionData:
    def __init__(self, data_path, data_name):
        logging.info(f"DataDir of {data_name}:\n{data_path} ......\n")
        validate_path(data_path, "PRECISION DATA PATH", OUTPUT_REGEX_STR)
        self.data_path = data_path
        self.data_name = data_name
        self.data_parse_func_map = {DUMP_INFO_FUNC_KEY:self.parse_dump_info,
                                    BATCH_DATASET_FUNC_KEY:self.parse_batch_data,
                                    SPARSE_MODEL_FUNC_KEY:self.parse_sparse_model,
                                    DENSE_MODEL_FUNC_KEY :self.parse_dense_model,
                                    OP_DATA_FUNC_KEY:self.parse_op_data}

    def parse_dump_info(self):
        logging.info(f"=========== {self.data_name} [{DUMP_INFO_FUNC_KEY}] parsing start...... ===========")
        self.dump_info = DumpInfo(self.data_path)
        logging.info(f"=========== {self.data_name} [{DUMP_INFO_FUNC_KEY}] parsing succeed...... ===========")
        return self.dump_info

    def parse_batch_data(self):
        logging.info(f"=========== {self.data_name} [{BATCH_DATASET_FUNC_KEY}] parsing start...... ===========")
        self.batch_data_set = BatchDataSet(self.data_path)
        logging.info(f"=========== {self.data_name} [{BATCH_DATASET_FUNC_KEY}] parsing succeed...... ===========")
        return self.batch_data_set
    
    def parse_sparse_model(self):
        logging.info(f"=========== {self.data_name} [{SPARSE_MODEL_FUNC_KEY}] parsing start...... ===========")
        self.sparse_ckpt = SparseModel(self.data_path)
        logging.info(f"=========== {self.data_name} [{SPARSE_MODEL_FUNC_KEY}] parsing succeed...... ===========")
        return self.sparse_ckpt

    def parse_dense_model(self):
        logging.info(f"=========== {self.data_name} [{DENSE_MODEL_FUNC_KEY}] parsing start...... ===========")
        self.dense_ckpt = DenseModel(self.data_path)
        logging.info(f"=========== {self.data_name} [{DENSE_MODEL_FUNC_KEY}] parsing succeed...... ===========")
        return self.dense_ckpt

    def parse_op_data(self):
        logging.info(f"=========== {self.data_name} [{OP_DATA_FUNC_KEY}] parsing start...... ===========")
        self.op_data = OpData(self.data_path, self.dump_info)
        logging.info(f"=========== {self.data_name} [{OP_DATA_FUNC_KEY}] parsing succeed...... ===========")
        return self.op_data


def construct_precision_comparison(test_data, golden_data, select_func_list):
    for func_key in select_func_list:
        test_parsed_data = test_data.data_parse_func_map[func_key]()
        golden_parsed_data = golden_data.data_parse_func_map[func_key]()

        logging.info(f"===========[{func_key}] comparison start ===========")
        if test_parsed_data != golden_parsed_data:
            logging.error(f"===========[x] [{func_key}] does not match ===========\n")
        else:
            logging.info(f"===========[√] [{func_key}] match ===========\n")


if __name__ == "__main__":
    init_logger()
    test_path, golden_path = parse_input_param()


    test_data = PrecisionData(test_path, "[Test Data]")
    golden_data = PrecisionData(golden_path, "[Golden Data]")


    func_list = [DUMP_INFO_FUNC_KEY, BATCH_DATASET_FUNC_KEY, SPARSE_MODEL_FUNC_KEY,
                 DENSE_MODEL_FUNC_KEY, OP_DATA_FUNC_KEY]
    # func_list = [DUMP_INFO_FUNC_KEY, OP_DATA_FUNC_KEY]
    construct_precision_comparison(test_data, golden_data, func_list)
    # import pdb
    # pdb.set_trace()

















