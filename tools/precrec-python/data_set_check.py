import logging
import sys
import os
import re

import numpy as np

from utils import validate_path


DATASET_REGEX_STR = r'^data_batch_\d+_.+\.npy$'
DATA_SET_NUMPY_ATOL = '1e-10'


class BatchDataSet:
    def __init__(self, data_dir):
        self.batch_data_path = os.path.join(data_dir, "01dump_dataset")
        self.batch_data_name_list = self.get_dump_data_names()
        self.bathc_data = self.parse_batch_data()


    def __eq__(self, other):
        logging.info(f"[BatchDataSet] comparison start......")
        if not isinstance(other, BatchDataSet):
            logging.error(f"[BatchDataSet] comparison must between BatchDataSet, but {other.__class__} is given")
            return False
        
        if self.batch_data_name_list != other.batch_data_name_list:
            logging.error(f"[BatchDataSet] comparison must content same batch data files, " 
                          f"Test: {self.batch_data_name_list} Gold: {other.batch_data_name_list}")
            return False

        for batch_data_name in self.batch_data_name_list:
            test_data = self.bathc_data[batch_data_name]
            golden_data = other.bathc_data[batch_data_name]
            if not np.allclose(test_data, golden_data, rtol=float(DATA_SET_NUMPY_ATOL)):
                logging.error(f"[BatchDataSet] data different, batch data name {batch_data_name}" 
                              f"Test data: {test_data} Gold data: {golden_data}")
                return False

        return True

    def get_dump_data_names(self):
        dataset_file_names = []
        for dirpath , _, files in os.walk(self.batch_data_path):
            for filename in files:
                file_path = os.path.join(dirpath , filename)
                validate_path(file_path, "dump_dataset", DATASET_REGEX_STR)
                dataset_file_names.append(filename)

        dataset_file_names = sorted(dataset_file_names)
        return dataset_file_names
    
    def parse_batch_data(self):
        batch_data_dict = {}
        for batch_data_name in self.batch_data_name_list:
            batch_data_path = os.path.join(self.batch_data_path, batch_data_name)
            batch_data = np.load(batch_data_path)
            batch_data_dict[batch_data_name] = batch_data
        return batch_data_dict