import json
import logging
import subprocess
import sys
import os
import re
import numpy as np

from dataclasses import dataclass


from utils import validate_path, nested_dict_to_str


DUMP_OP_NUMPY_REGEX_STR = r'^.+\.npy$'
OP_NUMPY_ATOL = 1e-10

DUMP_NP_LEN = 8

INSTRUCT_PYTHON = "python3"
INSTRUCT_CONVERT = "convert"
INSTRUCT_D =  "-d"
INSTRUCT_OUT = "-out"

ASCEND_TOOLKIT_MSACCUCMP_PATH = "/usr/local/Ascend/ascend-toolkit/latest/tools/operator_cmp/compare/msaccucmp.py"

DYN_EXP_OP_LIST = set(["EmbeddingLookupByAddress", "EmbeddingUpdateByAddress"])
NO_DYN_OP_LIST = set(["GatherV2", "ScatterNdAdd"])

DYN_ARCH_OP_TYPE = "EmbeddingLookupByAddress"
NODYN_ARCH_OP_TYPE = "GatherV2"

EMB_LOOK_OPS_KEY = "emb_look_ops"
EMB_UPDATE_OPS_KEY = "emb_update_ops"

OP_TYPE = "op_type"
OP_DATA_PATH = "op_data_path"
DATA_TYPE = "data_type"
DATA_INDEX = "data_index"

LOOKUP_RESULT = "lookup_result"
UPDATE_GRAD = "update_grad"


NODYN_STAMP_CONSTRUCT_DICT = {"lookup_result_stamp": ["output", "0"], 
"update_grad_stamp": ["input", "2"]}
DYN_STAMP_CONSTRUCT_DICT = {"lookup_result_stamp": ["output", "0"],
"update_grad_stamp": ["input", "1"],}


class OpData:
    def __init__(self, data_dir, dump_info):
        self.dump_data_path = find_match_op_dump_data(data_dir, 0, 0)
        self.dump_info = dump_info
        validate_path(ASCEND_TOOLKIT_MSACCUCMP_PATH, "msaccucmp.py")
        self.output_numpy_path = exe_msaccucmp_convert(self.dump_data_path)
        self.op_data_dict = self.parse_numpy_data()
        # print(nested_dict_to_str(self.op_data_dict))

    def __eq__(self, other):
        test_op_data_dict = self.op_data_dict
        golden_op_data_dict = other.op_data_dict

        test_op_data_keys = test_op_data_dict.keys()
        golden_op_data_keys = golden_op_data_dict.keys()
        test_op_list = sorted(list(test_op_data_keys))
        golden_op_list = sorted(list(golden_op_data_keys))

        if test_op_list != golden_op_list:
            logging.error(f"[OpData]Test data and Golden data should have the same names, " \
                        + f"but Test:{test_op_list} Golden:{golden_op_list} are given.")
            return False

        for table_name in golden_op_list:
            test_table_data = test_op_data_dict[table_name]
            golden_table_data = golden_op_data_dict[table_name]
            for emb_type in golden_table_data.keys():
                test_emb_data = test_table_data[emb_type][0]
                golden_emb_data = golden_table_data[emb_type][0]

                logging.debug(f"[OpData][{table_name}][{emb_type}] Data are shown as below.\n"
                                  f"Test: {test_emb_data.dtype} {test_emb_data.shape}\n"
                                  f"Golden: {golden_emb_data.dtype} {golden_emb_data.shape}\n"
                                  f"Test: {test_emb_data}\n"
                                  f"Golden: {golden_emb_data}\n")

                if test_emb_data.dtype != golden_emb_data.dtype:
                    logging.error(f"[OpData][{table_name}][{emb_type}] Test and Golden shape not equal.\n"
                                  f"Test:{test_emb_data.dtype}\n"
                                  f"Golden:{golden_emb_data.dtype}\n")
                    return False
                  
                if test_emb_data.shape != golden_emb_data.shape:
                    logging.error(f"[OpData][{table_name}][{emb_type}] Test and Golden shape not equal.\n"
                                  f"Test:{test_emb_data.shape}\n"
                                  f"Golden:{golden_emb_data.shape}\n")
                    return False

                if not np.allclose(test_emb_data, golden_emb_data, OP_NUMPY_ATOL):
                    logging.error(f"[OpData][{table_name}][{emb_type}] Test and Golden value not equal.\n"
                                  f"Test:{test_emb_data}\n"
                                  f"Golden:{golden_emb_data}\n")
                    return False
        return True

    def parse_numpy_data(self):
        for dirpath, dirs, files in os.walk(self.output_numpy_path):
            if dirs:
                logging.warning(f"Dump op numpy path should not contain any directory, your file may have been tampered: {self.output_numpy_path}")
            files = sorted(files)

            logging.debug(f"parsing numpy op data to desc start......\nNumpy dir:{dirpath}  Numpy files: {files}.")
            op_numpy_des_dict = self.parse_op_numpy_to_desc(dirpath, files)
            logging.debug(f"Parsing numpy op data to desc succeed.")

            # print(nested_dict_to_str(op_numpy_des_dict))

            op_numpy_data_dict = parse_op_desc_to_data(self.dump_info.dump_emb_op_info, op_numpy_des_dict)
        return op_numpy_data_dict


    def parse_op_numpy_to_desc(self, dir_path, file_names):
        op_desc_dict = {}
        for filename in file_names:
            cur_op_data_path=os.path.join(dir_path, filename)
            validate_path(cur_op_data_path, "op_dump_numpy", DUMP_OP_NUMPY_REGEX_STR)

            file_name_split = filename.split(".")
            if len(file_name_split) != DUMP_NP_LEN:
                raise ValueError(f"Dump op numpy may have been tamperd or msaccucmp updated. Path:{cur_op_data_path}")

            cur_op_type = file_name_split[0]
            cur_op_name = file_name_split[1]
            cur_op_data_type = file_name_split[-3]
            cur_op_data_index= file_name_split[-2]

            op_desc = {DATA_TYPE:cur_op_data_type,
                    DATA_INDEX:cur_op_data_index,
                    OP_TYPE:cur_op_type,
                    OP_DATA_PATH:cur_op_data_path}
            if not op_desc_dict.get(cur_op_name):
                op_desc_dict[cur_op_name] = []
            
            op_desc_dict[cur_op_name].append(op_desc)
            logging.debug(f"Parsing op name to op desc succeed. Cur_op_name:{cur_op_name} Op_desc: {op_desc}.")


        op_type_set = set([op_desc[0][OP_TYPE] for _, op_desc in op_desc_dict.items()])

        if op_type_set == DYN_EXP_OP_LIST:
            self.use_dyn_exp = True
        elif op_type_set == NO_DYN_OP_LIST:
            self.use_dyn_exp = False

        op_numpy_desc_dict = div_op_desc_by_table(self.use_dyn_exp, op_desc_dict)
        return op_numpy_desc_dict

def convert_dump_op_to_numpy(data_path, rank_id, step):
    dump_data_path = find_match_op_dump_data(data_path, rank_id, step)
    output_numpy_path = exe_msaccucmp_convert(dump_data_path)
    return output_numpy_path

def find_match_op_dump_data(dump_data_path, rank_id, step):
    # 算子最终要去解析的文件是 /xxxx/20240724_141123/03dump_op/20240724141134/0/ge_default_20240724141135_31/4/0
    pattern_str = (
        f'^{re.escape(dump_data_path)}/'  # 匹配 precision check的位置:/xxxx/20240724_141123
        r'03dump_op/'  # 匹配 03dump_op
        r'\d{14}/'  # 匹配 日期: 20240724141134
        f'{re.escape(str(rank_id))}/'  # 匹配 rankid: 0
        r'ge_default_\d{14}_\d+/'  # 匹配ge_default_后跟日期和时间戳
        r'\d+/'  # 匹配 model id: 4
        f'{re.escape(str(step))}'  # 匹配 step: 0
    )
    pattern = re.compile(pattern_str)
    for root, dirs, files in os.walk(dump_data_path):
        for dir in dirs:
            op_path = os.path.join(root, dir)
            if pattern.match(op_path):
                logging.debug(f"Find matched Dump op path: {op_path}")
                return op_path

def exe_msaccucmp_convert(op_data_path):
    output_numpy_path = os.path.join(op_data_path, "dump_op_np")
    logging.info(f"convert target path: {output_numpy_path}")

    if os.path.exists(output_numpy_path):
        logging.warning(f"Dump op data have already been parsed and Convertion stop!!! This may cause some mistakes, please check the path: {output_numpy_path}")
        return output_numpy_path
    else:
        os.makedirs(output_numpy_path)
        logging.debug(f"Dump op parse output dir created, path: {output_numpy_path}")


    instruct_item_command = [INSTRUCT_PYTHON, ASCEND_TOOLKIT_MSACCUCMP_PATH, INSTRUCT_CONVERT,
                             INSTRUCT_D, op_data_path, INSTRUCT_OUT, output_numpy_path]
    logging.debug(f"msaccucmp convert exec instruction: {instruct_item_command}")
    
    convert_result = subprocess.run(instruct_item_command, capture_output=True, text=True)

    # 检查命令是否成功执行
    if convert_result.returncode == 0:
        logging.info(f"Msaccucmp convert dump op to numpy succeed.")
    else:
        raise ValueError(f"Msaccucmp convert dump op to numpy Failed!\n" \
                        + f"{convert_result.stdout}")

    return output_numpy_path

def parse_dump_json_info(data_path, json_name):
    json_info_path = os.path.join(data_path, json_name)
    if  not os.path.exists(json_info_path):
        raise ValueError(f"{json_name} exist! Your files may have been tampered:{json_info_path}")
    with open(json_info_path, 'r') as f:
        json_info = json.load(f)
    return json_info

def div_op_desc_by_table(use_dyn_exp, op_desc_dict):
    table_op_desc_dict = {"use_dyn_exp": use_dyn_exp}
    table_name_set = set()

    if use_dyn_exp:
        arch_op_type = DYN_ARCH_OP_TYPE
    else:
        arch_op_type = NODYN_ARCH_OP_TYPE

    op_name_list = sorted(list(op_desc_dict.keys()))
    logging.debug(f"Div op desc by table start......\nUse_dyn_exp:{use_dyn_exp} Op_name_list:{op_name_list}")

    for op_name in op_name_list:
        op_desc_list = op_desc_dict[op_name]
        if op_desc_list[0][OP_TYPE] == arch_op_type :
            if not op_name.startswith("LazyAdam"):
                table_name = op_name.split("__")[0]
            else:
                op_name_str = op_name.split("_")
                table_name = "_".join([op_name_str[3], op_name_str[4]])
            
            table_name_set.add(table_name)
            table_op_desc_dict[table_name] = {}
    
    logging.debug(f"Parsing table names succeed.Table list: \n{table_name_set}")
    

    for table_name in table_name_set:
        for op_name in op_name_list:
            if table_name in op_name:
                table_op_desc_dict[table_name][op_name] = op_desc_dict[op_name]
    logging.debug(f"Div op desc by table succeed.\nTable_op_desc_dict:{nested_dict_to_str(table_op_desc_dict)}")
    return table_op_desc_dict

@dataclass
class TABLE_EMB_DATA:
    lookup_result: np.array
    update_grad: np.array

def parse_op_desc_to_data(dump_emb_op_info, op_numpy_des_dict):
    # dump_emb_op_info
    # {"user_table": {"emb_look_ops": ["user_table//user_table_lookup/gather_for_id_offsets", "LazyAdam_0/update_user_table/GatherV2", "LazyAdam_0/update_user_table/GatherV2_1"], "emb_update_ops": ["LazyAdam_0/update_user_table/ScatterNdAdd", "LazyAdam_0/update_user_table/ScatterNdAdd_1", "LazyAdam_0/update_user_table/ScatterNdAdd_2"]}, 
    #  "item_table": {"emb_look_ops": ["item_table//item_table_lookup/gather_for_id_offsets", "LazyAdam_0/update_item_table/GatherV2", "LazyAdam_0/update_item_table/GatherV2_1"], "emb_update_ops": ["LazyAdam_0/update_item_table/ScatterNdAdd", "LazyAdam_0/update_item_table/ScatterNdAdd_1", "LazyAdam_0/update_item_table/ScatterNdAdd_2"]}}

    # op_numpy_des_dict
    # {"use_dyn_exp":bool
    #  "user_table":[OP_FILE_DESC], 
    #  "item_table":[OP_FILE_DESC]}
    table_list, dump_ops_info = parse_dump_data(dump_emb_op_info, op_numpy_des_dict)

    table_emb_dict = {}

    for table_name in table_list:
        table_op_info_list = dump_ops_info[table_name]
        table_op_des_list = op_numpy_des_dict[table_name]
        use_dyn_exp = op_numpy_des_dict["use_dyn_exp"]
        emb_data_dict = parse_table_des_to_data(table_op_info_list, table_op_des_list, use_dyn_exp)
        table_emb_dict[table_name] = emb_data_dict
        
    return table_emb_dict


def parse_dump_data(dump_emb_op_info, op_numpy_des_dict):
    info_tables = sorted(dump_emb_op_info.keys())
    data_tables_list = list(op_numpy_des_dict.keys())
    data_tables_list.remove("use_dyn_exp")
    data_tables = sorted(data_tables_list)
    
    if info_tables != data_tables:
        raise ValueError(f"dump info and dump data table names not match! Your file may have been tampered.\n"
                         f"Info:{info_tables}\nData:{data_tables}")
    
    info_ops_list = []
    data_ops_list = []

    for table in info_tables:
        temp_emb_look_ops = dump_emb_op_info[table][EMB_LOOK_OPS_KEY]
        dump_emb_op_info[table][EMB_LOOK_OPS_KEY] = [ops_name.replace("/", "_") for ops_name in temp_emb_look_ops]

        temp_emb_update_ops = dump_emb_op_info[table][EMB_UPDATE_OPS_KEY]
        dump_emb_op_info[table][EMB_UPDATE_OPS_KEY] = [ops_name.replace("/", "_") for ops_name in temp_emb_update_ops]

        temp_info_ops_name = dump_emb_op_info[table][EMB_LOOK_OPS_KEY] + dump_emb_op_info[table][EMB_UPDATE_OPS_KEY]
        info_ops_list.extend(temp_info_ops_name)

        temp_data_ops_name = [op_name for op_name, _ in op_numpy_des_dict[table].items()]
        data_ops_list.extend(temp_data_ops_name)

    info_ops_list = set(info_ops_list)
    data_ops_list = set(data_ops_list)

    if info_ops_list != data_ops_list:
        raise ValueError(f"dump info and dump data ops data names not match! Your file may have been tampered.\n"
                         f"Info:{info_ops_list}\nData:{data_ops_list}")
    
    return data_tables, dump_emb_op_info

def parse_table_des_to_data(table_op_info_list, table_op_des_dict, use_dyn_exp):
    emb_look_ops_names = table_op_info_list[EMB_LOOK_OPS_KEY]
    emb_update_ops = table_op_info_list[EMB_UPDATE_OPS_KEY]

    stamp_path_pair = construct_stamp_path_dict(table_op_des_dict)
    lookup_result = construc_numpy_data(emb_look_ops_names, stamp_path_pair, "lookup_result_stamp", use_dyn_exp)
    update_grad = construc_numpy_data(emb_update_ops, stamp_path_pair, "update_grad_stamp", use_dyn_exp)
    emb_data_dict = {LOOKUP_RESULT: (lookup_result, lookup_result.shape),
                    UPDATE_GRAD: (update_grad, update_grad.shape)}
    return emb_data_dict

def construc_numpy_data(ops_names, stamp_path_pair, stamp_type, use_dyn_exp):
    lookup_table_stamps = construc_stamp(ops_names, stamp_type, use_dyn_exp)
    numpy_list = []

    for lookup_table_stamp in lookup_table_stamps:
        temp_nump = np.load(stamp_path_pair[lookup_table_stamp])
        numpy_list.append(temp_nump)
    numpy_data = np.concatenate(numpy_list, axis=1)
    return numpy_data

def construc_stamp(ops_names, stamp_type, use_dyn_exp):
    stamp_list = []

    if use_dyn_exp:
        construct_dict = DYN_STAMP_CONSTRUCT_DICT
    else:
        construct_dict = NODYN_STAMP_CONSTRUCT_DICT
    
    stamp_data_type = construct_dict[stamp_type][0]
    stamp_data_index = construct_dict[stamp_type][1]

    for op_name in ops_names:
        stamp_name = ".".join([op_name, stamp_data_type, stamp_data_index])
        stamp_list.append(stamp_name)
    
    return stamp_list

def construct_stamp_path_dict(table_op_des_dict):
    stamp_value_dict = {}
    for op_name, op_des_list in table_op_des_dict.items():
        for op_des in op_des_list:
            op_stamp = ".".join([op_name, op_des[DATA_TYPE], op_des[DATA_INDEX]])
            stamp_value_dict[op_stamp] = op_des[OP_DATA_PATH]
    return stamp_value_dict



        






