#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2022-2025. All rights reserved.
import os
import numpy as np


def emb_read(out_dir ="save_dir", table_names = ["table1", "table2"], ranks = [0]):
    def do_in(table_name, rank, slice_name, dtype_in=np.float32):
        attribute_path = os.path.join(out_dir, table_name, f"rank{rank}", slice_name, "slice.attribute")
        data_path = os.path.join(out_dir, table_name, f"rank{rank}", slice_name, "slice.data")
        attr = np.fromfile(attribute_path, dtype=np.int64)
        data = np.fromfile(data_path, dtype=dtype_in).reshape(*attr[1:])
        return attr, data
    embedding_dt = dict()
    for table in table_names:
        tabledt = dict()
        print(f"--------------------{table}-----------------")
        for rank in ranks:
            rankdt = dict()
            print(f"--------------------rank{rank}-----------------")
            attrkey, datakey = do_in(table, rank, "key", np.int64)
            attrembedding, dataembedding =  do_in(table, rank, "embedding")
            attrmomentum1, datamomentum1 =  do_in(table, rank, "momentum1")
            # 打印所有返回值的形状
            print(f"--------------------key-----------------")
            print("attrkey.shape ", attrkey.shape)
            print("attrkey.value ", attrkey)
            print("datakey.shape ", datakey.shape)
            print("datakey.value ", datakey)

            print(f"--------------------embedding-----------------")
            print("attrembedding.shape ", attrembedding.shape)
            print("attrembedding.value ", attrembedding)
            print("dataembedding.shape ", dataembedding.shape)
            print("dataembedding.value ", dataembedding)

            print(f"--------------------attrmomentum1-----------------")
            print("attrmomentum1.shape ", attrmomentum1.shape)
            print("attrmomentum1.value ", attrmomentum1)
            print("datamomentum1.shape ", datamomentum1.shape)
            print("datamomentum1.value ", datamomentum1)
            for k, e, m in zip(datakey, dataembedding, datamomentum1):
                rankdt[k] = {"embedding":e, "momentum1":m}
            tabledt[rank] = rankdt
        embedding_dt[table] = tabledt
    return embedding_dt


import numpy as np


def compare_structures(struct1, struct2):
    """
    递归比较两个结构（字典或元组）中的所有元素。
    """
    flag = True
    # 检查结构类型是否一致
    if isinstance(struct1, dict) and isinstance(struct2, dict):
        # 比较字典的键
        if sorted(struct1.keys()) != sorted(struct2.keys()):
            return False
        # 递归比较每个键对应的值
        flag = True
        for key in sorted(struct1.keys()):
            if not compare_structures(struct1[key], struct2[key]):
                print(f"key {key}")
                flag = False
        return flag
    elif isinstance(struct1, np.ndarray) and isinstance(struct2, np.ndarray):
        # 比较每个numpy数组
        if not np.array_equal(struct1, struct2):
            # print(f"struct1 {struct1}, struct2 {struct2}")
            return False
    else:
        # 其他类型直接比较
        if struct1 != struct2:
            # print(f"struct1 {struct1}, struct2 {struct2}")
            return False

    return True


def compare_embedding_dicts(embedding_dt1, embedding_dt2):
    """
    比较两个嵌入字典，使用递归函数处理不同层次的结构。
    """
    # 检查字典长度
    assert len(embedding_dt1) == len(embedding_dt2)
    print("table_num: ", len(embedding_dt1))

    # 递归比较整个结构
    if not compare_structures(embedding_dt1, embedding_dt2):
        print("Failed")
        return False

    print("All arrays match")
    print("Passed")
    return True

if __name__ == "__main__":
    path = "/home/zengxiong/20250401_torchrec/torchrec/contrib/torchrec_embcache/embcache_embedding/tests/acc_test/"
    # table1 = emb_read(os.path.join(path, "save_dir/sparse"), table_names=["table0", "table1"], ranks=[0, 1])
    table1 = emb_read("/home/zengxiong/20250401_torchrec/torchrec/contrib/torchrec_embcache/embcache_cpp/tests/sparse", table_names=["table0", "table1"], ranks=[0, 1])
    table2 = emb_read(os.path.join(path, "save_dir2/sparse"), table_names=["table0", "table1"], ranks=[0, 1])
    table3 = emb_read(os.path.join(path, "save_dir3/sparse"), table_names=["table0", "table1"], ranks=[0, 1])
    print("-----compare_embedding_dicts save_dir  save_dir2")
    assert True == compare_embedding_dicts(table1, table2)
    print("-----compare_embedding_dicts save_dir  save_dir3")
    assert True == compare_embedding_dicts(table2, table3)
