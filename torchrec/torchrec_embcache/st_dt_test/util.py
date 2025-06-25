#!/usr/bin/env python3
# Copyright (c) Huawei Platforms, Inc. and affiliates.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import logging
import pytz
import torch
import numpy as np
import os

from typing import Callable

from parse_configs import load_all_configs


OVER_COUNT = 10

TEST_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


# utils for logging
def setup_logging(rank):
    from datetime import datetime

    this_time = str(
        datetime.now(tz=pytz.timezone("PRC")).strftime(
            "%m_%d_%H_%M_%S",
        )
    )
    format = logging.Formatter(
        fmt=f"[rank{rank}][%(levelname)s][%(asctime)s.%(msecs)03d] %(message)s",
        datefmt="%m-%d %H:%M:%S",
    )
    logger = logging.getLogger()
    file_handler = logging.FileHandler(
        f"test_rank{rank}_{this_time}.log", encoding="utf-8"
    )
    file_handler.setFormatter(format)
    logger.addHandler(file_handler)
    logger.setLevel(logging.DEBUG)


# utils for config checking
def is_bad_case(config):
    return config.get("is_bad_case", False)


def is_lookup_out_of_bound(config):
    if not is_bad_case(config):
        return False
    return config.get("RecDataset", "Random") == "BoundOutOfRange"


def feature_name_exists(config):
    if not is_bad_case(config):
        return False
    return config.get("RecDataset", "Random") == "FeatureNameNotInConfig"


def check_config(config):
    # 如果是dataloader越界，需要检查loop lookup_lens
    if config["RecDataset"] == "BoundOutOfRange":
        bound_out_of_range = False
        # if LOOP_TIMES*config["lookup_lens"]
        for i in range(config["table_num"]):
            if config["lookup_lens"]*config["BATCH_NUM"] > config["num_embeddings"][i]+OVER_COUNT:
                bound_out_of_range = True
                break
        if not bound_out_of_range:
            raise ValueError("lookup_lens and BATCH_NUM is too small, if you want to test out of range, please set lookup_lens*BATCH_NUM*LOOP_TIMES > num_embeddings+OVER_COUNT")
        
    # 需要检查是否超出显存
    # 表的大小，要考虑分表的情况：sum(instances*embedding_dim*num_embeddings)/WORLD_SIZE
    table_size = 0
    for embedding_dim, num_embedding in zip(config["embedding_dims"], config["num_embeddings"]):
        table_size += embedding_dim * num_embedding
    table_size = table_size * config["table_num"] / config["WORLD_SIZE"]
    # 查表的大小，lookup_lens*embedding_dim*len(feature_names)
    lookup_size = 0
    for embedding_dim, feature_names in zip(config["embedding_dims"], config["feature_names_lst"]):
        lookup_size += embedding_dim * len(feature_names)
    total_size = (table_size + lookup_size)*4/(1024*1024)
    max_size = 65536
    if total_size > max_size:
        raise ValueError(f"table size is too large, please reduce the table size or increase the WORLD_SIZE, total_size: {total_size}, max_size: {max_size}")

    # 需要检查HBM缓存是否够用
    multi_hot_sizes = [1]*config["table_num"]
    dtype_size = 4 # default fp32
    weight_and_optim_count = 2
    # undo 准入准出会占用一个位置，lookup_lens固定+1，后续根据准入准出的参数形式进行判断
    min_mem = np.sum(np.dot(np.multiply(config["embedding_dims"], multi_hot_sizes), 2*dtype_size*config["lookup_lens"]*weight_and_optim_count)) 
    max_hbm_for_vectors = os.getenv("EMBCACHE_SIZE_ON_HBM")
    if not max_hbm_for_vectors:
        raise EnvironmentError("EMBCACHE_SIZE_ON_HBM is not set, please set it in the environment")
    max_hbm_for_vectors = int(max_hbm_for_vectors)
    if max_hbm_for_vectors < min_mem:
        raise ValueError(f"max_hbm_for_vectors is not enough, \
            please increase the EMBCACHE_SIZE_ON_HBM or reduce the embedding_dim or lookup_lens, \
                current EMBCACHE_SIZE_ON_HBM: {max_hbm_for_vectors}, min_mem: {min_mem}")


# utils for weight init
def create_weight_init(init_fn: Callable[[int], torch.Tensor]):
    """
    创建一个只接受 param 参数的 weight_init 函数。
    
    Args:
        init_fn: 一个根据输入维度返回初始化张量的函数。
        
    Returns:
        一个新的 weight_init 函数，它只接受 param 参数。
    """
    def weight_init(param: torch.nn.Parameter):
        if len(param.shape) != 2:
            return
        in_dim = param.shape[1]
        torch.manual_seed(in_dim)
        result = init_fn(in_dim).repeat(param.shape[0], 1)
        param.data.copy_(result)
    return weight_init


# 初始化器定义
def init_random(in_dim):
    return torch.randn((1, in_dim))


def init_linspace(in_dim):
    return torch.linspace(0, 1, steps=in_dim).unsqueeze(0)


def init_ones(in_dim):
    return torch.ones((1, in_dim))


def init_zeros(in_dim):
    return torch.zeros((1, in_dim))


def init_uniform(in_dim):
    return torch.empty((1, in_dim)).uniform_()


# utils for conftest
def generate_test_cases(metafunc, ALL_CONFIGS):
    """
    Pytest hook to generate tests dynamically based on the configurations.
    """
    test_case_name = metafunc.function.__name__
    configs_for_case = ALL_CONFIGS.get(test_case_name, [])

    # 获取命令行参数
    config_file = metafunc.config.getoption("--test-config-file")

    if config_file:
        # 如果指定了单个配置文件，则只使用该配置
        config_file = config_file.split(".")[0].split("/")[-1]  # Remove the file extension
        # Find the matching config for the specified file
        selected_config = next((cfg for fname, cfg in configs_for_case if fname == config_file), None)
        if selected_config is not None:
            metafunc.parametrize("config", [selected_config], ids=[config_file])

    elif configs_for_case:
        metafunc.parametrize("config", [cfg for _, cfg in configs_for_case], ids=[fname for fname, _ in configs_for_case])
    else:
        metafunc.parametrize("config", [], scope="function")


def get_all_configs(MODULE_NAME):
    CONFIG_DIR = os.path.join(TEST_ROOT_DIR, "configs", MODULE_NAME)

    if not os.path.exists(CONFIG_DIR):
        raise FileNotFoundError(f"Config directory {CONFIG_DIR} does not exist.")
    ALL_CONFIGS = load_all_configs(CONFIG_DIR)
    return ALL_CONFIGS


# utils for compare methods
def compare_tensors(tensor1, tensor2):
    if tensor1 is None and tensor2 is None:
        return True
    if tensor1 is None or tensor2 is None:
        return False
    return torch.allclose(tensor1, tensor2)


def compare_list(list1, list2):
    if list1 is None and list2 is None:
        return True
    if list1 is None or list2 is None:
        return False
    if len(list1) != len(list2):
        return False
    
    for item1, item2 in zip(list1, list2):
        if isinstance(item1, list) and isinstance(item2, list):
            if not compare_list(item1, item2):
                return False
        elif isinstance(item1, torch.Tensor) and isinstance(item2, torch.Tensor):
            if not compare_tensors(item1, item2):
                return False
        elif item1 != item2:
            return False
    return True