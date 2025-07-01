#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2024. Huawei Technologies Co.,Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import json
import os
from typing import Dict
from enum import Enum

from mxrec.python.utils import logger
from mxrec.python.constants import MPIParams, CommParams
from mxrec.python.config import get_comm_node_info, get_use_ranktable
from mxrec.python.utils.validator import (
    StringValidator,
    str_safe_check,
    file_safe_check,
    class_safe_check,
    int_safe_check,
)


class RankTableInfo(Enum):
    RANK_TABLE_FILE: str = "RANK_TABLE_FILE"
    RANK_LIST: str = "rank_list"
    RANK_ID: str = "rank_id"
    LOCAL_ID: str = "local_id"


def get_rank_id() -> int:
    """Get the rank id for the current device in the collective communication group.

    Note: this method should be used after mpi init.

    Returns:
        Int, the rank id of the calling process.
    """
    rank_id = os.getenv(MPIParams.OMPI_COMM_WORLD_RANK.value)
    if rank_id is None:
        raise EnvironmentError("get rank id failed, please init MPI first")

    return _comm_env_value_str2int(rank_id, 0)


def get_rank_size() -> int:
    """Get the rank size of the default collective communication group.

    Note: this method should be used after mpi init.

    Returns:
        Int, the rank size of the group.
    """
    rank_size = os.getenv(MPIParams.OMPI_COMM_WORLD_SIZE.value)
    if rank_size is None:
        raise EnvironmentError("get rank size failed, please init MPI first")

    return _comm_env_value_str2int(rank_size, 1)


def get_local_rank_size() -> int:
    """Get the local rank size of the default collective communication group.

    Note: this method should be used after mpi init.

    Returns:
        Int, the local rank size of the group.
    """
    local_rank_size = os.getenv(MPIParams.OMPI_COMM_WORLD_LOCAL_SIZE.value)
    if local_rank_size is None:
        raise EnvironmentError("get local rank size failed, please init MPI first")

    return _comm_env_value_str2int(local_rank_size, 1)


def get_min_device_id() -> int:
    """Get the min device logic id of the calling process.

    Note: this method should be used after mpi init.

    Returns:
        Int, the min device id of the calling process.
    """
    rank_map = _get_rank_info_with_ranktable()
    return min(rank_map)


def get_device_id() -> int:
    """Get the device logic id of the calling process.

    Note: this method should be used after mpi init.

    Returns:
        Int, the device logic id of the calling process.
    """
    if get_use_ranktable():
        rank_to_device_dict = _get_rank_info_with_ranktable()
    else:
        rank_to_device_dict = _get_rank_info_without_ranktable()
    local_rank_id = get_rank_id() % get_local_rank_size()
    device_id = rank_to_device_dict.get(local_rank_id)
    if device_id is None:
        raise KeyError(
            f"the rank id {local_rank_id} is not in the hccl info dict {rank_to_device_dict}, please init MPI first"
        )

    return device_id


def _get_rank_info_without_ranktable() -> Dict[int, int]:
    device_count = get_rank_size()
    device_list = [i for i in range(device_count)]
    comm_node_info = get_comm_node_info()
    chief_device = comm_node_info.cm_chief_device
    worker_size = comm_node_info.cm_worker_size

    if chief_device not in device_list:
        raise EnvironmentError(
            f"the environment variable CM_CHIEF_DEVICE {chief_device} is not in the local device list {device_list}"
        )
    if worker_size != device_count:
        raise EnvironmentError(
            f"the environment variable CM_WORKER_SIZE {worker_size} is not equal to the MPI worker size {device_count}"
        )

    rank_to_device_dict = {}
    chief_index = device_list.index(chief_device)
    device_list = device_list[chief_index:] + device_list[:chief_index]
    device_list = device_list[:worker_size]
    for rank_id, device_id in enumerate(device_list):
        rank_to_device_dict[rank_id] = device_id

    logger.debug("In no ranktable, the rank info dict is %s.", rank_to_device_dict)
    return rank_to_device_dict


def _get_rank_info_with_ranktable() -> Dict[int, int]:
    rank_table_path = os.getenv(RankTableInfo.RANK_TABLE_FILE.value, "")
    with open(rank_table_path, "r", encoding="utf-8") as file:
        file_safe_check(RankTableInfo.RANK_TABLE_FILE.value, rank_table_path)

        try:
            ranktable_info = json.load(file)
        except FileNotFoundError as e:
            raise ValueError("ranktable file not found, please export RANK_TABLE_FILE first") from e
        except json.JSONDecodeError as e:
            raise ValueError("ranktable file is unable to parse as json") from e
        class_safe_check("ranktable_info", ranktable_info, (dict,))

        if RankTableInfo.RANK_LIST.value not in ranktable_info:
            raise AttributeError("lack of attribute rank_list")
        rank_list = ranktable_info.get(RankTableInfo.RANK_LIST.value)
        if not rank_list:
            raise ValueError("rank_list is empty")
        class_safe_check("rank_list", rank_list, (list,))

    rank_to_device_dict = dict()
    for infos in rank_list:
        if RankTableInfo.LOCAL_ID.value not in infos:
            raise AttributeError("lack of attribute local_id")
        local_id = infos.get(RankTableInfo.LOCAL_ID.value)
        int_safe_check("local_id", local_id, min_value=0, max_value=CommParams.MAX_LOCAL_ID.value)
        if RankTableInfo.RANK_ID.value not in infos:
            raise AttributeError("lack of attribute rank_id")
        rank_id = infos.get(RankTableInfo.RANK_ID.value)
        int_safe_check("rank_id", rank_id, min_value=0, max_value=CommParams.MAX_RANK_ID.value)

        rank_to_device_dict[rank_id] = local_id

    logger.debug("In ranktable, the rank info dict is %s.", rank_to_device_dict)
    return rank_to_device_dict


def _comm_env_value_str2int(value: str, greater_or_equal: int = 0) -> int:
    str_safe_check("communication environment value", value)

    if not StringValidator("communication environment value", value).can_be_transformed2int().is_valid():
        raise ValueError(f"the value should be number, but got the value: {value}")

    int_value = int(value)
    if int_value < greater_or_equal:
        raise ValueError(f"the value must be greater than or equal to {greater_or_equal}, but got {int_value}")
    return int_value
