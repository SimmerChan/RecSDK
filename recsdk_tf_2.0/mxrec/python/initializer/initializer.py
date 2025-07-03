#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2025. Huawei Technologies Co.,Ltd. All rights reserved.
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

import os
from dataclasses import fields

from mxrec.python.constants import CommNodeInfo
from mxrec.python.config import TomlParser, get_comm_node_info, get_log_level, get_use_ranktable
from mxrec.python.utils import logger
from mxrec.python.communication import get_device_id, get_rank_id, get_rank_size


_MAX_TOML_PATH_LEN = 1024


def init(toml_path: str):
    if len(toml_path) < 1 or len(toml_path) > _MAX_TOML_PATH_LEN:
        raise ValueError(f"the length of toml path must be between [1, {_MAX_TOML_PATH_LEN}]")

    # Toml parser init.
    _parser_init(toml_path)

    # The logger init must after the parser init.
    _logger_init()

    # Environment variables init.
    _ascend_env_init()

    logger.info("MxRec initialization is complete.")


def _parser_init(path: str):
    TomlParser.set_instance(path)


def _logger_init():
    level = get_log_level()
    logger.set_instance(level)


def _ascend_env_init():
    if get_use_ranktable():
        rank_id = get_rank_id()
        os.environ["RANK_ID"] = str(rank_id)
        logger.info("The environment variable RANK_ID is set to %s.", rank_id)
        rank_size = get_rank_size()
        os.environ["RANK_SIZE"] = str(rank_size)
        logger.info("The environment variable RANK_SIZE is set to %s.", rank_size)
    else:
        env_info = []
        comm_node_info = get_comm_node_info()
        env_info.append(comm_node_info)

        for info in env_info:
            if not isinstance(info, (CommNodeInfo,)):
                raise ValueError(f"the environment info must be dataclass, but got {info}")

            for field in fields(info):
                env_name = field.name.upper()
                ori_env_var = os.getenv(env_name)
                if ori_env_var is not None:
                    continue

                env_var = getattr(info, field.name)
                os.environ[env_name] = str(env_var)
                logger.info("The environment variable %s is set to %s.", env_name, env_var)

    # Set ascend device id.
    device_id = str(get_device_id())
    os.environ["ASCEND_DEVICE_ID"] = device_id
    logger.info("The environment variable ASCEND_DEVICE_ID is set to %s.", device_id)
