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

from mxrec.python.constants import CommNodeInfo, USE_RANKTABLE, USE_FUSION_OP, \
                                    USE_LCCL_ALL2ALL_OP, FUSION_OP_TYPE, ALL2ALL_OP_TYPE
from mxrec.python.config.parser import TomlParser, parse_env_field
from mxrec.python.config.comm_parser import parse_comm_info
from mxrec.python.config.log_parser import parse_log_level
from mxrec.python.utils.validator import class_safe_check


def get_log_level() -> str:
    if TomlParser.get_instance().log_level is None:
        level = parse_log_level()
        TomlParser.get_instance().log_level = level

    return TomlParser.get_instance().log_level


def get_comm_node_info() -> CommNodeInfo:
    if TomlParser.get_instance().comm_node_info is None:
        info = parse_comm_info()
        TomlParser.get_instance().comm_node_info = info

    return TomlParser.get_instance().comm_node_info


def get_use_fusion_op() -> bool:
    if TomlParser.get_instance().use_fusion_op is None:
        use_fusion_op = _parse_use_fusion_op()
        TomlParser.get_instance().use_fusion_op = use_fusion_op
    return TomlParser.get_instance().use_fusion_op


def get_use_lccl_all2all_op() -> bool:
    if TomlParser.get_instance().use_lccl_all2all_op is None:
        use_lccl_all2all_op = _parse_use_lccl_all2all_op()
        TomlParser.get_instance().use_lccl_all2all_op = use_lccl_all2all_op
    return TomlParser.get_instance().use_lccl_all2all_op


def get_fusion_op_type() -> str:
    if TomlParser.get_instance().fusion_op_type is None:
        fusion_op_type = _parse_fusion_op_type()
        TomlParser.get_instance().fusion_op_type = fusion_op_type
    return TomlParser.get_instance().fusion_op_type


def get_all2all_op_type() -> str:
    if TomlParser.get_instance().all2all_op_type is None:
        all2all_op_type = _parse_all2all_op_type()
        TomlParser.get_instance().all2all_op_type = all2all_op_type
    return TomlParser.get_instance().all2all_op_type


def _parse_use_fusion_op() -> bool:
    config = TomlParser.get_instance().config
    use_fusion_op = parse_env_field(config, USE_FUSION_OP)
    class_safe_check("use_fusion_op", use_fusion_op, (bool,))
    return use_fusion_op


def _parse_use_lccl_all2all_op() -> bool:
    config = TomlParser.get_instance().config
    use_lccl_all2all_op = parse_env_field(config, USE_LCCL_ALL2ALL_OP)
    class_safe_check("use_lccl_all2all_op", use_lccl_all2all_op, (bool,))
    return use_lccl_all2all_op


def _parse_fusion_op_type() -> str:
    config = TomlParser.get_instance().config
    fusion_op_type = parse_env_field(config, FUSION_OP_TYPE)
    class_safe_check("fusion_op_type", fusion_op_type, (str,))
    return fusion_op_type


def _parse_all2all_op_type() -> str:
    config = TomlParser.get_instance().config
    all2all_op_type = parse_env_field(config, ALL2ALL_OP_TYPE)
    class_safe_check("all2all_op_type", all2all_op_type, (str,))
    return all2all_op_type


def get_use_ranktable() -> bool:
    if TomlParser.get_instance().use_ranktable is None:
        use_ranktable = _parse_use_ranktable()
        TomlParser.get_instance().use_ranktable = use_ranktable

    return TomlParser.get_instance().use_ranktable


def _parse_use_ranktable() -> bool:
    config = TomlParser.get_instance().config
    use_ranktable = parse_env_field(config, USE_RANKTABLE)
    class_safe_check("use_ranktable", use_ranktable, (bool,))
    if not use_ranktable:
        raise ValueError(
            "currently, the use_ranktable only supports True, please check if the toml file is configured correctly"
        )

    return use_ranktable
