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

import os
from typing import Any, Optional

import toml

from mxrec.python.utils.validator import class_safe_check, file_safe_check
from mxrec.python.constants import CommNodeInfo, MXREC, USE_RANKTABLE


class TomlParser:
    _instance: "TomlParser" = None

    def __init__(self, path: str):
        self._path = path
        self._config = None
        self._comm_node_info = None
        self._log_level = None
        self._use_ranktable = None
        self._use_fusion_op = None
        self._use_lccl_all2all_op = None
        self._fusion_op_type = None
        self._all2all_op_type = None

        self._parse_config()

    @property
    def config(self) -> Optional[dict]:
        return self._config

    @property
    def comm_node_info(self) -> Optional[CommNodeInfo]:
        return self._comm_node_info

    @property
    def log_level(self) -> Optional[str]:
        return self._log_level

    @property
    def use_ranktable(self) -> Optional[bool]:
        return self._use_ranktable
    
    @property
    def use_fusion_op(self) -> Optional[bool]:
        return self._use_fusion_op
    
    @property
    def use_lccl_all2all_op(self) -> Optional[bool]:
        return self._use_lccl_all2all_op
    
    @property
    def fusion_op_type(self) -> Optional[bool]:
        return self._fusion_op_type
    
    @property
    def all2all_op_type(self) -> Optional[bool]:
        return self._all2all_op_type
    
    @config.setter
    def config(self, config: Optional[dict]):
        self._config = config

    @comm_node_info.setter
    def comm_node_info(self, node_info: CommNodeInfo):
        self._comm_node_info = node_info

    @log_level.setter
    def log_level(self, level: str):
        self._log_level = level

    @use_ranktable.setter
    def use_ranktable(self, use_ranktable: bool):
        self._use_ranktable = use_ranktable

    @use_fusion_op.setter
    def use_fusion_op(self, use_fusion_op: bool):
        self._use_fusion_op = use_fusion_op

    @use_lccl_all2all_op.setter
    def use_lccl_all2all_op(self, use_lccl_all2all_op: bool):
        self._use_lccl_all2all_op = use_lccl_all2all_op

    @fusion_op_type.setter
    def fusion_op_type(self, fusion_op_type: str):
        self._fusion_op_type = fusion_op_type

    @all2all_op_type.setter
    def all2all_op_type(self, all2all_op_type: str):
        self._all2all_op_type = all2all_op_type

    @classmethod
    def get_instance(cls) -> "TomlParser":
        if cls._instance is None:
            raise RuntimeError("the TomlParser instance is None, please call mxrec.init() first")

        return cls._instance

    @classmethod
    def set_instance(cls, path: str):
        if cls._instance is not None:
            raise RuntimeError("TomlParser has been initialized once, twice initialization was forbidden")

        cls._instance = TomlParser(path)

    def _parse_config(self):
        with open(self._path, "r") as f:
            file_safe_check("toml path", self._path)
            fields = toml.load(f)
            config = parse_field(fields, MXREC)
            self._config = config


def parse_field(config: dict, key: str) -> Any:
    class_safe_check("toml field", config, dict)
    field = config.get(key)
    if field is None:
        raise KeyError(f"the {key} field is missing, please check if the field exists in the toml file")
    return field


def parse_env_field(config: dict, key: str) -> Any:
    field = os.getenv(key)
    if field is not None:
        return field

    return parse_field(config, key)
