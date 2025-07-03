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

import pytest
import toml

import mxrec
from mxrec.python.utils import logger
from mxrec.python.config import TomlParser
from mxrec.python.config.comm_parser import parse_comm_info
from mxrec.python.constants import CommNodeInfo


class TestParseCommInfo:
    """Test for 'from mxrec.python.config.log_parser.parse_comm_info'."""

    @staticmethod
    def setup_method():
        mxrec.init("./ut_test.toml")

    @staticmethod
    def teardown_method():
        TomlParser._instance = None
        logger._instance = None

    @staticmethod
    def test_ok():
        info = parse_comm_info()
        assert isinstance(info, CommNodeInfo)

    @staticmethod
    def test_chief_ip_ipv4_err(monkeypatch):
        mock_toml = """
                    [mxrec]
                    [mxrec.cm-node-info]
                    cm_chief_ip = "256.0.0.1"
                    cm_chief_port = 60001
                    cm_chief_device = 0
                    cm_worker_ip = "127.0.0.1"
                    cm_worker_size = 1
                    """
        config = toml.loads(mock_toml).get("mxrec")
        monkeypatch.setattr("mxrec.python.config.parser.TomlParser.config", config)

        with pytest.raises(ValueError) as excinfo:
            parse_comm_info()
        assert "is not a valid IPv4 address" in str(excinfo.value)

    @staticmethod
    def test_chief_port_value_err(monkeypatch):
        mock_toml = """
                    [mxrec]
                    [mxrec.cm-node-info]
                    cm_chief_ip = "127.0.0.1"
                    cm_chief_port = 66666
                    cm_chief_device = 0
                    cm_worker_ip = "127.0.0.1"
                    cm_worker_size = 1
                    """
        config = toml.loads(mock_toml).get("mxrec")
        monkeypatch.setattr("mxrec.python.config.parser.TomlParser.config", config)

        with pytest.raises(ValueError) as excinfo:
            parse_comm_info()
        assert "is bigger than" in str(excinfo.value)

    @staticmethod
    def test_chief_device_value_err(monkeypatch):
        mock_toml = """
                    [mxrec]
                    [mxrec.cm-node-info]
                    cm_chief_ip = "127.0.0.1"
                    cm_chief_port = 60001
                    cm_chief_device = -1
                    cm_worker_ip = "127.0.0.1"
                    cm_worker_size = 1
                    """
        config = toml.loads(mock_toml).get("mxrec")
        monkeypatch.setattr("mxrec.python.config.parser.TomlParser.config", config)

        with pytest.raises(ValueError) as excinfo:
            parse_comm_info()
        assert "is less than" in str(excinfo.value)

    @staticmethod
    def test_woker_ip_ipv4_err(monkeypatch):
        mock_toml = """
                    [mxrec]
                    [mxrec.cm-node-info]
                    cm_chief_ip = "127.0.0.1"
                    cm_chief_port = 60001
                    cm_chief_device = 0
                    cm_worker_ip = "127.0.0.256"
                    cm_worker_size = 1
                    """
        config = toml.loads(mock_toml).get("mxrec")
        monkeypatch.setattr("mxrec.python.config.parser.TomlParser.config", config)

        with pytest.raises(ValueError) as excinfo:
            parse_comm_info()
        assert "is not a valid IPv4 address" in str(excinfo.value)

    @staticmethod
    def test_worker_size_value_err(monkeypatch):
        mock_toml = """
                    [mxrec]
                    [mxrec.cm-node-info]
                    cm_chief_ip = "127.0.0.1"
                    cm_chief_port = 60001
                    cm_chief_device = 0
                    cm_worker_ip = "127.0.0.1"
                    cm_worker_size = -1
                    """
        config = toml.loads(mock_toml).get("mxrec")
        monkeypatch.setattr("mxrec.python.config.parser.TomlParser.config", config)

        with pytest.raises(ValueError) as excinfo:
            parse_comm_info()
        assert "is less than" in str(excinfo.value)
