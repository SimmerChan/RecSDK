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

import pytest
import toml

import mxrec
from mxrec.python.utils import logger
from mxrec.python.config import TomlParser
from mxrec.python.config.log_parser import parse_log_level


class TestParseLogLevel:
    """Test for 'from mxrec.python.config.log_parser.parse_log_level'."""

    @staticmethod
    def setup_method():
        mxrec.init("./ut_test.toml")

    @staticmethod
    def teardown_method():
        TomlParser._instance = None
        logger._instance = None

    @staticmethod
    def test_ok():
        level = parse_log_level()
        # Default log level is INFO.
        assert level == "INFO"

    @staticmethod
    def test_level_not_exist_err(monkeypatch):
        mock_toml = """
                    [mxrec]
                    log_level = "XXX"
                    """
        config = toml.loads(mock_toml).get("mxrec")
        monkeypatch.setattr("mxrec.python.config.parser.TomlParser.config", config)

        with pytest.raises(ValueError) as excinfo:
            parse_log_level()
        assert "log level is invalid, only" in str(excinfo.value)
