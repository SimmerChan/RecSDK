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

import mxrec
from mxrec.python.utils import logger
from mxrec.python.communication import get_rank_id, get_rank_size, get_local_rank_size, get_device_id
from mxrec.python.constants import MPIParams, CommNodeInfo
from mxrec.python.config import TomlParser


class TestGetRankId:
    """Test for 'from mxrec.python.communication.rank_info.get_rank_id'."""

    @staticmethod
    def test_ok():
        assert get_rank_id() >= 0

    @staticmethod
    def test_env_none_err(monkeypatch):
        monkeypatch.delenv(MPIParams.OMPI_COMM_WORLD_RANK.value)
        with pytest.raises(EnvironmentError):
            get_rank_id()

    @staticmethod
    def test_safe_check_type_err(monkeypatch):
        def _mock_os_getenv(*args, **kwargs):
            return 1

        monkeypatch.setattr("mxrec.python.communication.rank_info.os.getenv", _mock_os_getenv)
        with pytest.raises(ValueError) as excinfo:
            get_rank_id()
        assert "is not str" in str(excinfo.value)

    @staticmethod
    def test_safe_check_whitelist_err(monkeypatch):
        monkeypatch.setenv(MPIParams.OMPI_COMM_WORLD_RANK.value, r"\n")
        with pytest.raises(ValueError) as excinfo:
            get_rank_id()
        assert "It should be a string consisting of '[0-9A-Za-z_.-]'" in str(excinfo.value)

    @staticmethod
    def test_str2int_err(monkeypatch):
        monkeypatch.setenv(MPIParams.OMPI_COMM_WORLD_RANK.value, "ab")
        with pytest.raises(ValueError) as excinfo:
            get_rank_id()
        assert "the value should be number" in str(excinfo.value)

    @staticmethod
    def test_negative_num_err(monkeypatch):
        monkeypatch.setenv(MPIParams.OMPI_COMM_WORLD_RANK.value, "-1")
        with pytest.raises(ValueError) as excinfo:
            get_rank_id()
        assert "the value must be greater than or equal to 0" in str(excinfo.value)


class TestGetRankSize:
    """Test for 'from mxrec.python.communication.rank_info.get_rank_size'."""

    @staticmethod
    def test_ok():
        assert get_rank_size() >= 1

    @staticmethod
    def test_env_none_err(monkeypatch):
        monkeypatch.delenv(MPIParams.OMPI_COMM_WORLD_SIZE.value)
        with pytest.raises(EnvironmentError):
            get_rank_size()

    @staticmethod
    def test_safe_check_type_err(monkeypatch):
        def _mock_os_getenv(*args, **kwargs):
            return 1

        monkeypatch.setattr("mxrec.python.communication.rank_info.os.getenv", _mock_os_getenv)
        with pytest.raises(ValueError) as excinfo:
            get_rank_size()
        assert "is not str" in str(excinfo.value)

    @staticmethod
    def test_safe_check_whitelist_err(monkeypatch):
        monkeypatch.setenv(MPIParams.OMPI_COMM_WORLD_SIZE.value, r"\n")
        with pytest.raises(ValueError) as excinfo:
            get_rank_size()
        assert "It should be a string consisting of '[0-9A-Za-z_.-]'" in str(excinfo.value)

    @staticmethod
    def test_str2int_err(monkeypatch):
        monkeypatch.setenv(MPIParams.OMPI_COMM_WORLD_SIZE.value, "ab")
        with pytest.raises(ValueError) as excinfo:
            get_rank_size()
        assert "the value should be number" in str(excinfo.value)

    @staticmethod
    def test_zero_num_err(monkeypatch):
        monkeypatch.setenv(MPIParams.OMPI_COMM_WORLD_SIZE.value, "0")
        with pytest.raises(ValueError) as excinfo:
            get_rank_size()
        assert "the value must be greater than or equal to 1" in str(excinfo.value)


class TestGetLocalRankSize:
    """Test for 'from mxrec.python.communication.rank_info.get_local_rank_size'."""

    @staticmethod
    def test_ok():
        assert get_local_rank_size() >= 1

    @staticmethod
    def test_env_none_err(monkeypatch):
        monkeypatch.delenv(MPIParams.OMPI_COMM_WORLD_LOCAL_SIZE.value)
        with pytest.raises(EnvironmentError):
            get_local_rank_size()

    @staticmethod
    def test_safe_check_type_err(monkeypatch):
        def _mock_os_getenv(*args, **kwargs):
            return 1

        monkeypatch.setattr("mxrec.python.communication.rank_info.os.getenv", _mock_os_getenv)
        with pytest.raises(ValueError) as excinfo:
            get_local_rank_size()
        assert "is not str" in str(excinfo.value)

    @staticmethod
    def test_safe_check_whitelist_err(monkeypatch):
        monkeypatch.setenv(MPIParams.OMPI_COMM_WORLD_LOCAL_SIZE.value, r"\n")
        with pytest.raises(ValueError) as excinfo:
            get_local_rank_size()
        assert "It should be a string consisting of '[0-9A-Za-z_.-]'" in str(excinfo.value)

    @staticmethod
    def test_str2int_err(monkeypatch):
        monkeypatch.setenv(MPIParams.OMPI_COMM_WORLD_LOCAL_SIZE.value, "ab")
        with pytest.raises(ValueError) as excinfo:
            get_local_rank_size()
        assert "the value should be number" in str(excinfo.value)

    @staticmethod
    def test_zero_num_err(monkeypatch):
        monkeypatch.setenv(MPIParams.OMPI_COMM_WORLD_LOCAL_SIZE.value, "0")
        with pytest.raises(ValueError) as excinfo:
            get_local_rank_size()
        assert "the value must be greater than or equal to 1" in str(excinfo.value)


class TestGetDeviceId:
    """Test for 'from mxrec.python.communication.rank_info.get_device_id'."""

    @staticmethod
    def setup_method():
        mxrec.init("./ut_test.toml")

    @staticmethod
    def teardown_method():
        TomlParser._instance = None
        logger._instance = None

    @staticmethod
    def test_ok_without_ranktable(monkeypatch):
        monkeypatch.setattr("mxrec.python.config.parser.TomlParser.use_ranktable", False)
        assert get_device_id() >= 0

    @staticmethod
    def test_ok_with_ranktable(monkeypatch):
        monkeypatch.setattr("mxrec.python.config.parser.TomlParser.use_ranktable", True)
        assert get_device_id() >= 0

    @staticmethod
    def test_chief_device_not_exist_err(monkeypatch):
        mock_comm_node_info = CommNodeInfo(
            cm_chief_ip="127.0.0.1",
            cm_chief_port=60001,
            cm_chief_device=88,
            cm_worker_ip="127.0.0.1",
            cm_worker_size=1,
        )
        monkeypatch.setattr("mxrec.python.config.parser.TomlParser.use_ranktable", False)
        monkeypatch.setattr("mxrec.python.config.parser.TomlParser.comm_node_info", mock_comm_node_info)

        with pytest.raises(EnvironmentError) as excinfo:
            get_device_id()
        assert "the environment variable CM_CHIEF_DEVICE" in str(excinfo.value)

    @staticmethod
    def test_worker_size_not_equal_err(monkeypatch):
        mock_comm_node_info = CommNodeInfo(
            cm_chief_ip="127.0.0.1",
            cm_chief_port=60001,
            cm_chief_device=0,
            cm_worker_ip="127.0.0.1",
            cm_worker_size=88,
        )
        monkeypatch.setattr("mxrec.python.config.parser.TomlParser.use_ranktable", False)
        monkeypatch.setattr("mxrec.python.config.parser.TomlParser.comm_node_info", mock_comm_node_info)

        with pytest.raises(EnvironmentError) as excinfo:
            get_device_id()
        assert "the environment variable CM_WORKER_SIZE" in str(excinfo.value)

    @staticmethod
    def test_device_id_none_err(monkeypatch):
        def _mock_get_rank_id():
            return mock_rank_id

        def _mock_get_local_rank_size():
            return 8

        mock_rank_id = 3
        monkeypatch.setattr("mxrec.python.communication.rank_info.get_rank_id", _mock_get_rank_id)
        monkeypatch.setattr("mxrec.python.communication.rank_info.get_local_rank_size", _mock_get_local_rank_size)

        mock_comm_node_info = CommNodeInfo(
            cm_chief_ip="127.0.0.1",
            cm_chief_port=60001,
            cm_chief_device=0,
            cm_worker_ip="127.0.0.1",
            cm_worker_size=1,
        )
        monkeypatch.setattr("mxrec.python.config.parser.TomlParser.use_ranktable", False)
        monkeypatch.setattr("mxrec.python.config.parser.TomlParser.comm_node_info", mock_comm_node_info)

        with pytest.raises(KeyError) as excinfo:
            get_device_id()
        assert f"the rank id {mock_rank_id} is not in the hccl info dict" in str(excinfo.value)
