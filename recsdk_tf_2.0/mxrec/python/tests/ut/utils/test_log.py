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

from mxrec.python.utils import logger
from mxrec.python.constants import LogParams


class TestLoggingProxy:
    """Test for 'mxrec.python.utils.log.LoggingProxy'."""

    @staticmethod
    def teardown_method():
        logger._instance = None

    @staticmethod
    def test_init_logger_ok():
        try:
            logger.set_instance(LogParams.INFO.value)
        except Exception as e:
            pytest.fail(f"unexpected exception raised: {e}")

    @staticmethod
    def test_twice_init_err():
        with pytest.raises(RuntimeError) as excinfo:
            logger.set_instance(LogParams.INFO.value)
            logger.set_instance(LogParams.INFO.value)
        assert "LoggingProxy has been initialized once, twice initialization was forbidden" in str(excinfo.value)

    @staticmethod
    def test_info_log_ok():
        logger.set_instance(LogParams.INFO.value)
        try:
            logger.info("Test log, we have a %s, log level is %s.", "interesting problem", "info")
        except Exception as e:
            pytest.fail(f"unexpected exception raised: {e}")

    @staticmethod
    def test_debug_log_ok():
        logger.set_instance(LogParams.DEBUG.value)
        try:
            logger.debug("Test log, we have a %s, log level is %s.", "thorny problem", "debug")
        except Exception as e:
            pytest.fail(f"unexpected exception raised: {e}")

    @staticmethod
    def test_warning_log_ok():
        logger.set_instance(LogParams.WARNING.value)
        try:
            logger.warning("Test log, we have a %s, log level is %s.", "bit of a problem", "warning")
        except Exception as e:
            pytest.fail(f"unexpected exception raised: {e}")

    @staticmethod
    def test_error_log_ok():
        logger.set_instance(LogParams.ERROR.value)
        try:
            logger.error("Test log, we have a %s, log level is %s.", "major problem", "error")
        except Exception as e:
            pytest.fail(f"unexpected exception raised: {e}")

    @staticmethod
    def test_exception_log_ok():
        logger.set_instance(LogParams.ERROR.value)
        try:
            logger.exception("Test log, we have a %s, log level is %s.", "exception problem", "exception")
        except Exception as e:
            pytest.fail(f"unexpected exception raised: {e}")

    @staticmethod
    def test_critical_log_ok():
        logger.set_instance(LogParams.CRITICAL.value)
        try:
            logger.critical("Test log, we have a %s, log level is %s.", "major disaster", "critical")
        except Exception as e:
            pytest.fail(f"unexpected exception raised: {e}")
