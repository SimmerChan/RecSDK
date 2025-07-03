import os
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import pytest
from hybrid_torchrec.distributed.sharding.hybrid_rw_sharding import (ThreadPoolExecutorSingleton,
                                                                     DEFAULT_INPUT_DIST_THREADS,
                                                                     MAX_INPUT_DIST_THREADS)


class TestThreadPoolExecutorSingleton:
    def test_singleton_instance(self):
        """测试单例模式是否正常工作"""
        instance1 = ThreadPoolExecutorSingleton()
        instance2 = ThreadPoolExecutorSingleton()
        assert instance1 is instance2
        assert isinstance(instance1.executor, ThreadPoolExecutor)

    @patch.dict(os.environ, {"INPUT_DIST_THREADS": "4"})
    def test_with_valid_env_threads(self):
        """测试有效的环境变量设置"""
        instance = ThreadPoolExecutorSingleton()
        assert instance.executor._max_workers == 4

    def test_with_default_threads(self):
        """测试默认线程数设置"""
        instance = ThreadPoolExecutorSingleton()
        assert instance.executor._max_workers == DEFAULT_INPUT_DIST_THREADS

    @patch.dict(os.environ, {"INPUT_DIST_THREADS": "0"})
    def test_with_too_small_threads(self):
        """测试线程数过小的情况"""
        with pytest.raises(ValueError) as excinfo:
            ThreadPoolExecutorSingleton()
        assert f"expected in range [1, {MAX_INPUT_DIST_THREADS}]" in str(excinfo.value)

    @patch.dict(os.environ, {"INPUT_DIST_THREADS": str(MAX_INPUT_DIST_THREADS + 1)})
    def test_with_too_large_threads(self):
        """测试线程数过大的情况"""
        with pytest.raises(ValueError) as excinfo:
            ThreadPoolExecutorSingleton()
        assert f"expected in range [1, {MAX_INPUT_DIST_THREADS}]" in str(excinfo.value)

    @patch.dict(os.environ, {"INPUT_DIST_THREADS": "invalid"})
    def test_with_invalid_threads(self):
        """测试非数字环境变量"""
        with pytest.raises(Exception) as excinfo:
            ThreadPoolExecutorSingleton()
        assert "not a valid integer" in str(excinfo.value)
