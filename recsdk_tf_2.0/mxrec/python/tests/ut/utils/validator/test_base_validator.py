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
import shutil
import pytest

from mxrec.python.utils.validator import StringValidator, IntValidator, ClassValidator, DirectoryValidator


class TestStringValidator:
    """Test for 'mxrec.python.utils.validator.base_validator.StringValidator'."""

    @staticmethod
    def test_type_ok():
        validator = StringValidator("test_value", "xxx")
        assert validator.is_valid()

    @staticmethod
    def test_type_err():
        validator = StringValidator("test_value", 1)
        with pytest.raises(ValueError) as excinfo:
            validator.check()
        assert "is not str" in str(excinfo.value)

    @staticmethod
    def test_len_less_ok():
        validator = StringValidator("test_value", "xxx", min_len=3)
        assert validator.check_string_length().is_valid()

    @staticmethod
    def test_len_less_err():
        validator = StringValidator("test_value", "xxx", min_len=4)
        with pytest.raises(ValueError) as excinfo:
            validator.check_string_length().check()
        assert "test_value length not in" in str(excinfo.value)

    @staticmethod
    def test_len_greater_ok():
        validator = StringValidator("test_value", "xxx", max_len=3)
        assert validator.check_string_length().is_valid()

    @staticmethod
    def test_len_greater_err():
        validator = StringValidator("test_value", "xxx", max_len=2)
        with pytest.raises(ValueError) as excinfo:
            validator.check_string_length().check()
        assert "test_value length not in" in str(excinfo.value)

    @staticmethod
    def test_contain_black_element_ok():
        validator = StringValidator("test_value", "xxx", element="k")
        assert validator.check_not_contain_black_element().is_valid()

    @staticmethod
    def test_contain_black_element_err():
        validator = StringValidator("test_value", "xxx", element="x")
        with pytest.raises(ValueError) as excinfo:
            validator.check_not_contain_black_element().check()
        assert "contain black element" in str(excinfo.value)

    @staticmethod
    def test_whitelist_ok():
        validator = StringValidator("test_value", "xxx_XXX")
        assert validator.check_whitelist().is_valid()

    @staticmethod
    def test_whitelist_err():
        validator = StringValidator("test_value", r"ab*\nc")
        with pytest.raises(ValueError) as excinfo:
            validator.check_whitelist().check()
        assert "Note: It should be a string consisting of '[0-9A-Za-z_.-]'" in str(excinfo.value)

    @staticmethod
    def test_can_be_to_int_ok():
        validator = StringValidator("test_value", "1")
        assert validator.can_be_transformed2int().is_valid()

    @staticmethod
    def test_can_be_to_int_err():
        validator = StringValidator("test_value", "xx")
        assert not validator.can_be_transformed2int().is_valid()

    @staticmethod
    def test_can_be_to_float_ok():
        validator = StringValidator("test_value", "1.1")
        assert validator.can_be_transformed2float().is_valid()

    @staticmethod
    def test_can_be_to_float_err():
        validator = StringValidator("test_value", "xx")
        assert not validator.can_be_transformed2float().is_valid()


class TestIntValidator:
    """Test for 'mxrec.python.utils.validator.base_validator.IntValidator'."""

    @staticmethod
    def test_type_ok():
        validator = IntValidator("test_value", 1)
        assert validator.is_valid()

    @staticmethod
    def test_type_err():
        validator = IntValidator("test_value", "1")
        with pytest.raises(ValueError) as excinfo:
            validator.check()
        assert "is not int" in str(excinfo.value)

    @staticmethod
    def test_special_type_bool():
        validator = IntValidator("test_value", True)
        with pytest.raises(ValueError) as excinfo:
            validator.check()
        assert "is not int" in str(excinfo.value)

    @staticmethod
    def test_max_value_ok():
        validator = IntValidator("test_value", 10, max_value=10)
        assert validator.check_value().is_valid()

    @staticmethod
    def test_max_value_err():
        validator = IntValidator("test_value", 10, max_value=9)
        with pytest.raises(ValueError) as excinfo:
            validator.check_value().check()
        assert "is bigger than" in str(excinfo.value)

    @staticmethod
    def test_min_value_ok():
        validator = IntValidator("test_value", 10, min_value=10)
        assert validator.check_value().is_valid()

    @staticmethod
    def test_min_value_err():
        validator = IntValidator("test_value", 10, min_value=11)
        with pytest.raises(ValueError) as excinfo:
            validator.check_value().check()
        assert "is less than" in str(excinfo.value)


class TestClassValidator:
    """Test for 'mxrec.python.utils.validator.base_validator.ClassValidator'."""

    @staticmethod
    def test_type_ok():
        validator = ClassValidator("test_value", 1, int)
        assert validator.is_valid()

    @staticmethod
    def test_type_err():
        validator = ClassValidator("test_value", 1, str)
        with pytest.raises(ValueError) as excinfo:
            validator.check()
        assert "is not <class 'str'>" in str(excinfo.value)


class TestDirectoryValidator:
    """Test for 'mxrec.python.utils.validator.base_validator.DirectoryValidator'."""

    @staticmethod
    def test_check_is_not_none_ok():
        validator = DirectoryValidator("name", "test_path")
        assert validator.check_is_not_none().check()

    @staticmethod
    def test_check_is_not_none_err():
        validator = DirectoryValidator("name", None)
        with pytest.raises(ValueError) as excinfo:
            validator.check_is_not_none().check()
        assert "is not str" in str(excinfo.value)

    @staticmethod
    def test_check_not_soft_link_ok():
        path = "test_path"
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

        validator = DirectoryValidator("path", path)
        assert validator.check_not_soft_link().check()
        shutil.rmtree(path)

    @staticmethod
    def test_check_not_soft_link_err():
        path = "test_path"
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

        sym_path = "test_sym_path"
        os.symlink(path, sym_path)
        validator = DirectoryValidator("path", sym_path)
        with pytest.raises(ValueError) as excinfo:
            validator.check_not_soft_link().check()
        assert f"path: {sym_path} is soft link" in str(excinfo.value)

        os.remove(sym_path)
        shutil.rmtree(path)

    @staticmethod
    def test_path_should_exist_ok():
        path = "test_path"
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

        validator = DirectoryValidator("path", path)
        assert validator.path_should_exist().check()
        shutil.rmtree(path)

    @staticmethod
    def test_path_should_exist_err():
        path = "test_path"
        validator = DirectoryValidator("path", path)
        with pytest.raises(ValueError) as excinfo:
            validator.path_should_exist().check()
        assert f"path: {path} does not exist" in str(excinfo.value)

    @staticmethod
    def test_should_not_contains_sensitive_words_ok():
        path = "test_path"
        validator = DirectoryValidator("path", path)
        assert validator.should_not_contains_sensitive_words().check()

    @staticmethod
    def test_should_not_contains_sensitive_words_err():
        path = "password"
        validator = DirectoryValidator("path", path)
        with pytest.raises(ValueError) as execinfo:
            validator.should_not_contains_sensitive_words().check()
        assert f"path: {path} contains sensitive words" in str(execinfo.value)

    @staticmethod
    def test_with_blacklist_exact_compare_is_true_ok():
        path = "test_path"
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

        validator = DirectoryValidator("path", path)
        assert validator.with_blacklist().check()
        shutil.rmtree(path)

    @staticmethod
    def test_with_blacklist_exact_compare_is_false_ok():
        path = "test_path"
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

        validator = DirectoryValidator("path", path)
        assert validator.with_blacklist(exact_compare=False).check()
        shutil.rmtree(path)
