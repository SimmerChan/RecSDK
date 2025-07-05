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
import re
import stat
from typing import Optional, Callable, Any, Union, Tuple, List

import tensorflow as tf

from mxrec.python.constants import ValidatorParams


class BaseValidator:
    """A base validator to check the input parameters."""

    def __init__(self, name: str, value: Any, msg: str = "value is invalid"):
        self._name = name
        self._value = value
        self._msg = msg
        self._checkers = []
        self._is_valid_state = None

    def check(self):
        if self._is_valid_state is None:
            self._is_valid_state = True
        for checker, msg in self._checkers:
            if not checker():
                self._msg = msg
                raise ValueError(self._msg)
        if self._is_valid_state:
            self._msg = None
        return self

    def is_valid(self):
        if self._is_valid_state is None:
            self.check()
        return self._is_valid_state

    def _register_checker(self, checker: Callable[[], bool], msg: str = ""):
        self._checkers.append((checker, msg if msg else self._msg))


class StringValidator(BaseValidator):
    """String type validator."""

    def __init__(
        self,
        name: str,
        value: str,
        min_len: int = ValidatorParams.STR_MIN_LENGTH.value,
        max_len: int = ValidatorParams.STR_MAX_LENGTH.value,
        element: Optional[str] = None,
        msg: str = "",
    ):
        super(StringValidator, self).__init__(name, value)
        self._min_len = min_len
        self._max_len = max_len
        self._element = element
        self._whitelist = "^[0-9A-Za-z_.-]+$"

        msg = msg if msg else f"type of '{name}' is not str, '{value}' is '{type(value)}'"
        self._register_checker(lambda: isinstance(value, str), msg)

    def check_string_length(self):
        self._register_checker(
            lambda: (len(self._value) >= self._min_len) and (len(self._value) <= self._max_len),
            f"{self._name} length not in [{self._min_len}, {self._max_len}]"
        )
        return self

    def check_not_contain_black_element(self):
        if self._value is not None and self._element is not None and self._element != "":
            self._register_checker(
                lambda: self._value.find(self._element) == -1, f"'{self._name}' contain black element '{self._element}'"
            )
        return self

    def check_whitelist(self):
        self._register_checker(
            lambda: self._value is not None and re.match(self._whitelist, self._value) is not None,
            f"the string '{self._name}' is invalid, please check the input string. "
            "Note: It should be a string consisting of '[0-9A-Za-z_.-]'",
        )
        return self

    def can_be_transformed2int(self):
        can_transformed = True
        try:
            int(self._value)
        except (ValueError, TypeError):
            can_transformed = False
        finally:
            if self._is_valid_state is not None:
                self._is_valid_state &= can_transformed
            else:
                self._is_valid_state = can_transformed
        return self

    def can_be_transformed2float(self):
        can_transformed = True
        try:
            float(self._value)
        except (ValueError, TypeError):
            can_transformed = False
        finally:
            if self._is_valid_state is not None:
                self._is_valid_state &= can_transformed
            else:
                self._is_valid_state = can_transformed
        return self


class NumValidator(BaseValidator):
    """Number validator for float or int."""

    def __init__(
        self,
        name: str,
        value: Union[int, float],
        min_value: Union[int, float] = None,
        max_value: Union[int, float] = None,
    ):
        super(NumValidator, self).__init__(name, value)

        self._min_value = min_value
        self._max_value = max_value

    def check_value(self):
        self._register_checker(lambda: self._value >= self._min_value, f"'{self._name}' is less than {self._min_value}")
        self._register_checker(
            lambda: self._value <= self._max_value, f"'{self._name}' is bigger than {self._max_value}"
        )
        return self

    def check_value_for_open_interval(self):
        if self._min_value is not None:
            self._register_checker(
                lambda: self._value > self._min_value, f"'{self._name}' is less than or equal {self._min_value}"
            )
        if self._max_value is not None:
            self._register_checker(
                lambda: self._value < self._max_value, f"'{self._name}' is bigger than or equal {self._max_value}"
            )
        return self

    def check_value_for_left_open_interval(self):
        if self._min_value is not None:
            self._register_checker(
                lambda: self._value > self._min_value, f"'{self._name}' is less than or equal {self._min_value}"
            )
        if self._max_value is not None:
            self._register_checker(
                lambda: self._value <= self._max_value, f"'{self._name}' is bigger than {self._max_value}"
            )
        return self

    def check_value_for_right_open_interval(self):
        if self._min_value is not None:
            self._register_checker(
                lambda: self._value >= self._min_value, f"'{self._name}' is less than {self._min_value}"
            )
        if self._max_value is not None:
            self._register_checker(
                lambda: self._value < self._max_value, f"'{self._name}' is bigger than or equal {self._max_value}"
            )
        return self


class IntValidator(NumValidator):
    """Int type validator."""

    def __init__(
        self,
        name: str,
        value: int,
        min_value: int = ValidatorParams.MIN_INT32.value,
        max_value: int = ValidatorParams.MAX_INT32.value,
        msg: str = "",
    ):
        super(IntValidator, self).__init__(name, value, min_value, max_value)

        self._register_checker(self._check_type, msg if msg else f"type of '{name}' is not int")

    def _check_type(self):
        # Bool is subclass of int.
        if isinstance(self._value, bool):
            return False
        return isinstance(self._value, int)


class FloatValidator(NumValidator):
    """Float type validator."""

    def __init__(
        self,
        name: str,
        value: float,
        min_value: float = ValidatorParams.MIN_FLOAT32.value,
        max_value: float = ValidatorParams.MAX_FLOAT32.value,
        msg: str = "",
    ):
        super(FloatValidator, self).__init__(name, value, min_value, max_value)

        self._register_checker(self._check_type, msg if msg else f"type of '{name}' is not float")

    def _check_type(self):
        return isinstance(self._value, float)


class ClassValidator(BaseValidator):
    """Check class validator."""

    def __init__(self, name: str, value: Any, classes: Union[Any, Tuple[Any]]):
        super(ClassValidator, self).__init__(name, value)

        self._classes = classes
        self._register_checker(
            lambda: isinstance(self._value, self._classes),
            f"type of '{self._name}' is not {self._classes}",
        )


class FileValidator(StringValidator):
    """Check if file is valid."""

    def __init__(self, name: str, value: str):
        super(FileValidator, self).__init__(name, value)
        self._register_checker(lambda: isinstance(self._value, str), "parameter value type is not str")

    def check_file_size(
        self, min_size=ValidatorParams.FILE_MIN_SIZE.value, max_size=ValidatorParams.FILE_MAX_SIZE.value
    ):
        file_stat = tf.io.gfile.stat(self._value)
        self._register_checker(
            lambda: min_size <= file_stat.length <= max_size,
            f"file size {file_stat.length} is invalid, not in [{min_size}, {max_size}]",
        )
        return self

    def check_not_soft_link(self):
        self._register_checker(
            lambda: os.path.abspath(self._value) == os.path.realpath(self._value),
            f"soft link or relative path {self._value} should not be in the path parameter",
        )
        return self

    def check_user_group(self):
        process_uid = os.geteuid()
        process_gid = os.getegid()
        stat_info = os.stat(self._value)
        file_uid = stat_info.st_uid
        file_gid = stat_info.st_gid
        self._register_checker(
            lambda: process_uid == file_uid or process_gid == file_gid, "invalid log file user or group."
        )
        return self

    def check_file_mode(self, unsupported_mode: int = 0o022):
        stat_info = os.stat(self._value)
        mode = stat.S_IMODE(stat_info.st_mode)
        self._register_checker(lambda: mode & unsupported_mode == 0, f"current file mode {oct(mode)} is unsupported")
        return self


class DirectoryValidator(StringValidator):
    def __init__(self, name, value, max_len=None, min_len=1):
        """
        @param value: the path, should not be emtpy string, should not contain double dot(../)
        """
        super(DirectoryValidator, self).__init__(name, value, max_len=max_len, min_len=min_len)
        self._register_checker(lambda: isinstance(value, str), "type is not str")

    @staticmethod
    def remove_prefix(string: Optional[str], prefix: Optional[str]) -> Tuple[bool, Optional[str]]:
        if string is None or prefix is None or len(string) < len(prefix):
            return False, string
        if string.startswith(prefix):
            return True, string[len(prefix):]
        else:
            return False, string

    @staticmethod
    def check_is_children_path(path_: str, target_: str):
        if not target_:
            return False

        try:
            realpath_ = os.path.realpath(path_)
        except (TypeError, ValueError, OSError):
            return False

        try:
            realpath_target = os.path.realpath(target_)
        except (TypeError, ValueError, OSError):
            return False

        is_prefix, rest_part = DirectoryValidator.remove_prefix(realpath_target, realpath_)

        if rest_part.startswith(os.path.sep):
            rest_part = rest_part.lstrip(os.path.sep)
        if is_prefix:
            joint_path = os.path.join(realpath_, rest_part)
            return os.path.realpath(joint_path) == realpath_target
        else:
            return False

    @staticmethod
    def __check_with_sensitive_words(path: str, words: List):
        _, name = os.path.split(path)
        if name:
            return not any(map(lambda x: x in path, words))
        else:
            return True

    def check_is_not_none(self):
        self._register_checker(lambda: self._value is not None and len(self._value) > 0, "Invalid directory parameter")
        return self

    def check_not_soft_link(self):
        self._register_checker(
            lambda: os.path.abspath(self._value) == os.path.realpath(self._value),
            f"path: {self._value} is soft link",
        )
        return self

    def path_should_exist(self, is_file=False, msg=None):
        self._register_checker(lambda: os.path.exists(self._value),
                               msg if msg else f"path: {self._value} does not exist")
        if is_file:
            self._register_checker(lambda: os.path.isfile(self._value),
                                   msg if msg else f"path: {self._value} is not a file")
        return self

    def with_blacklist(self, lst: List = None, exact_compare: bool = True, msg: str = None):
        if lst is None:
            lst = ["/usr/bin", "/usr/sbin", "/etc", "/usr/lib", "/usr/lib64", "/usr/local"]
        if len(lst) == 0:
            return self
        if msg is None:
            msg = "path should not in blacklist"
        if exact_compare:
            self._register_checker(lambda: self._value not in [os.path.realpath(each) for each in lst], msg)
        else:
            self._register_checker(
                lambda: not any([DirectoryValidator.check_is_children_path(each, self._value) for each in lst]), msg
            )
        return self

    def should_not_contains_sensitive_words(self, words: List = None, msg: str = None):
        if words is None:
            words = ["Key", "password", "privatekey"]
        if msg is None:
            msg = f"path: {self._value} contains sensitive words: {words}"
        self._register_checker(lambda: DirectoryValidator.__check_with_sensitive_words(self._value, words), msg)
        return self
