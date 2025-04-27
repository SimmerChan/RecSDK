#!/bin/bash
# Copyright 2025. Huawei Technologies Co.,Ltd. All rights reserved.

set -e

VERSION="1.1.0-npu"
ARCH=$(uname -m)

package_name="Ascend-mindxsdk-torchrec-"${VERSION}"-linux-"${ARCH}".tar.gz"
if [ -f "${package_name}"]; then
  rm "${package_name}"
fi
python3 setup.py bdist_wheel --plat-name linux_"${ARCH}"
cp requirements.txt dist/
tar -czvf "${package_name}" -C dist .