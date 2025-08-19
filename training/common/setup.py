import os
import argparse
import sys
from setuptools import setup, find_packages
import shutil

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default="7.2.RC1")
    parser.add_argument("--discription", default="")

    # 分离 setuptools 参数和自定义参数
    args, unknown = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + unknown  # 将剩余参数传回 setuptools
    return args

args = parse_args()

if os.path.exists("mx_rec_common"):
    shutil.rmtree("mx_rec_common")
shutil.copytree("python", "mx_rec_common")
setup(
    name='mx_rec_common',
    version=args.version,
    author='HUAWEI Inc',
    description='MindSDK Recommend',
    long_description=args.discription,
    # include mx_rec
    packages=find_packages(
        where=".",
        include=["mx_rec_common*"]
    ),
    # other file
    package_data={'': ['tools/*', 'tools/*/*', '*.yml', '*.sh', '*.so*']},
    # dependency
    python_requires='>=3.7.5'
)
shutil.rmtree("mx_rec_common")