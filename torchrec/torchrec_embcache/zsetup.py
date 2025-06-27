from setuptools import find_packages, setup

setup(
    name="torchrec_embcache",
    version="0.2.0",
    package_data={
        'torchrec_embcache': ['*.so*']
    },
    package_dir={"": "src"},
    packages=find_packages("src"),
    install_requires=["torch"],
    python_requires=">=3.7",
    author="Huawei Inc",
    description="Embedding cache implementation",
    zip_safe=False,
)