rm -rf build
mkdir build
cd build
export CMAKE_BUILD_PARALLEL_LEVEL=24
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=`python3 -c 'import torch;print(torch.utils.cmake_prefix_path)'` ..
cmake  --build . --config Release