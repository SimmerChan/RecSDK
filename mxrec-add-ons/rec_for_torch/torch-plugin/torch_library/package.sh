#!/bin/bash
set -e

# 输出目录
OUTPUT_DIR=output

if [ -d output ]; then
  rm -r "$OUTPUT_DIR"
fi

# 创建一个空的文本文件存储文件夹列表
> version_folders_list.txt

# 遍历当前目录所有文件夹，添加到文本文件中
for dir in */; do
  echo "$dir" >> version_folders_list.txt
done

if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
fi

# version_folders_list.txt文件夹中的每一行

while IFS= read -r version_folder; do
  if [ ! -d "$OUTPUT_DIR"/"$version_folder" ]; then
    mkdir -p "$OUTPUT_DIR"/"$version_folder"
  fi
  # 指定要读取的文件路径
  filename="op_list.txt"
  # 检查文件是否存在
  if [ ! -f "$version_folder"/"$filename" ]; then
    echo "文件不存在: "$version_folder"/"$filename""
    exit 1
  fi

  dos2unix "$version_folder"/"$filename"

  # 逐行读取
  while  IFS= read -r op_dir; do
    if [ ! -e "$op_dir" ]; then
      echo "算子目录不存在:$op_dir"
      exit 1
    fi
    cp "$op_dir" "$OUTPUT_DIR"/"$version_folder" -rd
  done < "$version_folder"/"$filename"
done < version_folders_list.txt
# torch2.6.0需要使用CXX11_ABI=1构建.参考官网说明: https://pytorch.ac.cn/blog/pytorch2-6/
find output/2.6.0 -type f -name "CMakeLists.txt" -exec sed -i 's/-D_GLIBCXX_USE_CXX11_ABI=0/-D_GLIBCXX_USE_CXX11_ABI=1/g' {} \;

rm version_folders_list.txt
echo "package finished"
