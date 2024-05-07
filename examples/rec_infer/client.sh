export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/larest/lib64:/usr/local/Ascend/ascend-toolkit/larest/lib64/plugin/opskernel:$LD_LIBRARY_PATH
unset http_proxy
unset https_proxy
python3 client.py