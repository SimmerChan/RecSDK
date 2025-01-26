export ASCEND_RT_VISIBLE_DEVICES=0,1
torchx run -s local_cwd dist.ddp -j 2 --script test_hybrid.py