#export ASCEND_SLOG_PRINT_TO_STDOUT=1
#export ASCEND_GLOBAL_LOG_LEVEL=1
#export DUMP_GE_GRAPH=1
#export DUMP_GRAPH_LEVEL=1
#export DUMP_GRAPH_PATH=./dumpgraph
#export PROFILING_MODE=true
#export PROFILING_OPTIONS='{"output":"/tmp/profiling", "task_trace":"on", "fp_point":"","aic_metrics":"PipeUtilization"}'

taskset -c 0-32 /home/lmp/serving-1.15.0/bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server \
  --model_name=saved_model \
  --model_base_path=$(pwd)/inference_model/saved_model/ \
  --port=9999 \
  --rest_api_prot=9991 \
  --platform_config_file=test.cfg