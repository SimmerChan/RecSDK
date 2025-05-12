/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <sstream>
#include <unordered_map>

#include "absl/strings/str_split.h"
#include "llvm/Support/CommandLine.h"
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Support/Timing.h"    // from @llvm-project
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/init_main.h"
#include "tf_mlir/mlir_converter.h"

namespace tensorflow {
std::unordered_map<std::string, std::string> parse_envs(
    llvm::cl::list<std::string>& envs) {
  std::unordered_map<std::string, std::string> env_pair;
  for (auto& env : envs) {
    std::vector<std::string> kvs = absl::StrSplit(env, '=');
    if (kvs.size() != 2) {
      LOG(FATAL) << "env option value should be ENV=VAL: " << env;
    }
    env_pair[kvs[0]] = kvs[1];
  }
  return env_pair;
}

Status RealMain(int argc, char** argv) {
  llvm::cl::OptionCategory tf_mlir("tf_mlir", "Options for tf_mlir.");

  llvm::cl::opt<std::string> input_fn{
      llvm::cl::Positional, llvm::cl::desc("<input file>"), llvm::cl::init("-"),
      llvm::cl::cat(tf_mlir)};
  llvm::cl::opt<std::string> output_fn{
      llvm::cl::Positional, llvm::cl::desc("<output file>"),
      llvm::cl::init("/tmp/compilation_result.pbtxt"), llvm::cl::cat(tf_mlir)};

  llvm::cl::opt<std::string> convert_pb_txt{
      "convert-to-pbtxt",
      llvm::cl::desc("convert the protobuf message into txt format."),
      llvm::cl::cat(tf_mlir)};

  llvm::cl::list<std::string> envs{
      "env",
      llvm::cl::desc("override environment variables in the input pb file,"
                     "this option can be specified zero or more times."),
      llvm::cl::ZeroOrMore, llvm::cl::cat(tf_mlir)};

  llvm::cl::HideUnrelatedOptions(tf_mlir);
  mlir::registerPassManagerCLOptions();
  mlir::registerDefaultTimingManagerCLOptions();
  mlir::registerMLIRContextCLOptions();
  llvm::cl::ParseCommandLineOptions(argc, argv, "Welcome tf_mlir!\n");

  tensorflow::npu_xla::CompilerInput input;
  TF_RETURN_IF_ERROR(ReadBinaryProto(Env::Default(), input_fn, &input));

  if (!convert_pb_txt.empty()) {
    VLOG(0) << "the output protobuf message file with text format: "
            << convert_pb_txt;
    return WriteTextProto(Env::Default(), convert_pb_txt, input);
  }

  if (!input.env().empty()) {
    VLOG(1) << "Setting up environment variable compiler:";
    for (auto& kv : input.env()) {
      setenv(kv.first.c_str(), kv.second.c_str(), 1);
      VLOG(1) << "    " << kv.first << "=" << kv.second;
    }
  }

  if (!envs.empty()) {
    auto cmd_envs = parse_envs(envs);
    VLOG(1) << "Setting up environment variable from cmdline:";
    for (auto& kv : cmd_envs) {
      VLOG(1) << "    " << kv.first << "=" << kv.second;
      setenv(kv.first.c_str(), kv.second.c_str(), 1);
    }
  }

  tensorflow::npu_xla::MlirConverter converter;
  TF_RETURN_IF_ERROR(converter.ConvertGraphdefToStablehlo(input, output_fn));

  return absl::OkStatus();
}
}  // namespace tensorflow

int main(int argc, char** argv) {
  tensorflow::port::InitMain(argv[0], &argc, &argv);
  auto status = tensorflow::RealMain(argc, argv);
  if (!status.ok()) {
    std::string err_msg = status.ToString();
    absl::StatusCode code = status.code();
    VLOG(0) << "Failed! " << err_msg << " code " << code;
    return 1;
  }
  return 0;
}