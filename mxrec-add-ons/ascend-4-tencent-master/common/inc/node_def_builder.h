/**
 * Copyright (c) 2022-2022 Huawei Technologies Co., Ltd.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef NODE_DEF_BUILDER_H
#define NODE_DEF_BUILDER_H
#include <string>
#include "cpu_kernel.h"
#include "status.h"
#include "cpu_kernel_register.h"
#include "aicpu_task_struct.h"
#include "device_cpu_kernel.h"

namespace aicpu {
class NodeDefBuilder {
  public:
  	struct InputOutputNode{
  		std::string node;
  		aicpu::DataType dType;
  		std::vector<int64_t> dims;
  		void *data;
		aicpu::Format format;
  	};

	static std::shared_ptr<NodeDef> CreateNodeDef();

	NodeDefBuilder(NodeDef *nodeDef, std::string name, std::string opName);

	NodeDefBuilder& Input(const InputOutputNode& input);

	NodeDefBuilder& Output(const InputOutputNode& output);

	NodeDefBuilder& Attr(std::string name, int32_t value);

	NodeDefBuilder& Attr(std::string name, int64_t value);

	NodeDefBuilder& Attr(std::string name, float value);

	NodeDefBuilder& Attr(std::string name, double value);

	NodeDefBuilder& Attr(std::string name, bool value);

	NodeDefBuilder& Attr(std::string name, aicpu::DataType value);

	NodeDefBuilder& Attr(std::string name, const std::vector<bool> &value);

	NodeDefBuilder& Attr(std::string name, const std::string &value);

	NodeDefBuilder& Attr(std::string name, const std::vector<std::string> &value);

	NodeDefBuilder& Attr(std::string name, const std::vector<int64_t> &value);

	NodeDefBuilder& Attr(std::string name, const std::vector<std::vector<int64_t>> &value);

	NodeDefBuilder& Attr(std::string name, const std::vector<float> &value);

	NodeDefBuilder& Attr(std::string name, const std::vector<aicpu::DataType> &value);

	NodeDefBuilder& Attr(std::string name, const std::vector<int64_t> &dims, std::string type);

	NodeDefBuilder& Attr(std::string name, const std::vector<std::vector<int64_t>> &shapeLists, std::string type);

	NodeDefBuilder& Attr(std::string name, aicpu::Tensor *tensor);

	NodeDefBuilder& Attr(std::string name, std::vector<aicpu::Tensor *> &tensors);

  private:
  	void BuildNodeFromInputOutputNode(const InputOutputNode& node, bool isInput);

  	NodeDef *nodeDef_;

  	std::string name_;

  	std::string opName_;
};
}

#endif
