#!/bin/bash
# Copyright 2024. Huawei Technologies Co.,Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

deterministic_patch(){
    sed -i "s/allow_mix_precision/must_keep_origin_dtype/g" config.py
    sed -i '/must_keep_origin_dtype/a\    custom_op.parameter_map["deterministic"].i = 1' config.py

    sed -i "/tf.compat.v1.disable_eager_execution()/a\np.random.seed(128)" main.py
    sed -i "/tf.compat.v1.disable_eager_execution()/a\tf.random.set_random_seed(128)" main.py
    sed -i "/tf.compat.v1.disable_eager_execution()/a\import numpy as np" main.py

    sed -i "s/tf.compat.v1.truncated_normal_initializer()/tf.compat.v1.constant_initializer(0)/g" main.py

    sed -i "s/self.session.run(\[self.train_ops, self.train_model.loss_list\])/_,loss=self.session.run(\[self.train_ops, self.train_model.loss_list\])/g" run_mode.py
    sed -i '/self.session.run(\[self.train_ops, self.train_model.loss_list\])/a\                logger.info(f"deterministic_loss: {loss\[0\]}")' run_mode.py
}

if [ ! -e deterministic_patch_file ];then
    deterministic_patch
    touch deterministic_patch_file
fi

sh run.sh main.py | tee log

grep -rn "loss" log | grep "1,0" | awk '{print $NF}'> loss

soc_name=`python3 -c 'import acl;print(acl.get_soc_name())'`
echo "soc_name: $soc_name"

loss_file=deterministic_loss/$soc_name

if [ ! -e $loss_file ];then
    echo "$loss_file file does not exist"
    rm -f loss
    exit
fi

diff $loss_file loss

if [ $? -eq 0 ]; then
  echo "deterministic loss check passed"
else
  echo "deterministic loss check failed"
fi

rm -f loss
