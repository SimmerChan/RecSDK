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

#ifndef AICPU_RESOURCE_MGR_H
#define AICPU_RESOURCE_MGR_H
#include <set>
#include <map>
#include <mutex>
#include <atomic>
#include "tensor_sequence.h"
namespace aicpu {
using TensorSeqPtr = std::shared_ptr<TensorSeq>;
class ResourceMgr {
public:
  ResourceMgr() = default;
  ~ResourceMgr();
  uint32_t Create(const uint64_t handle, TensorSeqPtr resource);
  uint32_t Lookup(const uint64_t handle, TensorSeqPtr* resource);
  uint32_t ClearStepResource();
  uint32_t ClearSpecialStepResource(const uint64_t handle);
  void ClearAllResource();
  uint64_t GetHandle() const {
    static std::atomic<uint64_t> handle(0UL);
    return ++handle;
  }
  void StoreStepHandle(const uint64_t id) {
    const std::lock_guard<std::mutex> lock(mu_);
    step_handle_.insert(id);
  }
private:
  mutable std::mutex mu_;
  std::map<uint64_t, TensorSeqPtr> handle_map_;
  std::set<uint64_t> step_handle_;
};

using ResourceMgrPtr = std::shared_ptr<ResourceMgr>;
class Session {
public:
  explicit Session(const uint64_t session_id);
  ~Session();
  uint32_t CreateRm(const uint64_t container_id);
  void GetOrCreateRm(const uint64_t container_id, ResourceMgrPtr &rm);
  uint32_t GetRm(const uint64_t container_id, ResourceMgrPtr &rm);
  uint32_t ClearRm(const uint64_t container_id);
  uint32_t ClearAllRm();
  void StoreStepcontainerId(const uint64_t id) {
    const std::lock_guard<std::mutex> lock(mutex_);
    container_id_.insert(id);
  }
private:
  using IdToRmMap = std::map<uint64_t, ResourceMgrPtr>;
  uint64_t session_id_;
  std::set<uint64_t> container_id_;
  IdToRmMap rm_map_;
  mutable std::mutex mutex_;
  ResourceMgrPtr NewRm() {
    return std::make_shared<ResourceMgr>();
  }
};

using SessionPtr = std::shared_ptr<Session>;
class SessionMgr {
public:
  static SessionMgr *GetInstance();
  SessionMgr() = default;
  ~SessionMgr();
  uint32_t CreateSession(const uint64_t session_id);
  uint32_t DestroySession(const uint64_t session_id);
  void GetOrCreateSession(const uint64_t session_id, SessionPtr& session);
  uint32_t GetSession(const uint64_t session_id, SessionPtr &out_session);
  void GetRm(const uint64_t session_id, const uint64_t container_id,
                 ResourceMgrPtr &out_rm);
private:
  SessionPtr NewSession(const uint64_t session_id) {
    return std::make_shared<Session>(session_id);
  }
private:
  using IdToSessionMap = std::map<uint64_t, SessionPtr>;
  mutable std::mutex session_mutex_;
  IdToSessionMap session_map_;
};
}  //  end namespace aicpu
#endif  // AICPU_RESOURCE_MGR_H