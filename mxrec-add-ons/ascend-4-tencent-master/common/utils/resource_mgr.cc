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

#include "resource_mgr.h"
namespace aicpu {
ResourceMgr::~ResourceMgr() {
    const std::lock_guard<std::mutex> lock(mu_);
    handle_map_.clear();
    step_handle_.clear();
}

uint32_t ResourceMgr::Create(const uint64_t handle, TensorSeqPtr resource) {
    KERNEL_LOG_DEBUG("Create handle: [%ld]", handle);
    const std::lock_guard<std::mutex> lock(mu_);
    if (handle_map_.count(handle) != 0) {
        KERNEL_LOG_ERROR("handle [%ld] has already exist.", handle);
        return KERNEL_STATUS_PARAM_INVALID;
    } else {
        handle_map_[handle] = resource;
    }
    return KERNEL_STATUS_OK;
}

uint32_t ResourceMgr::Lookup(const uint64_t handle, TensorSeqPtr* resource) {
    KERNEL_LOG_DEBUG("Lookup handle: [%ld]", handle);
    const std::lock_guard<std::mutex> lock(mu_);
    const auto iter =  handle_map_.find(handle);
    if (iter == handle_map_.end()) {
        KERNEL_LOG_ERROR("handle [%ld] does not exist.", handle);
        return KERNEL_STATUS_PARAM_INVALID;
    }
    *resource = iter->second;
    return KERNEL_STATUS_OK;
}

uint32_t ResourceMgr::ClearSpecialStepResource(const uint64_t handle) {
    KERNEL_LOG_DEBUG("Clear handle: [%ld]", handle);
    const std::lock_guard<std::mutex> lock(mu_);
    for (auto &h : step_handle_) {
        if (h == handle) {
            const auto iter = handle_map_.find(h);
            if (iter == handle_map_.end()) {
                KERNEL_LOG_ERROR("handle [%ld] does not exist.", h);
                return KERNEL_STATUS_PARAM_INVALID;
            }
            step_handle_.erase(h);
            handle_map_.erase(iter);
            return KERNEL_STATUS_OK;
        }
    }
    return KERNEL_STATUS_PARAM_INVALID;
}

uint32_t ResourceMgr::ClearStepResource() {
    const std::lock_guard<std::mutex> lock(mu_);
    for (auto it = step_handle_.begin(); it != step_handle_.end();) {
        const auto iter = handle_map_.find(*it);
        if (iter == handle_map_.end()) {
            step_handle_.erase(it++);
            KERNEL_LOG_ERROR("handle [%ld] does not exist.", *it);
            continue;
        }
        step_handle_.erase(it++);
        handle_map_.erase(iter);
    }
    return KERNEL_STATUS_OK;
}

void ResourceMgr::ClearAllResource() {
    const std::lock_guard<std::mutex> lock(mu_);
    handle_map_.clear();
    step_handle_.clear();
    return;
}

SessionMgr* SessionMgr::GetInstance() {
    static SessionMgr inst;
    return &inst;
}

SessionMgr::~SessionMgr() {
    const std::lock_guard<std::mutex> lock(session_mutex_);
    session_map_.clear();
}

void SessionMgr::GetOrCreateSession(const uint64_t session_id, SessionPtr& session) {
    KERNEL_LOG_DEBUG("GetOrCreate session_id: [%ld]", session_id);
    const std::lock_guard<std::mutex> lock(session_mutex_);
    const auto iter = session_map_.find(session_id);
    if (iter != session_map_.end()) {
        session = iter->second;
    } else {
        SessionPtr sess = NewSession(session_id);
        session_map_.insert({session_id, sess});
        session = sess;
    }
    return;
}

uint32_t SessionMgr::CreateSession(const uint64_t session_id) {
    KERNEL_LOG_DEBUG("Create session_id: [%ld]", session_id);
    const std::lock_guard<std::mutex> lock(session_mutex_);
    const auto iter = session_map_.find(session_id);
    if (iter != session_map_.end()) {
        KERNEL_LOG_ERROR("session_id [%ld] has already exist.", session_id);
        return KERNEL_STATUS_PARAM_INVALID;
    }
    session_map_.insert({session_id, NewSession(session_id)});
    return KERNEL_STATUS_OK;
}

uint32_t SessionMgr::GetSession(const uint64_t session_id, SessionPtr& session) {
    KERNEL_LOG_DEBUG("Get session_id: [%ld]", session_id);
    const std::lock_guard<std::mutex> lock(session_mutex_);
    const auto iter = session_map_.find(session_id);
    if (iter == session_map_.end()) {
        KERNEL_LOG_ERROR("session_id [%ld] does not exist.", session_id);
        return KERNEL_STATUS_PARAM_INVALID;
    }
    session = iter->second;
    return KERNEL_STATUS_OK;
}

uint32_t SessionMgr::DestroySession(const uint64_t session_id) {
    KERNEL_LOG_DEBUG("Destroy session_id: [%ld]", session_id);
    const std::lock_guard<std::mutex> lock(session_mutex_);
    const auto iter = session_map_.find(session_id);
    if (iter == session_map_.end()) {
        KERNEL_LOG_ERROR("session_id [%ld] does not exist.", session_id);
        return KERNEL_STATUS_PARAM_INVALID;
    }
    session_map_.erase(iter);
    return KERNEL_STATUS_OK;
}

void SessionMgr::GetRm(const uint64_t session_id, const uint64_t container_id, ResourceMgrPtr &out_rm) {
    KERNEL_LOG_DEBUG("session_id: [%ld], container_id: [%ld]", session_id, container_id);
    SessionPtr session;
    SessionMgr::GetInstance()->GetOrCreateSession(session_id, session);
    ResourceMgrPtr rm;
    session->GetOrCreateRm(container_id, rm);
    out_rm = rm;
    return;
}

Session::Session(const uint64_t session_id) : session_id_(session_id) {}

Session::~Session() {
    const std::lock_guard<std::mutex> lock(mutex_);
    rm_map_.clear();
}

void Session::GetOrCreateRm(const uint64_t container_id, ResourceMgrPtr &rm) {
    KERNEL_LOG_DEBUG("GetOrCreateRm container_id: [%ld]", container_id);
    const std::lock_guard<std::mutex> lock(mutex_);
    const auto iter = rm_map_.find(container_id);
    if (iter != rm_map_.end()) {
        rm = iter->second;
    } else {
        ResourceMgrPtr tmp_rm = NewRm();
        rm_map_.insert({container_id, tmp_rm});
        rm = tmp_rm;
    }
    return;
}

uint32_t Session::CreateRm(const uint64_t container_id) {
    KERNEL_LOG_DEBUG("Create container_id: [%ld]", container_id);
    const std::lock_guard<std::mutex> lock(mutex_);
    const auto iter = rm_map_.find(container_id);
    if (iter != rm_map_.end()) {
        KERNEL_LOG_ERROR("container_id [%ld] has already exist.", container_id);
        return KERNEL_STATUS_PARAM_INVALID;
    }
    rm_map_.insert({container_id, NewRm()});
    return KERNEL_STATUS_OK;
}

uint32_t Session::ClearRm(const uint64_t container_id) {
    KERNEL_LOG_DEBUG("Clear container_id: [%ld]", container_id);
    const std::lock_guard<std::mutex> lock(mutex_);
    const auto iter = rm_map_.find(container_id);
    if (iter == rm_map_.end()) {
        KERNEL_LOG_ERROR("container_id [%ld] does not exist.", container_id);
        return KERNEL_STATUS_PARAM_INVALID;
    } else {
        auto rm = iter->second;
        rm->ClearAllResource();
        return KERNEL_STATUS_OK;
    }
}

uint32_t Session::ClearAllRm() {
    const std::lock_guard<std::mutex> lock(mutex_);
    for (auto &id : container_id_) {
        auto ret = ClearRm(id);
        if (ret != 0) {
            KERNEL_LOG_ERROR("clear container_id [%ld] failed.", id);
            return KERNEL_STATUS_PARAM_INVALID;
        }
    }
    return KERNEL_STATUS_OK;
}
uint32_t Session::GetRm(const uint64_t container_id, ResourceMgrPtr &rm) {
    KERNEL_LOG_DEBUG("Get container_id: [%ld]", container_id);
    const std::lock_guard<std::mutex> lock(mutex_);
    const auto iter = rm_map_.find(container_id);
    if (iter == rm_map_.end()) {
        KERNEL_LOG_ERROR("container_id [%ld] does not exist.", container_id);
        return KERNEL_STATUS_PARAM_INVALID;
    }
    rm = iter->second;
    return KERNEL_STATUS_OK;
}
}