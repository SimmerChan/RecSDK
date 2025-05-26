/* Copyright 2025. Huawei Technologies Co.,Ltd. All rights reserved.

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

#include <cstdlib>
#include "tensorflow/c/experimental/stream_executor/stream_executor.h"
#include "tsl/platform/logging.h"
#include "tensorflow/core/platform/stacktrace.h"

#include "common_hdrs/types.h"
#include "adaptor/device_interface.h"
#include "pluggable_npu_device.h"

/** Copy from stream_excutor_test_util.h **/
/** ------------------------------------ **/
struct SP_Stream_st {
    explicit SP_Stream_st(int id) : streamId(id) {}
    int streamId;
};

struct SP_Event_st {
    explicit SP_Event_st(int id) : eventId(id) {}
    int eventId;
};

struct SP_Timer_st {
    explicit SP_Timer_st(int id) : timerId(id) {}
    int timerId;
};
/** ------------------------------------ **/

namespace tensorflow {
namespace npu_xla {
constexpr int K_DEVICE_COUNT = 1;
constexpr char K_DEVICE_NAME[] = "ASCEND_NPU";

/*** Functions for creating SP_Device ***/
void GetDeviceCount(const SP_Platform* platform, int* deviceCount, TF_Status* status)
{
    VLOG(VLOG_LEVEL_2) << "Pluggable NPU GetDeviceCount";
    TF_SetStatus(status, TF_OK, "");
    *deviceCount = K_DEVICE_COUNT;
}

void CreateDevice(const SP_Platform* platform, SE_CreateDeviceParams* params, TF_Status* status)
{
    VLOG(VLOG_LEVEL_2) << "Pluggable NPU CreateDevice";
    TF_SetStatus(status, TF_OK, "");
    params->device->struct_size = {SP_DEVICE_STRUCT_SIZE};
    params->device->ordinal = 0;
    params->device->device_handle = nullptr;
    params->ordinal = 0;
}

void DestroyDevice(const SP_Platform* platform, SP_Device* device)
{
    VLOG(VLOG_LEVEL_2) << "Pluggable NPU DestroyDevice";
}

void CreateDeviceFns(const SP_Platform* platform, SE_CreateDeviceFnsParams* params, TF_Status* status)
{
    VLOG(VLOG_LEVEL_2) << "Pluggable NPU CreateDeviceFns";
    TF_SetStatus(status, TF_OK, "");
    params->device_fns->struct_size = {SP_DEVICE_FNS_STRUCT_SIZE};
}

void DestroyDeviceFns(const SP_Platform* platform, SP_DeviceFns* device_fns)
{
    VLOG(VLOG_LEVEL_2) << "Pluggable NPU DestroyDeviceFns";
}

/*** Functions for creating SP_StreamExecutor ***/
void Allocate(const SP_Device* const device, uint64_t size, int64_t memorySpace, SP_DeviceMemoryBase* const mem)
{
    VLOG(VLOG_LEVEL_2) << "Pluggable NPU Allocate";
    DeviceInterface& adaptor = DeviceInterface::Create(device->ordinal);
    mem->opaque = adaptor.Allocate(size);
    mem->size = mem->opaque != nullptr ? size : 0;
    mem->payload = 0;
    VLOG(VLOG_LEVEL_2) << "Pluggable NPU Allocate opaque " << mem->opaque;
}

void Deallocate(const SP_Device* const device, SP_DeviceMemoryBase* const mem)
{
    VLOG(VLOG_LEVEL_2) << "Pluggable NPU Deallocate";
    DeviceInterface& adaptor = DeviceInterface::Create(device->ordinal);
    adaptor.Deallocate(mem->opaque);
    VLOG(VLOG_LEVEL_2) << "Pluggable NPU Deallocate opaque " << mem->opaque;
    mem->opaque = nullptr;
    mem->size = 0;
}

void* HostMemoryAllocate(const SP_Device* const device, uint64_t size)
{
    VLOG(VLOG_LEVEL_2) << "Pluggable NPU HostMemoryAllocate";
    if (size == 0) {
        VLOG(VLOG_LEVEL_3) << "HostMemoryAllocate error, input size:" << size;
        return nullptr;
    }

    void* mem = std::aligned_alloc(64, size);
    if (mem == nullptr) {
        VLOG(VLOG_LEVEL_3) << "HostMemoryAllocate new error, input size:" << size;
    }
    VLOG(VLOG_LEVEL_2) << "HostMemoryAllocate size:" << size << " mem:" << mem;
    return mem;
}

void HostMemoryDeallocate(const SP_Device* const device, void* mem)
{
    VLOG(VLOG_LEVEL_2) << "Pluggable NPU HostMemoryDeallocate mem:" << mem;
    free(mem);
}

TF_Bool GetAllocatorStats(const SP_Device* const device, SP_AllocatorStats* const stats)
{
    VLOG(VLOG_LEVEL_2) << "Pluggable NPU GetAllocatorStats";
    return true;
}

TF_Bool DeviceMemoryUsage(const SP_Device* const device, int64_t* const free, int64_t* const total)
{
    VLOG(VLOG_LEVEL_2) << "Pluggable NPU DeviceMemoryUsage";
    return true;
}

void CreateStream(const SP_Device* const device, SP_Stream* stream, TF_Status* const status)
{
    VLOG(VLOG_LEVEL_2) << "Pluggable NPU CreateStream";
    *stream = nullptr;
}

void DestroyStream(const SP_Device* const device, SP_Stream stream)
{
    VLOG(VLOG_LEVEL_2) << "Pluggable NPU DestroyStream";
}

void CreateStreamDependency(const SP_Device* const device, SP_Stream dependent, SP_Stream other,
                            TF_Status* const status)
{
    VLOG(VLOG_LEVEL_2) << "Pluggable NPU CreateStreamDependency";
}

void GetStreamStatus(const SP_Device* const device, SP_Stream stream, TF_Status* const status)
{
    VLOG(VLOG_LEVEL_2) << "Pluggable NPU GetStreamStatus";
}

void CreateEvent(const SP_Device* const device, SP_Event* event, TF_Status* const status)
{
    VLOG(VLOG_LEVEL_2) << "Pluggable NPU CreateEvent";
    *event = new SP_Event_st(0);
    VLOG(VLOG_LEVEL_2) << "Pluggable NPU CreateEvent event:" << *event;
    TF_SetStatus(status, TF_OK, "");
}

void DestroyEvent(const SP_Device* const device, SP_Event event)
{
    VLOG(VLOG_LEVEL_2) << "Pluggable NPU DestroyEvent";
    delete event;
    VLOG(VLOG_LEVEL_2) << "Pluggable NPU DestroyEvent event:" << event;
}

SE_EventStatus GetEventStatus(const SP_Device* const device, SP_Event event)
{
    VLOG(VLOG_LEVEL_2) << "Pluggable NPU GetEventStatus";
    return SE_EVENT_COMPLETE;
}

void RecordEvent(const SP_Device* const device, SP_Stream stream, SP_Event event, TF_Status* const status)
{
    VLOG(VLOG_LEVEL_2) << "Pluggable NPU RecordEvent";
}

void WaitForEvent(const SP_Device* const device, SP_Stream stream, SP_Event event, TF_Status* const status)
{
    VLOG(VLOG_LEVEL_2) << "Pluggable NPU WaitForEvent";
}

void CreateTimer(const SP_Device* const device, SP_Timer* timer, TF_Status* const status)
{
    VLOG(VLOG_LEVEL_2) << "Pluggable NPU CreateTimer";
}

void DestroyTimer(const SP_Device* const device, SP_Timer timer)
{
    VLOG(VLOG_LEVEL_2) << "Pluggable NPU DestroyTimer";
}

void StartTimer(const SP_Device* const device, SP_Stream stream, SP_Timer timer, TF_Status* const status)
{
    VLOG(VLOG_LEVEL_2) << "Pluggable NPU StartTimer";
}

void StopTimer(const SP_Device* const device, SP_Stream stream, SP_Timer timer, TF_Status* const status)
{
    VLOG(VLOG_LEVEL_2) << "Pluggable NPU StopTimer";
}

void SyncMemcpyDToH(const SP_Device* const device, void* hostDst, const SP_DeviceMemoryBase* const device_src,
                    uint64_t size, TF_Status* const status)
{
    VLOG(VLOG_LEVEL_2) << "Pluggable NPU SyncMemcpyDToH";
    DeviceInterface& adaptor = DeviceInterface::Create(device->ordinal);
    auto success = adaptor.MemcpyDToH(hostDst, size, device_src->opaque, device_src->size);
    if (!success) {
        TF_SetStatus(status, TF_INTERNAL, "MemcpyDToH failed");
    } else {
        TF_SetStatus(status, TF_OK, "");
    }
}

void SyncMemcpyHToD(const SP_Device* const device, SP_DeviceMemoryBase* const device_dst, const void* hostSrc,
                    uint64_t size, TF_Status* const status)
{
    VLOG(VLOG_LEVEL_2) << "Pluggable NPU SyncMemcpyHToD";
    DeviceInterface& adaptor = DeviceInterface::Create(device->ordinal);
    auto success = adaptor.MemcpyHToD(device_dst->opaque, device_dst->size, hostSrc, size);
    if (!success) {
        TF_SetStatus(status, TF_INTERNAL, "MemcpyHToD failed");
    } else {
        TF_SetStatus(status, TF_OK, "");
    }
}

void MemcpyDToH(const SP_Device* const device, SP_Stream stream, void* hostDst,
                const SP_DeviceMemoryBase* const device_src, uint64_t size, TF_Status* const status)
{
    VLOG(VLOG_LEVEL_2) << "Pluggable NPU MemcpyDToH";
    SyncMemcpyDToH(device, hostDst, device_src, size, status);
}

void MemcpyHToD(const SP_Device* const device, SP_Stream stream, SP_DeviceMemoryBase* const device_dst,
                const void* hostSrc, uint64_t size, TF_Status* const status)
{
    VLOG(VLOG_LEVEL_2) << "Pluggable NPU MemcpyHToD";
    SyncMemcpyHToD(device, device_dst, hostSrc, size, status);
}

void BlockHostForEvent(const SP_Device* const device, SP_Event event, TF_Status* const status)
{
    VLOG(VLOG_LEVEL_2) << "Pluggable NPU BlockHostForEvent";
}

void SynchronizeAllActivity(const SP_Device* const device, TF_Status* const status)
{
    VLOG(VLOG_LEVEL_2) << "Pluggable NPU SynchronizeAllActivity";
}

TF_Bool HostCallback(const SP_Device* const device, SP_Stream stream, SE_StatusCallbackFn const callback_fn,
                     void* const callback_arg)
{
    VLOG(VLOG_LEVEL_2) << "Pluggable NPU HostCallback";
    return true;
}

void MemZero(const SP_Device* device, SP_Stream stream, SP_DeviceMemoryBase* location, uint64_t size, TF_Status* status)
{
    VLOG(VLOG_LEVEL_2) << "Pluggable NPU MemZero";
}

void Memset(const SP_Device* device, SP_Stream stream, SP_DeviceMemoryBase* location, uint8_t pattern, uint64_t size,
            TF_Status* status)
{
    VLOG(VLOG_LEVEL_2) << "Pluggable NPU Memset";
}

void Memset32(const SP_Device* device, SP_Stream stream, SP_DeviceMemoryBase* location, uint32_t pattern, uint64_t size,
              TF_Status* status)
{
    VLOG(VLOG_LEVEL_2) << "Pluggable NPU Memset32";
}

void PopulateNpuStreamExecutor(SP_StreamExecutor* se)
{
    *se = {SP_STREAMEXECUTOR_STRUCT_SIZE};
    se->allocate = Allocate;
    se->deallocate = Deallocate;
    se->host_memory_allocate = HostMemoryAllocate;
    se->host_memory_deallocate = HostMemoryDeallocate;
    se->get_allocator_stats = GetAllocatorStats;
    se->device_memory_usage = DeviceMemoryUsage;
    se->create_stream = CreateStream;
    se->destroy_stream = DestroyStream;
    se->create_stream_dependency = CreateStreamDependency;
    se->get_stream_status = GetStreamStatus;
    se->create_event = CreateEvent;
    se->destroy_event = DestroyEvent;
    se->get_event_status = GetEventStatus;
    se->record_event = RecordEvent;
    se->wait_for_event = WaitForEvent;
    se->create_timer = CreateTimer;
    se->destroy_timer = DestroyTimer;
    se->start_timer = StartTimer;
    se->stop_timer = StopTimer;
    se->memcpy_dtoh = MemcpyDToH;
    se->memcpy_htod = MemcpyHToD;
    se->sync_memcpy_dtoh = SyncMemcpyDToH;
    se->sync_memcpy_htod = SyncMemcpyHToD;
    se->block_host_for_event = BlockHostForEvent;
    se->synchronize_all_activity = SynchronizeAllActivity;
    se->host_callback = HostCallback;
    se->mem_zero = MemZero;
    se->memset = Memset;
    se->memset32 = Memset32;
}

void CreateStreamExecutor(const SP_Platform* platform, SE_CreateStreamExecutorParams* params, TF_Status* status)
{
    VLOG(VLOG_LEVEL_2) << "Pluggable NPU CreateStreamExecutor";
    TF_SetStatus(status, TF_OK, "");
    PopulateNpuStreamExecutor(params->stream_executor);
}

void DestroyStreamExecutor(const SP_Platform* platform, SP_StreamExecutor* se)
{
    VLOG(VLOG_LEVEL_2) << "Pluggable NPU DestroyStreamExecutor";
}

uint64_t Nanoseconds(SP_Timer timer)
{
    return timer->timerId;
}

void PopulateNpuTimerFns(SP_TimerFns* timer_fns)
{
    timer_fns->nanoseconds = Nanoseconds;
}

void CreateTimerFns(const SP_Platform* platform, SP_TimerFns* timer_fns, TF_Status* status)
{
    TF_SetStatus(status, TF_OK, "");
    PopulateNpuTimerFns(timer_fns);
}

void DestroyTimerFns(const SP_Platform* platform, SP_TimerFns* timer_fns) {}

void PopulateNpuPlatform(SP_Platform* platform, SP_PlatformFns* platform_fns)
{
    *platform = {SP_PLATFORM_STRUCT_SIZE};
    platform->name = K_DEVICE_NAME;
    platform->type = DEVICE_NPU;
    platform_fns->get_device_count = GetDeviceCount;
    platform_fns->create_device = CreateDevice;
    platform_fns->destroy_device = DestroyDevice;
    platform_fns->create_device_fns = CreateDeviceFns;
    platform_fns->destroy_device_fns = DestroyDeviceFns;
    platform_fns->create_stream_executor = CreateStreamExecutor;
    platform_fns->destroy_stream_executor = DestroyStreamExecutor;
    platform_fns->create_timer_fns = CreateTimerFns;
    platform_fns->destroy_timer_fns = DestroyTimerFns;
}

void DestroyPlatform(SP_Platform* platform) {}

void DestroyPlatformFns(SP_PlatformFns* platform_fns) {}

void PopulateNpuPlatformRegistrationParams(SE_PlatformRegistrationParams* const params)
{
    PopulateNpuPlatform(params->platform, params->platform_fns);
    params->destroy_platform = DestroyPlatform;
    params->destroy_platform_fns = DestroyPlatformFns;
}

}  // namespace npu_xla
}  // namespace tensorflow

extern "C" {
/*
 * Entry point for NPU device plugin initialization in TensorFlow.
 * This function is automatically detected and called by TensorFlow's plugin
 * system to register and initialize the NPU device as a StreamExecutor
 * platform.
 *
 * @param params: Platform registration parameters that will be populated with
 * NPU platform information
 * @param status: Status object to report any errors during initialization
 */
void SE_InitPlugin(SE_PlatformRegistrationParams* const params, TF_Status* const status)
{
    TF_SetStatus(status, TF_OK, "");
    tensorflow::npu_xla::PopulateNpuPlatformRegistrationParams(params);
}
}