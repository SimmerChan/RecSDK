#include "device_adaptor.h"
#include "acl_adaptor.h"
#include <cstdlib>

using namespace tensorflow::npu_xla;

// 用于扩展，可在此检查环境变量，返回对应实例
DeviceInterface& DeviceAdaptor::GetDevice(int32_t deviceId) {
    return AclAdaptor::GetInstance(deviceId);
}