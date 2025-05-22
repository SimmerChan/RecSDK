#ifndef DEVICE_ADAPTOR_H
#define DEVICE_ADAPTOR_H

#include "device_interface.h"
#include <cstdint>

namespace tensorflow {
namespace npu_xla {


class DeviceAdaptor {
public:
    static DeviceInterface& GetDevice(int32_t deviceId);
};

} // namespace npu_xla
} // namespace tensorflow

#endif // DEVICE_ADAPTOR_H