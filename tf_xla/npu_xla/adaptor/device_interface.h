#ifndef DEVICE_INTERFACE_H
#define DEVICE_INTERFACE_H

#include <cstdint>
#include <cstddef>

namespace tensorflow {
namespace npu_xla {

class DeviceInterface {
public:
    virtual ~DeviceInterface() = default;

    virtual void* Allocate(size_t size) = 0;

    virtual void Deallocate(void* ptr) = 0;

    virtual bool MemcpyHToD(void* dst, size_t dstSize, const void* src, size_t srcSize) = 0;

    virtual bool MemcpyDToH(void* dst, size_t dstSize, const void* src, size_t srcSize) = 0;
};

} // namespace npu_xla
} // namespace tensorflow

#endif // DEVICE_INTERFACE_H