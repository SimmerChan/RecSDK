#ifndef COMMON_UTILS_H
#define COMMON_UTILS_H
#include <cstdint>
#include <cstdio>

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

template<typename T>
bool CheckPtrIsNull(T ptr, const char* errorMessage)
{
    if (ptr == nullptr) {
        printf("[ERROR] Failed to get %s!\n", errorMessage);
        return true;
    }
    return false;
}
#endif