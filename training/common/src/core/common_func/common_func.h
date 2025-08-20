//
// Created by ROME on 2025/8/20.
//

#ifndef RECSDK_REFACTORING_COMMON_FUNC_H
#define RECSDK_REFACTORING_COMMON_FUNC_H

#include <cstring>
#include <memory>
#include <sstream>
#include <dsmi_common_interface.h>
#include "log/logger.h"

namespace MxRec {
using namespace std;
constexpr int GLOG_MAX_BUF_SIZE = 1024;
const char* HUGE_TLB_ENABLE = "HUGE_TLB_ENABLE";

template <typename... Args>
string StringFormat(const string& format, Args... args)
{
    auto size = static_cast<size_t>(GLOG_MAX_BUF_SIZE);
    auto buf = std::make_unique<char[]>(size); // LCOV_EXCL_BR_LINE
    memset_s(buf.get(), size, 0, size);
    int nChar = snprintf_s(buf.get(), size, size - 1, format.c_str(), args...);
    if (nChar == -1) { // LCOV_EXCL_BR_LINE
        throw invalid_argument("StringFormat failed");
    }
    return string(buf.get(), buf.get() + nChar);
}

string GetChipName(int devID)
{
    int ret = 0;
    struct dsmi_chip_info_stru info = {{ 0 },
                                       { 0 },
                                       { 0 }};
    ret = dsmi_get_chip_info(devID, &info);
    if (ret == 0) {
        stringstream ss;
        ss << info.chip_name;
        LOG_DEBUG("dsmi_get_chip_info successful, ret = {}, chip_name = {}", ret, ss.str());
        return ss.str();
    }

    throw std::runtime_error("dsmi_get_chip_info failed, ret = " + to_string(ret));
}
}

#endif //RECSDK_REFACTORING_COMMON_FUNC_H
