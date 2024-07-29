#include <string>

#include "expected.h"

namespace MxRec {

enum class ErrorCode {
    // TODO add real error kinds
    kNotFound = 1,
    kFileNotExist = 2,
    kUnknown,
};

class Error {
public:
    Error() = delete;
    explicit Error(ErrorCode e) : e_(e), msg_() {}
    Error(ErrorCode e, const std::string& msg) : e_(e), msg_(msg) {}

    std::string ToString()
    {
        return "ERROR kind:" + this->CodeAsString() + ". Message: " + this->msg_;
    }

private:
    std::string CodeAsString()
    {
        switch (this->e_) {
            case ErrorCode::kNotFound:
                return "NotFound";
            case ErrorCode::kFileNotExist:
                return "FileNotExist";
            default:
                return "Unknown";
        }
    }

    ErrorCode e_;
    std::string msg_;
};

template <typename T>
using Result = tl::expected<T, Error>;

}  // namespace MxRec
