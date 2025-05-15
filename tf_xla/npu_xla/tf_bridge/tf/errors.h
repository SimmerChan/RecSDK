/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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
// Copyright 2021 The BladeDISC Authors. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef NPU_XLA_TF_BRIDGE_TF_ERRORS_H_
#define NPU_XLA_TF_BRIDGE_TF_ERRORS_H_

#include "absl/base/log_severity.h"
#include "absl/status/statusor.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {
namespace npu_xla {

template <typename T>
using StatusOr = ::absl::StatusOr<T>;

template <typename T>
inline bool condition_to_bool(const T& condition)
{
    return condition;
}

template <>
inline bool condition_to_bool(const Status& status)
{
    return status.ok();
}

// Stream object used to collect error messages in MAKE_ERROR macros
// or append error messages with APPEND_ERROR.  It accepts any
// arguments with operator<< to build an error string, and then has an
// implicit cast operator to Status, which converts the
// logged string to a Status object and returns it, after logging the
// error.  At least one call to operator<< is required; a compile time
// error will be generated if none are given. Errors will only be
// logged by default for certain status codes, as defined in
// IsLoggedByDefault. This class will give ERROR errors if you don't
// retrieve a Status exactly once before destruction.
//
// The class converts into an intermediate wrapper object
// MakeErrorStreamWithOutput to check that the error stream gets at least one
// item of input.
class MakeErrorStream {
public:
    // Wrapper around MakeErrorStream that only allows for output. This
    // is created as output of the first operator<< call on
    // MakeErrorStream. The bare MakeErrorStream does not have a
    // Status operator. The net effect of that is that you
    // have to call operator<< at least once or else you'll get a
    // compile time error.
    class MakeErrorStreamWithOutput {
    public:
        explicit MakeErrorStreamWithOutput(MakeErrorStream* error_stream) : wrapped_error_stream_(error_stream) {}

        template <typename T>
        MakeErrorStreamWithOutput& operator<<(const T& value)
        {
            *wrapped_error_stream_ << value;
            return *this;
        }

        // Implicit cast operators to Status and StatusOr.
        // Exactly one of these must be called exactly once before destruction.
        operator Status()
        {
            return wrapped_error_stream_->GetStatus();
        }
        template <typename T>
        operator StatusOr<T>()
        {
            return wrapped_error_stream_->GetStatus();
        }

    private:
        MakeErrorStream* wrapped_error_stream_;

        TF_DISALLOW_COPY_AND_ASSIGN(MakeErrorStreamWithOutput);
    };

    // When starting from an existing error status, this determines whether we'll
    // append or prepend to that status's error message.
    enum PriorMessageHandling {
        kAppendToPriorMessage,
        kPrependToPriorMessage
    };

    // Make an error with the given code.
    template <typename ERROR_CODE_TYPE>
    MakeErrorStream(const char* file, int line, ERROR_CODE_TYPE code) : impl_(new Impl(file, line, code, this, true))
    {
    }

    template <typename T>
    MakeErrorStreamWithOutput& operator<<(const T& value)
    {
        CheckNotDone();
        impl_->stream_ << value;
        return impl_->make_error_stream_with_output_wrapper_;
    }

    // When this message is logged (see with_logging()), include the stack trace.
    MakeErrorStream& with_log_stack_trace()
    {
        impl_->should_log_stack_trace_ = true;
        return *this;
    }

    // Adds RET_CHECK failure text to error message.
    MakeErrorStreamWithOutput& add_ret_check_failure(const char* condition)
    {
        return *this << "RET_CHECK failure (" << impl_->file_ << ":" << impl_->line_ << ") " << condition << " ";
    }

private:
    class Impl {
    public:
        Impl(const char* file, int line, tsl::error::Code code, MakeErrorStream* error_stream,
             bool is_logged_by_default = true);
        Impl(const absl::Status& status, PriorMessageHandling prior_message_handling, const char* file, int line,
             MakeErrorStream* error_stream);

        ~Impl();

        // This must be called exactly once before destruction.
        Status GetStatus();

        void CheckNotDone() const;

    private:
        const char* file_;
        int line_;
        absl::StatusCode code_;

        PriorMessageHandling prior_message_handling_ = kAppendToPriorMessage;
        string prior_message_;
        bool is_done_;  // true after Status object has been returned
        std::ostringstream stream_;
        bool should_log_;
        absl::LogSeverity log_severity_;
        bool should_log_stack_trace_;

        // Wrapper around the MakeErrorStream object that has a
        // Status conversion. The first << operator called on
        // MakeErrorStream will return this object, and only this object
        // can implicitly convert to Status. The net effect of
        // this is that you'll get a compile time error if you call
        // MAKE_ERROR etc. without adding any output.
        MakeErrorStreamWithOutput make_error_stream_with_output_wrapper_;

        friend class MakeErrorStream;
        TF_DISALLOW_COPY_AND_ASSIGN(Impl);
    };

    void CheckNotDone() const;

    // Returns the status. Used by MakeErrorStreamWithOutput.
    Status GetStatus() const
    {
        return impl_->GetStatus();
    }

    // Store the actual data on the heap to reduce stack frame sizes.
    std::unique_ptr<Impl> impl_;

    TF_DISALLOW_COPY_AND_ASSIGN(MakeErrorStream);
};

// Provides a conversion to bool so that it can be used inside an if statement
// that declares a variable.
class StatusAdaptorForMacros {
public:
    explicit StatusAdaptorForMacros(Status status) : status_(std::move(status)) {}

    StatusAdaptorForMacros(const StatusAdaptorForMacros&) = delete;
    StatusAdaptorForMacros& operator=(const StatusAdaptorForMacros&) = delete;

    explicit operator bool() const
    {
        return TF_PREDICT_TRUE(status_.ok());
    }

    Status&& Consume()
    {
        return std::move(status_);
    }

private:
    Status status_;
};

}  // namespace npu_xla
}  // namespace tensorflow

#define TF_RET_CHECK(condition)                                                                  \
    while (TF_PREDICT_FALSE(!tensorflow::npu_xla::condition_to_bool(condition)))                 \
    return tensorflow::npu_xla::MakeErrorStream(__FILE__, __LINE__, tensorflow::error::INTERNAL) \
        .with_log_stack_trace()                                                                  \
        .add_ret_check_failure(#condition)

#define TF_STATUS_MACROS_CONCAT_NAME(x, y) TF_STATUS_MACROS_CONCAT_IMPL(x, y)
#define TF_STATUS_MACROS_CONCAT_IMPL(x, y) x##y

#define CHECK_OK(expr)                                 \
    do {                                               \
        auto __status = (expr);                        \
        if (TF_PREDICT_FALSE(!__status.ok())) {        \
            LOG(FATAL) << "Status not OK: " #expr " "; \
        }                                              \
    } while (0)

#endif  // NPU_XLA_TF_BRIDGE_TF_ERRORS_H_