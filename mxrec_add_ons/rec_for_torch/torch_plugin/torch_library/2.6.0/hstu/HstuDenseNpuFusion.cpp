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

#include <string>
#include <algorithm>
#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>

#include "../common/pytorch_npu_helper.hpp"
using torch::autograd::AutogradContext;
using torch::autograd::Function;
using tensor_list = std::vector<at::Tensor>;
using namespace at;

constexpr size_t MIN_SEQ_LEN = 1;
constexpr size_t MAX_SEQ_LEN = 20480;
constexpr uint32_t MASK_TYPE_TRIL = 0;
constexpr uint32_t MASK_TYPE_TRIU = 1;
constexpr uint32_t MASK_TYPE_CUSTOM = 3;
constexpr uint32_t CONST_4 = 4;
constexpr uint32_t CONST_3 = 3;
constexpr uint32_t CONST_2 = 2;

namespace hstu {
bool MaskCheck(int64_t maskType, uint32_t maskIsDefine)
{
    if (maskType < MASK_TYPE_TRIL || maskType > MASK_TYPE_CUSTOM) {
        printf("maskType expect in [0, 3], but value is %d\n", maskType);
        return false;
    }

    if (maskType == MASK_TYPE_TRIU) {
        printf("maskType current not support triu now, pls use custome mask\n");
        return false;
    }

    if (maskType == MASK_TYPE_CUSTOM && !maskIsDefine) {
        printf("use custome mask must have valide mask tensor \n");
        return false;
    }
    return true;
}

at::Tensor hstu_dense_forward_impl_npu(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const c10::optional<at::Tensor>& mask,
    const c10::optional<at::Tensor>& attnBias,
    const int64_t enable_bias,
    const int64_t maskType,
    const int64_t maxSeqLen,
    const double siluScale,
    const std::string layout,
    const c10::optional<at::Tensor>& seqOffset)
{
    TORCH_CHECK(q.dim() == CONST_4, "The q should be 4D in dense layout");

    auto denseQ = q.contiguous();
    auto denseK = k.contiguous();
    auto denseV = v.contiguous();
    auto denseBias = c10::value_or_else(attnBias, [] {return at::Tensor(); });
    auto maskNpu = c10::value_or_else(mask, [] {return at::Tensor(); });

    TORCH_CHECK(maxSeqLen >= MIN_SEQ_LEN && maxSeqLen <= MAX_SEQ_LEN,
                "maxSeqLen expect in [1, 20480], but value is ", maxSeqLen);
    TORCH_CHECK(MaskCheck(maskType, maskNpu.defined()), "maskType check failed");

    auto attnOutput = at::empty_like(denseQ);
    double realSiluScale = (siluScale == 0.0) ? 1.0f / maxSeqLen : siluScale;

    const auto _acSeqOffsetK = at::Tensor();
    const auto _acseqOffsetT = at::Tensor();
    const auto _kvCacheNpu = at::Tensor();
    const auto _pageOffsets = at::Tensor();
    const auto _pageIds = at::Tensor();
    const auto _lastPageLen = at::Tensor();
    const auto _numContext = at::Tensor();
    const auto _numTarget = at::Tensor();
    const auto _maxSeqLenK = int();
    const auto _target_group_size = int();


    EXEC_NPU_CMD(aclnnHstuDenseForward,
                 denseQ,
                 denseK,
                 denseV,
                 maskNpu,
                 denseBias,
                 seqOffset,
                 _acSeqOffsetK,
                 _acseqOffsetT,
                 _kvCacheNpu,
                 _pageOffsets,
                 _pageIds,
                 _lastPageLen,
                 _numContext,
                 _numTarget,
                 maskType,
                 maxSeqLen,
                 realSiluScale,
                 layout,
                 _maxSeqLenK,
                 enable_bias,
                 _target_group_size,
                 attnOutput);
    return attnOutput;
}

at::Tensor hstu_jagged_forward_impl_npu(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const c10::optional<at::Tensor>& mask,
    const c10::optional<at::Tensor>& attnBias,
    const int64_t enable_bias,
    const int64_t maskType,
    const int64_t maxSeqLen,
    const double siluScale,
    const std::string layout,
    const c10::optional<at::Tensor>& seqOffset,
    const c10::optional<at::Tensor>& num_context,
    const c10::optional<at::Tensor>& num_target,
    const int64_t target_group_size)
{
    TORCH_CHECK(q.dim() == CONST_3, "The q should be 3D in jagged layout");

    auto acSeqOffset = c10::value_or_else(seqOffset, [] {return at::Tensor(); });
    TORCH_CHECK(acSeqOffset.size(0) >= CONST_2, "acSeqOffset params error should have at least two element.");

    auto denseQ = q.contiguous();
    auto denseK = k.contiguous();
    auto denseV = v.contiguous();
    auto denseBias = c10::value_or_else(attnBias, [] {return at::Tensor(); });
    auto maskNpu = c10::value_or_else(mask, [] {return at::Tensor(); });
    auto numContext = c10::value_or_else(num_context, [] {return at::Tensor(); });
    auto numTarget = c10::value_or_else(num_target, [] {return at::Tensor(); });

    TORCH_CHECK(maxSeqLen >= MIN_SEQ_LEN && maxSeqLen <= MAX_SEQ_LEN,
                "maxSeqLen expect in [1, 20480], but value is ", maxSeqLen);

    TORCH_CHECK(MaskCheck(maskType, maskNpu.defined()), "maskType check failed");

    auto attnOutput = at::empty_like(denseQ);
    double realSiluScale = (siluScale == 0.0) ? 1.0f / maxSeqLen : siluScale;

    const auto _acSeqOffsetK = at::Tensor();
    const auto _acseqOffsetT = at::Tensor();
    const auto _kvCacheNpu = at::Tensor();
    const auto _pageOffsets = at::Tensor();
    const auto _pageIds = at::Tensor();
    const auto _lastPageLen = at::Tensor();
    const auto _maxSeqLenK = int();

    EXEC_NPU_CMD(aclnnHstuDenseForward,
                 denseQ,
                 denseK,
                 denseV,
                 maskNpu,
                 denseBias,
                 acSeqOffset,
                 _acSeqOffsetK,
                 _acseqOffsetT,
                 _kvCacheNpu,
                 _pageOffsets,
                 _pageIds,
                 _lastPageLen,
                 numContext,
                 numTarget,
                 maskType,
                 maxSeqLen,
                 realSiluScale,
                 layout,
                 _maxSeqLenK,
                 enable_bias,
                 target_group_size,
                 attnOutput);
    return attnOutput;
}

at::Tensor hstu_varlen_forward_impl_npu(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const c10::optional<at::Tensor>& mask,
    const c10::optional<at::Tensor>& attnBias,
    const int64_t enable_bias,
    const int64_t maskType,
    const int64_t maxSeqLen,
    const int64_t maxSeqLenK,
    const double siluScale,
    const std::string layout,
    const c10::optional<at::Tensor>& seqOffset,
    const c10::optional<at::Tensor>& seqOffsetK)
{
    TORCH_CHECK(q.dim() == CONST_3, "The q should be 3D in jagged layout");

    auto acSeqOffset = c10::value_or_else(seqOffset, [] {return at::Tensor(); });
    auto acSeqOffsetK = c10::value_or_else(seqOffsetK, [] {return at::Tensor(); });
    TORCH_CHECK(acSeqOffset.size(0) >= CONST_2, "acSeqOffset params error should have at least two element.");
    TORCH_CHECK(acSeqOffsetK.size(0) >= CONST_2, "acSeqOffsetK params error should have at least two element.");

    auto denseQ = q.contiguous();
    auto denseK = k.contiguous();
    auto denseV = v.contiguous();
    auto denseBias = c10::value_or_else(attnBias, [] {return at::Tensor(); });
    auto maskNpu = c10::value_or_else(mask, [] {return at::Tensor(); });

    TORCH_CHECK(maxSeqLen >= MIN_SEQ_LEN && maxSeqLen <= MAX_SEQ_LEN,
                "maxSeqLen expect in [1, 20480], but value is ", maxSeqLen);
    TORCH_CHECK(MaskCheck(maskType, maskNpu.defined()), "maskType check failed");

    auto attnOutput = at::empty_like(denseQ);
    double realSiluScale = (siluScale == 0.0) ? 1.0f / maxSeqLen : siluScale;

    const auto _acseqOffsetT = at::Tensor();
    const auto _kvCacheNpu = at::Tensor();
    const auto _pageOffsets = at::Tensor();
    const auto _pageIds = at::Tensor();
    const auto _lastPageLen = at::Tensor();
    const auto _numContext = at::Tensor();
    const auto _numTarget = at::Tensor();
    const auto _target_group_size = int();

    EXEC_NPU_CMD(aclnnHstuDenseForward,
                 denseQ,
                 denseK,
                 denseV,
                 maskNpu,
                 denseBias,
                 acSeqOffset,
                 acSeqOffsetK,
                 _acseqOffsetT,
                 _kvCacheNpu,
                 _pageOffsets,
                 _pageIds,
                 _lastPageLen,
                 _numContext,
                 _numTarget,
                 maskType,
                 maxSeqLen,
                 realSiluScale,
                 layout,
                 maxSeqLenK,
                 enable_bias,
                 _target_group_size,
                 attnOutput);
    return attnOutput;
}

at::Tensor hstu_paged_forward_impl_npu(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const c10::optional<at::Tensor>& kv_cache,
    const c10::optional<at::Tensor>& mask,
    const c10::optional<at::Tensor>& attnBias,
    const int64_t enable_bias,
    const int64_t maskType,
    const int64_t maxSeqLen,
    const int64_t maxSeqLenK,
    const double siluScale,
    const std::string layout,
    const c10::optional<at::Tensor>& seqOffset,
    const c10::optional<at::Tensor>& seqOffsetK,
    const c10::optional<at::Tensor>& seqOffsetT,
    const c10::optional<at::Tensor>& Page_offsets,
    const c10::optional<at::Tensor>& Page_ids,
    const c10::optional<at::Tensor>& Last_page_len)
{
    TORCH_CHECK(q.dim() == CONST_3, "The q should be 3D in jagged layout");

    auto acSeqOffset = c10::value_or_else(seqOffset, [] {return at::Tensor(); });
    auto acSeqOffsetK = c10::value_or_else(seqOffsetK, [] {return at::Tensor(); });
    TORCH_CHECK(acSeqOffset.size(0) >= CONST_2, "acSeqOffset params error should have at least two element.");
    TORCH_CHECK(acSeqOffsetK.size(0) >= CONST_2, "acSeqOffsetK params error should have at least two element.");

    auto denseQ = q.contiguous();
    auto denseK = k.contiguous();
    auto denseV = v.contiguous();
    auto denseBias = c10::value_or_else(attnBias, [] {return at::Tensor(); });
    auto maskNpu = c10::value_or_else(mask, [] {return at::Tensor(); });
    auto acseqOffsetT = c10::value_or_else(seqOffsetT, [] {return at::Tensor(); });
    auto pageOffsets = c10::value_or_else(Page_offsets, [] {return at::Tensor(); });
    auto pageIds = c10::value_or_else(Page_ids, [] {return at::Tensor(); });
    auto lastPageLen = c10::value_or_else(Last_page_len, [] {return at::Tensor(); });
    auto kvCacheNpu = c10::value_or_else(kv_cache, [] {return at::Tensor(); });

    TORCH_CHECK(maxSeqLen >= MIN_SEQ_LEN && maxSeqLen <= MAX_SEQ_LEN,
                "maxSeqLen expect in [1, 20480], but value is ", maxSeqLen);
    TORCH_CHECK(MaskCheck(maskType, maskNpu.defined()), "maskType check failed");

    auto attnOutput = at::empty_like(denseQ);
    double realSiluScale = (siluScale == 0.0) ? 1.0f / maxSeqLen : siluScale;

    const auto _numContext = at::Tensor();
    const auto _numTarget = at::Tensor();
    const auto _target_group_size = int();

    EXEC_NPU_CMD(aclnnHstuDenseForward,
                 denseQ,
                 denseK,
                 denseV,
                 maskNpu,
                 denseBias,
                 acSeqOffset,
                 acSeqOffsetK,
                 acseqOffsetT,
                 kvCacheNpu,
                 pageOffsets,
                 pageIds,
                 lastPageLen,
                 _numContext,
                 _numTarget,
                 maskType,
                 maxSeqLen,
                 realSiluScale,
                 layout,
                 maxSeqLenK,
                 enable_bias,
                 _target_group_size,
                 attnOutput);
    return attnOutput;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> hstu_dense_backward_impl_npu(
    const at::Tensor& grad,
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const c10::optional<at::Tensor> mask,
    const c10::optional<at::Tensor> attnBias,
    const int64_t enable_bias,
    const int64_t maskType,
    const int64_t maxSeqLen,
    const double siluScale,
    const std::string layout,
    const c10::optional<at::Tensor>& seqOffset)
{
    constexpr int dim = 4;
    TORCH_CHECK(grad.dim() == dim, "The grad should be 4D in normal layout");

    auto acAttnBias = attnBias.value_or(at::Tensor());
    auto acMask = mask.value_or(at::Tensor());

    auto denseGrad = grad.contiguous();
    auto denseQ = q.contiguous();
    auto denseK = k.contiguous();
    auto denseV = v.contiguous();
    auto denseAttnBias = acAttnBias.contiguous();
    auto denseMask = acMask.contiguous();


    uint32_t batchSize = denseGrad.size(0); // 0 means index 0
    uint32_t seqLen = denseGrad.size(1); // 1 means index 1
    uint32_t headNum = denseGrad.size(2); // 2 means index 2
    uint32_t headDim = denseGrad.size(3); // 3 means index 3

    TORCH_CHECK(seqLen >= MIN_SEQ_LEN && seqLen <= MAX_SEQ_LEN, "seqLen expect in [1, 20480], but value is ", seqLen);
    TORCH_CHECK(seqLen == maxSeqLen, "seqLen must be equal to maxSeqLen");

    double realSiluScale = (siluScale == 0.0) ? 1.0f / maxSeqLen : siluScale;

    auto qGradOutput = at::empty_like(denseQ);
    auto kGradOutput = at::empty_like(denseK);
    auto vGradOutput = at::empty_like(denseV);

    at::Tensor attnBiasGradOutput;
    if (denseAttnBias.defined()) {
        attnBiasGradOutput = at::empty_like(denseAttnBias);
    } else {
        auto biasGradSeqLen = (seqLen + 256 - 1) / 256 * 256; // get 256 bit aligned biasGrad space
        attnBiasGradOutput = at::empty({batchSize, headNum, biasGradSeqLen, biasGradSeqLen},
                                       at::device(denseGrad.device()).dtype(denseGrad.dtype()));
    }

    auto _denseNum_context = at::Tensor();
    auto _denseNum_target = at::Tensor();
    auto _target_group_size = int();

    EXEC_NPU_CMD(aclnnHstuDenseBackward,
                 denseGrad,
                 denseQ,
                 denseK,
                 denseV,
                 denseMask,
                 denseAttnBias,
                 _denseNum_context,
                 _denseNum_target,
                 layout,
                 maskType,
                 maxSeqLen,
                 realSiluScale,
                 seqOffset,
                 enable_bias,
                 _target_group_size,
                 qGradOutput,
                 kGradOutput,
                 vGradOutput,
                 attnBiasGradOutput);

    return std::make_tuple(qGradOutput, kGradOutput, vGradOutput, attnBiasGradOutput);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> hstu_jagged_backward_impl_npu(
    const at::Tensor& grad,
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const c10::optional<at::Tensor> mask,
    const c10::optional<at::Tensor> attnBias,
    const int64_t enable_bias,
    const int64_t maskType,
    const int64_t maxSeqLen,
    const double siluScale,
    const std::string layout,
    const c10::optional<at::Tensor>& seqOffset,
    const c10::optional<at::Tensor>& num_context,
    const c10::optional<at::Tensor>& num_target,
    const int64_t target_group_size)
{
    constexpr int dim = 3;
    TORCH_CHECK(grad.dim() == dim, "The grad should be 3D in jagged layout");

    auto acSeqOffset = seqOffset.value_or(at::Tensor());
    TORCH_CHECK(acSeqOffset.size(0) >= CONST_2, "acSeqOffset params error should have at least two element.");

    auto acAttnBias = attnBias.value_or(at::Tensor());
    auto acMask = mask.value_or(at::Tensor());
    auto acNum_context = num_context.value_or(at::Tensor());
    auto acNum_target = num_target.value_or(at::Tensor());

    auto denseGrad = grad.contiguous();
    auto denseQ = q.contiguous();
    auto denseK = k.contiguous();
    auto denseV = v.contiguous();
    auto denseAttnBias = acAttnBias.contiguous();
    auto denseMask = acMask.contiguous();
    auto denseNum_context = acNum_context.contiguous();
    auto denseNum_target = acNum_target.contiguous();
    //
    uint32_t batchSize = acSeqOffset.size(0) - 1;
    uint32_t headNum = denseGrad.size(1); // 1 means index 1
    uint32_t headDim = denseGrad.size(2); // 2 means index 2

    TORCH_CHECK(maxSeqLen >= MIN_SEQ_LEN && maxSeqLen <= MAX_SEQ_LEN,
                "maxSeqLen expect in [1, 20480], but value is ", maxSeqLen);

    if (static_cast<uint32_t>(maskType) == MASK_TYPE_CUSTOM) {
        TORCH_CHECK(denseMask.defined(), "use maskType:MASK_CUSTOM, but no mask given\n");
        // mask dim 2 must be equalto maxSeqLen
        TORCH_CHECK(denseMask.size(2) == maxSeqLen, "mask size 2 should be equal to maxSeqLen\n");
    }

    double realSiluScale = (siluScale == 0.0) ? 1.0f / maxSeqLen : siluScale;

    auto qGradOutput = at::empty_like(denseQ);
    auto kGradOutput = at::empty_like(denseK);
    auto vGradOutput = at::empty_like(denseV);

    at::Tensor attnBiasGradOutput;
    if (denseAttnBias.defined()) {
        attnBiasGradOutput = at::empty_like(denseAttnBias);
    } else {
        attnBiasGradOutput = at::Tensor();
    }

    EXEC_NPU_CMD(aclnnHstuDenseBackward,
                 denseGrad,
                 denseQ,
                 denseK,
                 denseV,
                 denseMask,
                 denseAttnBias,
                 denseNum_context,
                 denseNum_target,
                 layout,
                 maskType,
                 maxSeqLen,
                 realSiluScale,
                 seqOffset,
                 enable_bias,
                 target_group_size,
                 qGradOutput,
                 kGradOutput,
                 vGradOutput,
                 attnBiasGradOutput);

    if (denseAttnBias.defined()) {
        return std::make_tuple(qGradOutput, kGradOutput, vGradOutput, attnBiasGradOutput);
    } else {
        return std::make_tuple(qGradOutput, kGradOutput, vGradOutput, at::Tensor());
    }
}

//注册算子到mxrec
TORCH_LIBRARY_FRAGMENT(mxrec, m)
{
    m.def("hstu_dense(Tensor q, Tensor k, Tensor v, Tensor? mask=None, Tensor? attnBias=None, int enable_bias=1, \
           int maskType=0, int maxSeqLen=0, float siluScale=0.0, str layout=\"normal\", int[]? seqOffset=None) -> Tensor");
    m.def("hstu_jagged(Tensor q, Tensor k, Tensor v, Tensor? mask=None, Tensor? attnBias=None, int enable_bias=1, \
           int maskType=0, int maxSeqLen=0, float siluScale=0.0, str layout=\"jagged\", int[]? seqOffset=None, \
           Tensor? num_context=None, Tensor? num_target=None, int target_group_size=0) -> Tensor");
    m.def("hstu_varlen(Tensor q, Tensor k, Tensor v, Tensor? mask=None, Tensor? attnBias=None, int enable_bias=1, \
           int maskType=0, int maxSeqLen=0, int maxSeqLenK=0, float siluScale=0.0, str layout=\"jagged\", int[]? seqOffset=None, \
           int[]? seqOffsetK=None) -> Tensor");
    m.def("hstu_paged(Tensor q, Tensor k, Tensor v, Tensor? kv_cahce=None, Tensor? mask=None, Tensor? attnBias=None, int enable_bias=1, \
           int maskType=0, int maxSeqLen=0, int maxSeqLenK=0, float siluScale=0.0, str layout=\"jagged\", int[]? seqOffset=None, \
           int[]? seqOffsetK=None, Tensor? seqOffsetT=None, Tensor? Page_offsets=None, Tensor? Page_ids=None, \
           Tensor? Last_page_len=None) -> Tensor");
    m.def("hstu_dense_backward(Tensor grad, Tensor q, Tensor k, Tensor v, Tensor? mask, Tensor? attnBias, int enable_bias=1, \
           int maskType, int maxSeqLen, float siluScale=0.0, str layout=\"normal\", int[]? seqOffset=None) -> (Tensor, Tensor, \
           Tensor, Tensor)");
    m.def("hstu_jagged_backward(Tensor grad, Tensor q, Tensor k, Tensor v, Tensor? mask=None, Tensor? attnBias=None, int enable_bias=1, \
           int maskType=0, int maxSeqLen=0, float siluScale=0.0, str layout=\"jagged\", int[]? seqOffset=None, Tensor? num_context=None, \
           Tensor? num_target=None, int target_group_size=0) -> (Tensor, Tensor, Tensor, Tensor)");
}
//将算子名称映射到具体实现
TORCH_LIBRARY_IMPL(mxrec, PrivateUse1, m)
{
    m.impl("hstu_dense", &hstu_dense_forward_impl_npu);
    m.impl("hstu_jagged", &hstu_jagged_forward_impl_npu);
    m.impl("hstu_varlen", &hstu_varlen_forward_impl_npu);
    m.impl("hstu_paged", &hstu_paged_forward_impl_npu);
    m.impl("hstu_dense_backward", &hstu_dense_backward_impl_npu);
    m.impl("hstu_jagged_backward", &hstu_jagged_backward_impl_npu);
}

////////autograd实现
// 通过继承torch::autograd::Function类实现前反向绑定
class HstuDenseNpuFusion : public torch::autograd::Function<HstuDenseNpuFusion> {
public:
    static at::Tensor forward(AutogradContext *ctx,
                              const at::Tensor& q,
                              const at::Tensor& k,
                              const at::Tensor& v,
                              const c10::optional<at::Tensor>& mask,
                              const c10::optional<at::Tensor>& attnBias,
                              const int64_t enable_bias,
                              const int64_t maskType,
                              const int64_t maxSeqLen,
                              const double siluScale,
                              const std::string layout,
                              const c10::optional<at::Tensor> seqOffset)
    {
        at::AutoDispatchBelowADInplaceOrView guard;

        ctx->save_for_backward({ q, k, v, mask.value_or(at::Tensor()), attnBias.value_or(at::Tensor()), seqOffset.value_or(at::Tensor()) });
        ctx->saved_data["enable_bias"] = enable_bias;
        ctx->saved_data["maskType"] = maskType;
        ctx->saved_data["maxSeqLen"] = maxSeqLen;
        ctx->saved_data["siluScale"] = siluScale;
        ctx->saved_data["layout"] = layout;

        if (seqOffset.has_value()) {
            //auto seqOffsetVec = seqOffset->vec();
            //save_for_backward储存tensor
            ctx->saved_data["hasSeqOffset"] = true;
        } else {
            ctx->saved_data["hasSeqOffset"] = false;
        }

        return hstu_dense_forward_impl_npu(q, k, v, mask, attnBias, enable_bias, maskType,
                                           maxSeqLen, siluScale, layout, seqOffset);
    }

    static tensor_list backward(AutogradContext *ctx, tensor_list grad_outputs)
    {
        auto grad = grad_outputs[0];

        auto saved = ctx->get_saved_variables();
        auto q = saved[0];
        auto k = saved[1];
        auto v = saved[2];
        auto mask = saved[3];
        auto attnBias = saved[4];
        auto seqOffset = saved[5];

        auto enable_bias = ctx->saved_data["enable_bias"].toInt();
        auto maskType = ctx->saved_data["maskType"].toInt();
        auto maxSeqLen = ctx->saved_data["maxSeqLen"].toInt();
        auto siluScale = ctx->saved_data["siluScale"].toDouble();
        auto layout = ctx->saved_data["layout"].toStringRef();

        bool hasSeqOffset = ctx->saved_data["hasSeqOffset"].toBool();
        //        std::vector<int64_t> seqOffsetVec;
        //        c10::optional<at::Tensor> seqOffset;
        //        if (hasSeqOffset) {
        ////            ctx->save_for_backward["seqOffset"] = seqOffset;
        //            seqOffsetVec = ctx->saved_data["seqOffset"].toIntVector();
        //            seqOffset = at::Tensor(seqOffsetVec);
        //            seqOffset = ctx->save_for_backward["seqOffset"]
        //        }

        auto resultTuple = hstu_dense_backward_impl_npu(grad, q, k, v, mask, attnBias, enable_bias, maskType,
                                                        maxSeqLen, siluScale, layout, seqOffset);

        if (attnBias.defined()) {
            //返回q, k, v, mask, attnBias, enable_bias, maskType, maxSeqLen, siluScale, layout, seqOffset的梯度
            return { std::get<0>(resultTuple), std::get<1>(resultTuple), std::get<2>(resultTuple), at::Tensor(),
                    std::get<3>(resultTuple), at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(),
                    at::Tensor() };
        } else {
            return { std::get<0>(resultTuple), std::get<1>(resultTuple), std::get<2>(resultTuple), at::Tensor(),
                    at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(),
                    at::Tensor() };
        }
    }
};

class HstuJaggedNpuFusion : public torch::autograd::Function<HstuJaggedNpuFusion> {
public:
    static at::Tensor forward(AutogradContext *ctx,
                              const at::Tensor& q,
                              const at::Tensor& k,
                              const at::Tensor& v,
                              const c10::optional<at::Tensor>& mask,
                              const c10::optional<at::Tensor>& attnBias,
                              const int64_t enable_bias,
                              const int64_t maskType,
                              const int64_t maxSeqLen,
                              const double siluScale,
                              const std::string layout,
                              const c10::optional<at::Tensor> seqOffset,
                              const c10::optional<at::Tensor>& num_context,
                              const c10::optional<at::Tensor>& num_target,
                              const int64_t target_group_size)
    {
        at::AutoDispatchBelowADInplaceOrView guard;

        ctx->save_for_backward({ q, k, v, mask.value_or(at::Tensor()), attnBias.value_or(at::Tensor()),
                                num_context.value_or(at::Tensor()), num_target.value_or(at::Tensor()), seqOffset.value_or(at::Tensor()) });

        ctx->saved_data["enable_bias"] = enable_bias;
        ctx->saved_data["maskType"] = maskType;
        ctx->saved_data["maxSeqLen"] = maxSeqLen;
        ctx->saved_data["siluScale"] = siluScale;
        ctx->saved_data["layout"] = layout;
        ctx->saved_data["target_group_size"] = target_group_size;

        if (seqOffset.has_value()) {
            //            auto seqOffsetVec = seqOffset->vec();
            //            ctx->saved_data["seqOffset"] = seqOffsetVec;
            ctx->saved_data["hasSeqOffset"] = true;
        } else {
            ctx->saved_data["hasSeqOffset"] = false;
        }

        return hstu_jagged_forward_impl_npu(q, k, v, mask, attnBias, enable_bias, maskType,
                                            maxSeqLen, siluScale, layout, seqOffset,
                                            num_context, num_target, target_group_size);
    }

    static tensor_list backward(AutogradContext *ctx, tensor_list grad_outputs)
    {
        auto grad = grad_outputs[0];

        auto saved = ctx->get_saved_variables();
        auto q = saved[0];
        auto k = saved[1];
        auto v = saved[2];
        auto mask = saved[3];
        auto attnBias = saved[4];
        auto num_context = saved[5];
        auto num_target = saved[6];
        auto seqOffset = saved[7];

        auto enable_bias = ctx->saved_data["enable_bias"].toInt();
        auto maskType = ctx->saved_data["maskType"].toInt();
        auto maxSeqLen = ctx->saved_data["maxSeqLen"].toInt();
        auto siluScale = ctx->saved_data["siluScale"].toDouble();
        auto layout = ctx->saved_data["layout"].toStringRef();
        auto target_group_size = ctx->saved_data["target_group_size"].toInt();

        bool hasSeqOffset = ctx->saved_data["hasSeqOffset"].toBool();
        //        std::vector<int64_t> seqOffsetVec;
        //        c10::optional<at::Tensor> seqOffset;
        //        if (hasSeqOffset) {
        //            seqOffsetVec = ctx->saved_data["seqOffset"].toIntVector();
        //            seqOffset = at::Tensor(seqOffsetVec);
        //        }

        auto resultTuple = hstu_jagged_backward_impl_npu(grad, q, k, v, mask, attnBias, enable_bias, maskType,
                                                         maxSeqLen, siluScale, layout, seqOffset,
                                                         num_context, num_target, target_group_size);

        // 返回梯度数量必须与前向输入参数数量一致
        if (attnBias.defined()) {
            return { std::get<0>(resultTuple), std::get<1>(resultTuple), std::get<2>(resultTuple), at::Tensor(),
                    std::get<3>(resultTuple), at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(),
                    at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor() };
        } else {
            return { std::get<0>(resultTuple), std::get<1>(resultTuple), std::get<2>(resultTuple), at::Tensor(),
                    at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(),
                    at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor() };
        }
    }
};

// 将HstuDenseNpuFusion封装成hstu_dense_autograd，方便调用
at::Tensor hstu_dense_autograd(const at::Tensor& q,
                               const at::Tensor& k,
                               const at::Tensor& v,
                               const c10::optional<at::Tensor>& mask,
                               const c10::optional<at::Tensor>& attnBias,
                               const int64_t enable_bias,
                               const int64_t maskType,
                               const int64_t maxSeqLen,
                               const double siluScale,
                               const std::string layout,
                               c10::optional<at::Tensor> seqOffset)
{
    return HstuDenseNpuFusion::apply(q, k, v, mask, attnBias, enable_bias, maskType,
                                     maxSeqLen, siluScale, layout, seqOffset);
}

// 将HstuJaggedNpuFusion封装成hstu_jagged_autograd，方便调用
at::Tensor hstu_jagged_autograd(const at::Tensor& q,
                                const at::Tensor& k,
                                const at::Tensor& v,
                                const c10::optional<at::Tensor>& mask,
                                const c10::optional<at::Tensor>& attnBias,
                                const int64_t enable_bias,
                                const int64_t maskType,
                                const int64_t maxSeqLen,
                                const double siluScale,
                                const std::string layout,
                                c10::optional<at::Tensor> seqOffset,
                                const c10::optional<at::Tensor>& num_context,
                                const c10::optional<at::Tensor>& num_target,
                                const int64_t target_group_size)
{
    return HstuJaggedNpuFusion::apply(q, k, v, mask, attnBias, enable_bias, maskType,
                                      maxSeqLen, siluScale, layout, seqOffset,
                                      num_context, num_target, target_group_size);
}
//将算子入mxrec库，在test里调用
//将hstu_dense算子覆盖为具有自动求导功能的hstu_dense算子,不影响hstu_dense的单向使用
TORCH_LIBRARY_IMPL(mxrec, PrivateUse1, m)
{
    m.impl("hstu_dense", &hstu_dense_autograd);
    m.impl("hstu_jagged", &hstu_jagged_autograd);
}
}