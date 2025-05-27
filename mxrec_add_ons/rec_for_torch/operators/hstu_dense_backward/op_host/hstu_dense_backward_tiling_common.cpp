#include <cctype>
#include <string>
#include <unordered_map>

#include "register/op_def_registry.h"

#include "hstu_dense_backward_tiling_common.h"

ShapeRange::ShapeRange(int64_t lbound, int64_t ubound, int64_t mutiple, const char *name)
{
    this->lbound = lbound;
    this->ubound = ubound;
    this->mutiple = mutiple;
    this->name = name;
}

bool ShapeRange::Check(int64_t val) const
{
    OPS_LOGD_IF((val < lbound || val > ubound || val % mutiple != 0),
        printf("%s must meet range[%lld %lld] and mutiple of [%lld]. but get value %lld\n",
            name, lbound, ubound, mutiple, val),
        return false);
    return true;
}

ge::graphStatus GetInputLayout(const gert::RuntimeAttrs *attrs, InputLayout &layout)
{
    OPS_LOGD_IF_NULL(attrs, return ge::GRAPH_FAILED);

    const char *inputLayout = attrs->GetAttrPointer<char>(INDEX_T::INDEX_0);
    OPS_LOGD_IF_NULL(inputLayout, return ge::GRAPH_FAILED);

    std::string inputLayoutStr = std::string(inputLayout);
    for (auto &c : inputLayoutStr) {
        c = tolower(c);
    }

    if (inputLayoutStr == "normal") {
        layout = InputLayout::NORMAL;
    } else if (inputLayoutStr == "jagged") {
        layout = InputLayout::JAGGED;
    } else {
        OPS_LOGD("the input layout should be normal/jagged, but get %s.\n", inputLayoutStr.c_str());
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

bool IfMask(const int32_t &maskType, MaskType maskTypeEnum)
{
    return static_cast<int32_t>(maskTypeEnum) == maskType;
}

bool IsSameShape(const gert::Shape &shape0, const gert::Shape &shape1, int dim)
{
    if (shape0.GetDimNum() != dim || shape1.GetDimNum() != dim) {
        return false;
    }

    for (int i = 0; i < dim; i++) {
        if (shape0.GetDim(i) != shape1.GetDim(i)) {
            return false;
        }
    }

    return true;
}

bool BasicShapeCheck(int64_t batchSize, int64_t seqLen, int64_t headNum, int64_t dim)
{
    static const ShapeRange batchRange(1, MAX_BATCH_SIZE, 1, "batch size");
    static const ShapeRange seqRange(1, 20480, 1, "seq size");
    static const ShapeRange headRange(2, 8, 2, "head num");
    static const ShapeRange dimRange(16, 512, 16, "dim size");

    if (!batchRange.Check(batchSize)) {
        return false;
    }

    if (!seqRange.Check(seqLen)) {
        return false;
    }

    if (!headRange.Check(headNum)) {
        return false;
    }

    if (!dimRange.Check(dim)) {
        return false;
    }

    return true;
}