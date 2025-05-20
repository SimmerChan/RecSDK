#ifndef HYBRID_UNIQUE_H
#define HYBRID_UNIQUE_H
#include <omp.h>
#include <torch/torch.h>

#include "hashmap_dynamic_load.h"

namespace hybrid {
std::tuple<at::Tensor, at::Tensor> UniqueParallel(const at::Tensor& ids);
}  // namespace hybrid
#endif