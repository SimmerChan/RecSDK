from torchrec import optim as trec_optim
from hybrid_torchrec import optim as hybrid_optim
import torch
import logging
from typing import Any, Dict, List, Optional, Set, Tuple, Type, TypeVar, Union
from fbgemm_gpu.split_embedding_configs import EmbOptimType

def hybrid_optimizer_type_to_emb_opt_type(
        optimizer_class: Type[torch.optim.Optimizer],
) -> Optional[EmbOptimType]:
    # TODO add more optimizers to be in parity with ones provided by FBGEMM
    # TODO kwargs accepted by fbgemm and and canonical optimizers are different
    # may need to add special handling for them
    lookup = {
        torch.optim.SGD: EmbOptimType.EXACT_SGD,
        torch.optim.Adagrad: EmbOptimType.EXACT_ADAGRAD,
        torch.optim.Adam: EmbOptimType.ADAM,
        # below are torchrec wrappers over these optims.
        # they accept an **unused kwargs portion, that let us set FBGEMM specific args such as
        # max gradient, etc
        trec_optim.SGD: EmbOptimType.EXACT_SGD,
        trec_optim.LarsSGD: EmbOptimType.LARS_SGD,
        trec_optim.LAMB: EmbOptimType.LAMB,
        trec_optim.PartialRowWiseLAMB: EmbOptimType.PARTIAL_ROWWISE_LAMB,
        trec_optim.Adam: EmbOptimType.ADAM,
        trec_optim.PartialRowWiseAdam: EmbOptimType.PARTIAL_ROWWISE_ADAM,
        trec_optim.Adagrad: EmbOptimType.EXACT_ADAGRAD,
        trec_optim.RowWiseAdagrad: EmbOptimType.EXACT_ROWWISE_ADAGRAD,
        hybrid_optim.AccumulateAdagrad: EmbOptimType.EXACT_ADAGRAD,
        hybrid_optim.AccumulateAdam: EmbOptimType.ADAM,
        hybrid_optim.AccumulateSGD: EmbOptimType.EXACT_SGD,
    }
    if optimizer_class not in lookup:
        raise ValueError(f"Cannot cast {optimizer_class} to an EmbOptimType")
    return lookup[optimizer_class]