from typing import Any, Dict, Iterable, Type
import torch

def get_embedding_optim_num(optimizer_class: Type[torch.optim.Optimizer]) -> int:
    optim_cls_2_optim_num = {
        torch.optim.Adagrad: 1, 
        torch.optim.Adam: 2
        }
    return optim_cls_2_optim_num.get(optimizer_class, 0)