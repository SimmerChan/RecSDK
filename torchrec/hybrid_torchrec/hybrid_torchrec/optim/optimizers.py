import torch

class AccumulateAdagrad(torch.optim.Adagrad):
    def __init__(self, params, use_accumulate=False, accumulate_step=1, **kwargs):
        super().__init__(params, **kwargs)
        self.use_accumulate = use_accumulate
        self.accumulate_step = accumulate_step

class AccumulateSGD(torch.optim.SGD):
    def __init__(self, params, use_accumulate=False, accumulate_step=1, **kwargs):
        super().__init__(params, **kwargs)
        self.use_accumulate = use_accumulate
        self.accumulate_step = accumulate_step


class AccumulateAdam(torch.optim.Adam):
    def __init__(self, params, use_accumulate=False, accumulate_step=1, **kwargs):
        super().__init__(params, **kwargs)
        self.use_accumulate = use_accumulate
        self.accumulate_step = accumulate_step