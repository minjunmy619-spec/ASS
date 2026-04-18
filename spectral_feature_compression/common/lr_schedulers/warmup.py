# Copyright (c) 2026 National Institute of Advanced Industrial Science and Technology (AIST), Japan
#
# SPDX-License-Identifier: MIT

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


class WarmUp(LambdaLR):
    def __init__(self, optimizer: Optimizer, n_steps: int = 1000):
        super().__init__(optimizer, lambda epoch: min((epoch - 1) / (n_steps - 1), 1.0))


class WarmUpStepLR(LambdaLR):
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int = 1000,
        decay_start_step: int = 50000,
        decay_stop_step: int = 50000,
        step_size: int = 1500,
        decay: float = 0.98,
    ):
        # super().__init__(
        #     optimizer,
        #     lambda epoch: (
        #         min((epoch - 1) / (warmup_steps - 1), 1.0)
        #         if epoch < decay_start_step
        #         else decay ** ((min(epoch, decay_stop_step) - decay_start_step) // step_size)
        #     ),
        # )

        def lr_lambda(step: int):
            # warmup
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))

            # keep constant lr
            elif step < decay_start_step:
                return 1.0

            # decay
            else:
                effective_step = min(step, decay_stop_step) - decay_start_step
                num_decays = effective_step // step_size
                return decay**num_decays

        super().__init__(optimizer, lr_lambda)
