# Copyright (c) 2026 National Institute of Advanced Industrial Science and Technology (AIST), Japan
# Copyright (c) 2024 Kohei Saijo
#
# SPDX-License-Identifier: MIT
# SPDX-License-Identifier: MIT

import torch
from torch import nn


class ThresSASDRLoss(nn.Module):
    def __init__(self, snr_max=30, solve_perm=False):
        super().__init__()

        self.snr_max = snr_max
        self.temp = 0 if snr_max is None else 10 ** (-snr_max / 10)
        self.solve_perm = solve_perm

        assert not self.solve_perm, "PIT is not implemented yet"

    def forward(self, est, ref):
        assert est.ndim == ref.ndim, (est.shape, ref.shape)

        noise = ((ref - est) ** 2).sum(dim=-1)
        tgt = (ref**2).sum(dim=-1)

        neg_snr = 10 * torch.log10((noise + self.temp * tgt).sum(dim=-1))
        neg_snr -= 10 * torch.log10(tgt.sum(dim=-1))

        return neg_snr
