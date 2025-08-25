import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .nf_utils import Flow


class Radial(Flow):
    """Radial transformation.

    Args:
        dim: dimension of input/output data, int
    """

    def __init__(self, dim: int = 2):
        """Create and initialize an affine transformation."""
        super().__init__()

        self.dim = dim

        self.x0 = nn.Parameter(
            torch.Tensor(
                self.dim,
            )
        )  # Vector used to parametrize z_0
        self.pre_alpha = nn.Parameter(
            torch.Tensor(
                1,
            )
        )  # Scalar used to indirectly parametrized \alpha
        self.pre_beta = nn.Parameter(
            torch.Tensor(
                1,
            )
        )  # Scaler used to indireclty parametrized \beta

        stdv = 1.0 / math.sqrt(self.dim)
        self.pre_alpha.data.uniform_(-stdv, stdv)
        self.pre_beta.data.uniform_(-stdv, stdv)
        self.x0.data.uniform_(-stdv, stdv)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Compute the forward transformation for the given input x.

        Args:
            x: input sample, shape [batch_size, dim]

        Returns:
            y: sample after forward transformation, shape [batch_size, dim]
            log_det_jac: log determinant of the jacobian of the forward transformation, shape [batch_size]
        """
        B, D = x.shape

        # enforce parameter constraints
        # alpha = softplus(tilde_alpha) ensures alpha > 0
        alpha = F.softplus(self.pre_alpha)  # Shape [1]
        # beta = -alpha + softplus(tilde_beta) ensures beta >= -alpha
        beta = -alpha + F.softplus(self.pre_beta)  # Shape [1]

        # calculate the difference vector (z - z_0)
        diff = x - self.x0  # Shape [B, D] (self.x0 broadcasts)

        # calculate r = \|z - z_0\|_2 (Euclidean norm)
        # dim=1 specifies the dimension over which to compute the norm (the feature dimension for each sample)
        # keepdim=True ensures the result is [B, 1] instead of [B]
        r = torch.norm(diff, p=2, dim=1, keepdim=True)  # shape [B, 1]

        # calculate h(alpha, r) = 1 / (alpha + r)
        # alpha is [1], r is [B, 1]. Broadcasting makes this work element-wise for each batch sample.
        h_val = 1.0 / (alpha + r)  # shape [B, 1]

        # compute the forward transformation: y = z + beta * h(alpha, r) * (z - z_0)
        # beta is [1], h_val is [B, 1], diff is [B, D]. Broadcasting handles these shapes correctly.
        y = x + beta * h_val * diff  # shape [B, D]

        # calculate the log determinant of the Jacobian:
        # log|det(J_f)| = (D-1)log(1 + beta*h(alpha,r)) + log(1 + beta*h(alpha,r) + beta*r*h'(alpha,r))
        # where h'(alpha, r) = -1 / (alpha + r)^2

        # calculate h'(alpha, r)
        h_prime_val = -1.0 / ((alpha + r) ** 2)  # Shape [B, 1]

        # term for the first logarithm: (1 + beta * h(alpha, r))
        term1_inside_log = 1.0 + beta * h_val  # Shape [B, 1]

        # term for the second logarithm: (1 + beta * h(alpha, r) + beta * r * h'(alpha, r))
        term2_inside_log = 1.0 + beta * h_val + beta * r * h_prime_val  # Shape [B, 1]

        # combine terms to get the log determinant for each sample in the batch
        # arguments to torch.log are positive (guaranteed by constraints in the paper)
        log_det_jac_per_sample = (D - 1) * torch.log(term1_inside_log) + torch.log(term2_inside_log)

        # The result 'log_det_jac_per_sample' is [B, 1], but the assertion expects [B,]
        # Squeeze removes the dimension of size 1.
        log_det_jac = log_det_jac_per_sample.squeeze(1)

        assert y.shape == (B, D)
        assert log_det_jac.shape == (B,)

        return y, log_det_jac

    def inverse(self, y: Tensor) -> None:
        """Compute the inverse transformation given an input y.

        Args:
            y: input sample. shape [batch_size, dim]

        Returns:
            x: sample after inverse transformation. shape [batch_size, dim]
            inv_log_det_jac: log determinant of the jacobian of the inverse transformation, shape [batch_size]
        """
        raise ValueError("The inverse transformation is not known in closed form.")
