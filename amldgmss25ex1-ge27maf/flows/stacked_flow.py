from typing import List, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions.multivariate_normal import MultivariateNormal

from .nf_utils import Flow


class StackedFlows(nn.Module):
    """Stack a list of transformations with a given based distribtuion.

    Args:
        transforms: list fo stacked transformations. list of Flows
        dim: dimension of input/output data. int
        base_dist: name of the base distribution. options: ['Normal']
    """

    def __init__(
        self,
        transforms: List[Flow],
        dim: int = 2,
        base_dist: str = "Normal",
        device="cpu",
    ):
        super().__init__()

        if isinstance(transforms, Flow):
            self.transforms = nn.ModuleList(
                [
                    transforms,
                ]
            )
        elif isinstance(transforms, list):
            if not all(isinstance(t, Flow) for t in transforms):
                raise ValueError("transforms must be a Flow or a list of Flows")
            self.transforms = nn.ModuleList(transforms)
        else:
            raise ValueError(
                f"transforms must a Flow or a list, but was {type(transforms)}"
            )

        self.dim = dim
        if base_dist == "Normal":
            self.base_dist = MultivariateNormal(
                torch.zeros(self.dim).to(device), torch.eye(self.dim).to(device)
            )
        else:
            raise NotImplementedError

    def log_prob(self, x: Tensor) -> Tensor:
        """Compute log probability of a batch of data.

        Args:
            x: input sample. shape [batch_size, dim]

        Returns:
            log_prob: Log probability of the data, shape [batch_size]
        """

        B, D = x.shape

        # initialize the current variable to x (which is z_k, the final output of the flow)
        z_k = x
        # initialize accumulator for the sum of log determinants of the inverse Jacobians
        # this corresponds to sum(log |det(df_i^-1 / dz_i)|)
        sum_inv_log_det_jac = torch.zeros(B, device=x.device)

        # iterate through the transformations in reverse order (from f_k^-1 down to f_1^-1)
        # to map x (z_k) back to z_0.
        for transform in reversed(self.transforms):
            # apply inverse transformation of the current flow layer
            # 'z' here becomes the input to the next inverse transformation (e.g., z_i becomes z_{i-1})
            # 'inv_log_det_jac' is log|det(df_i^-1 / dz_i)|
            z_k, inv_log_det_jac_i = transform.inverse(z_k)
            # accumulate the log determinant terms
            sum_inv_log_det_jac += inv_log_det_jac_i

        # after iterating through all inverse transformations, z_k is now z_0
        z_0 = z_k

        # compute the log probability of z_0 under the base distribution p_{Z_0}(z_0)
        log_prob_base = self.base_dist.log_prob(z_0)

        # total log probability of x is:
        # log p_X(x) = log p_Z0(z_0) + sum(log |det(df_i^-1 / dz_i)|)
        log_prob = log_prob_base + sum_inv_log_det_jac

        assert log_prob.shape == (B,)

        return log_prob

    def rsample(self, batch_size: int) -> Tuple[Tensor, Tensor]:
        """Sample from the transformed distribution.

        Returns:
            x: sample after forward transformation, shape [batch_size, dim]
            log_prob: Log probability of x, shape [batch_size]
        """
        # sample z_0 from the base distribution
        z_0 = self.base_dist.sample((batch_size,))
        # get log probability of z_0 under the base distribution p_{Z_0}(z_0)
        log_prob_base = self.base_dist.log_prob(z_0)

        # initialize current variable for transformation.
        current_z = z_0
        # initialize accumulator for the sum of log determinants of the forward Jacobians
        # this corresponds to sum(log |det(df_i / dz_{i-1})|)
        sum_log_det_jac = torch.zeros(batch_size, device=z_0.device)

        # iterate through the transformations in forward order (f_1 to f_k)
        # to map z_0 to x (z_k).
        for transform in self.transforms:
            # apply forward transformation of the current flow layer
            # 'current_z' is z_{i-1}. 'next_z' becomes z_i.
            # 'log_det_jac_i' is log|det(df_i / dz_{i-1})|
            next_z, log_det_jac_i = transform.forward(current_z)
            # accumulate the log determinant terms
            sum_log_det_jac += log_det_jac_i
            # update current_z for the next iteration
            current_z = next_z

        # final transformed variable is x (which is z_k)
        x = current_z

        # total log probability of x is:
        # log p_X(x) = log p_Z0(z_0) - SUM(log |det(df_i / dz_{i-1})|)
        log_prob = log_prob_base - sum_log_det_jac

        assert x.shape == (batch_size, self.dim)
        assert log_prob.shape == (batch_size,)

        return x, log_prob
