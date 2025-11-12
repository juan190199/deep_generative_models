from functools import partial
import math

import einops as eo
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked
from tqdm import tqdm

patch_typeguard()

# A batch of (noisy) images
ImageBatch = TensorType["batch_size", "channels", "height", "width", torch.float32]
ModelOutputBatch = TensorType["batch_size", "output_channels", "height", "width", torch.float32]

# Integer noise level between 0 and N - 1
NoiseLevel = TensorType["batch_size", torch.long]

# Normalized noise level between 0 and 1
NormalizedNoiseLevel = TensorType["batch_size", torch.float32]


def batch_broadcast(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Make `a` broadcast along the batch dimension of `b`.
    We assume the batch dimension to be the first one.
    """
    assert a.ndim == 1
    return a.view(-1, *((1,) * (b.ndim - 1)))


class ResNet(nn.Module):
    """A minimal convolutional residual network."""

    def __init__(self, in_channels: int, hidden_dim: int, n_layers: int, output_dim: int):
        super().__init__()

        ConvLayer = partial(nn.Conv2d, kernel_size=3, padding=1)

        # Layers to map from data space to learned latent space and back
        self.embed = nn.Sequential(ConvLayer(in_channels + 1, hidden_dim), nn.SiLU())
        self.out = ConvLayer(hidden_dim, output_dim)

        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    ConvLayer(hidden_dim + 1, hidden_dim),
                    nn.SiLU(),
                    ConvLayer(hidden_dim, hidden_dim, kernel_size=3),
                )
                for i in range(n_layers)
            ]
        )

    @typechecked
    def forward(self, z_n: ImageBatch, n: NormalizedNoiseLevel) -> ModelOutputBatch:
        # Align n with the feature dimension of 2D image tensors
        n = n[:, None, None, None].expand(n.shape[0], -1, *z_n.shape[2:])

        z_n = self.embed(torch.cat((z_n, n), dim=-3))

        for layer in self.layers:
            z_n = z_n + layer(torch.cat((z_n, n), dim=-3))

        return self.out(z_n)


class MiniUnet(nn.Module):
    """A minimal U-net implementation [1].

    [1] Olaf Ronneberger, Philipp Fischer, Thomas Brox: "U-Net: Convolutional Networks
        for Biomedical Image Segmentation". https://arxiv.org/abs/1505.04597
    """

    def __init__(self, in_channels: int, hidden_dim: int, n_layers: int, output_dim: int):
        super().__init__()

        assert n_layers <= 2, (
            "MNIST images can only be downsampled twice "
            "without taking care of padding issues"
        )

        self.n_layers = n_layers

        ConvLayer = partial(nn.Conv2d, kernel_size=3, padding=1)

        # Layers to map from data space to learned latent space and back
        self.embed = nn.Sequential(ConvLayer(in_channels + 1, hidden_dim), nn.SiLU())
        self.out = ConvLayer(hidden_dim, output_dim)

        # At each scale, we perform one nonlinear map with residual connection
        self.downscaling = nn.ModuleList(
            [
                nn.Sequential(
                    ConvLayer(4 ** i * hidden_dim + 1, 4 ** i * hidden_dim),
                    nn.SiLU(),
                    nn.Conv2d(4 ** i * hidden_dim, 4 ** i * hidden_dim, kernel_size=1),
                )
                for i in range(n_layers)
            ]
        )
        bottom_channels = 4 ** n_layers * hidden_dim
        self.bottom_map = nn.Sequential(
            ConvLayer(bottom_channels + 1, bottom_channels),
            nn.SiLU(),
            ConvLayer(bottom_channels, bottom_channels),
        )
        self.upscaling = nn.ModuleList(
            [
                nn.Sequential(
                    ConvLayer(2 * 4 ** i * hidden_dim + 1, 4 ** i * hidden_dim),
                    nn.SiLU(),
                    nn.Conv2d(4 ** i * hidden_dim, 4 ** i * hidden_dim, kernel_size=1),
                )
                for i in reversed(range(1, n_layers + 1))
            ]
        )

    @typechecked
    def forward(self, z_n: ImageBatch, n: NormalizedNoiseLevel) -> ModelOutputBatch:
        # Align n with the feature dimension of 2D image tensors
        n = n[:, None, None, None]

        def cat_n(z_n, *tensors):
            return torch.cat((z_n, *tensors, n.expand(-1, -1, *z_n.shape[2:])), dim=-3)

        z_n = self.embed(cat_n(z_n))

        skip_connections = []
        for down_layer in self.downscaling:
            z_n = z_n + down_layer(cat_n(z_n))
            z_n = eo.rearrange(z_n, "b c (h h2) (w w2) -> b (c h2 w2) h w", h2=2, w2=2)
            skip_connections.append(z_n)

        z_n = self.bottom_map(cat_n(z_n))

        for up_layer in self.upscaling:
            z_n = z_n + up_layer(cat_n(z_n, skip_connections.pop()))
            z_n = eo.rearrange(z_n, "b (c h2 w2) h w -> b c (h h2) (w w2)", h2=2, w2=2)

        return self.out(z_n)


class DDPM(nn.Module):
    """A denoising diffusion model as described in [1].

    References:
    [1] "Denoising Diffusion Probabilistic Models", Ho et al., https://arxiv.org/abs/2006.11239
    """

    def __init__(
            self,
            N: int,
            type: str,
            hidden_dim: int,
            n_layers: int,
            in_channels: int,
            image_size: int,
            beta_schedule: str = "linear",
            var_schedule: str = "fixed",
            loss_type: str = "simple",
            importance_sampling: bool = False
    ):
        """Initialize the diffusion model.

        Args:
            N: number of diffusion steps
            type: "resnet" or "unet"
            hidden_dim: base hidden dimension for the model
            n_layers: number of layers (for ResNet) or downsampling blocks (for U-Net)
            in_channels: number of input image channels
            image_size: size of input image
            beta_schedule: "linear" (DDPM) or "cosine" (improved DDPM)
            var_schedule: "fixed" (fixed variance) or "learned" (improved DDPM)
            loss_type: "simple" (MSE on epsilon) or "hybrid" (MSE + VLB, improved DDPM)
        """
        super().__init__()

        self.N = N
        self.type = type
        self.var_schedule = var_schedule
        self.loss_type = loss_type
        self.importance_sampling = importance_sampling
        self.in_channels = in_channels
        self.image_size = image_size

        # determine model output dimension
        # original DDPM: 1x channels (predicts epsilon)
        # improved DDPM: 2x channels (predicts epsilon and variance)
        if var_schedule == "fixed":
            self.output_dim = self.in_channels
        elif var_schedule == "learned":
            self.output_dim = self.in_channels * 2
        else:
            raise ValueError(f"Unknown var schedule {var_schedule}")

        if type == "resnet":
            self.model = ResNet(
                in_channels=self.in_channels,
                hidden_dim=hidden_dim,
                n_layers=n_layers,
                output_dim=self.output_dim
            )
        elif type == "unet":
            self.model = MiniUnet(
                in_channels=self.in_channels,
                hidden_dim=hidden_dim,
                n_layers=n_layers,
                output_dim=self.output_dim
            )
        else:
            raise RuntimeError(f"Unknown model type {type}")

        # Compute a beta schedule
        if beta_schedule == "linear":
            beta = torch.linspace(1e-4, 0.02, self.N)
        elif beta_schedule == "cosine":
            beta = self._get_cosine_betas(self.N)
        else:
            raise ValueError(f"Unknown beta schedule {beta_schedule}")

        # beta = beta.clamp(min=1e-20, max=0.999)

        alpha = 1.0 - beta
        alpha_bar = torch.cumprod(alpha, dim=0)
        alpha_bar_prev = torch.cat([torch.tensor([1.0]), alpha_bar[:-1]], dim=0)

        # This is beta tilde from the DDPM paper (fixed variance for reverse process)
        beta_tilde = beta * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar)

        self.register_buffer("alpha", alpha.float())
        self.register_buffer("beta", beta.float())
        self.register_buffer("alpha_bar", alpha_bar.float())
        self.register_buffer("alpha_bar_prev", alpha_bar_prev.float())
        self.register_buffer("beta_tilde", beta_tilde.float())

        self.register_buffer("log_beta", torch.log(beta).float())
        self.register_buffer("log_beta_tilde", torch.log(beta_tilde.clamp(min=1e-20)).float())

        # EMA of VLB loss for each time step
        self.register_buffer("vlb_ema", torch.ones(self.N) * 10.0)
        self.register_buffer("vlb_counts", torch.zeros(self.N, dtype=torch.long))
        self.ema_decay = 0.9
        self.warmup_samples = 10

    def _get_cosine_betas(self, N: int, s: float = 0.008) -> torch.Tensor:
        """Utility for improved DDPM cosine schedule."""
        t = torch.linspace(0, N, N + 1, dtype=torch.float64)
        f_t = torch.cos(((t / N + s) / (1 + s)) * (math.pi / 2)) ** 2
        alpha_bar = f_t / f_t[0]

        # Calculate beta_t from alpha_bar_t
        alpha_bar_prev = torch.cat([torch.tensor([1.0]), alpha_bar[:-1]], dim=0)
        beta = 1.0 - (alpha_bar / alpha_bar_prev)

        return beta.clamp(max=0.999).float()[1:]

    def _get_epsilon_and_var(self, model_output: ModelOutputBatch) -> tuple[ImageBatch, ImageBatch | None]:
        """
        Split model output into predicted epsilon and variance 'v'.
        The 'v' parameter is used to interpolate the log-variance.
        """
        if self.var_schedule == "fixed":
            return model_output, None
        elif self.var_schedule == "learned":
            pred_epsilon, v = torch.chunk(model_output, 2, dim=1)
            return pred_epsilon, v
        else:
            raise ValueError(f"Unknown var schedule {self.var_schedule}")

    def _get_q_posterior_mean_var(self, x0: ImageBatch, z_n: ImageBatch, n: NoiseLevel) -> tuple[
        ImageBatch, ImageBatch]:
        """
        Calculate the mean and log-variance of the "true" posterior q(z_{n-1} | z_n, x0).
        This is the target value for the VLB loss
        """
        alpha_bar_prev_n = batch_broadcast(self.alpha_bar_prev[n], z_n)
        alpha_n = batch_broadcast(self.alpha[n], z_n)
        beta_n = batch_broadcast(self.beta[n], z_n)
        alpha_bar_n = batch_broadcast(self.alpha_bar[n], z_n)

        q_mean = ((torch.sqrt(alpha_bar_prev_n) * beta_n) / (1.0 - alpha_bar_prev_n)) * x0 + \
                 ((torch.sqrt(alpha_n) * (1.0 - alpha_bar_prev_n)) / (1.0 - alpha_bar_n)) * z_n

        q_log_var = batch_broadcast(self.log_beta_tilde[n], z_n)

        return q_mean, q_log_var

    def _get_p_theta_mean_var(
            self,
            model_output: ModelOutputBatch,
            z_n: ImageBatch,
            n: NoiseLevel
    ) -> tuple[ImageBatch, ImageBatch]:
        """
        Calculate the mean and log-variance of the reverse process p_theta(z_{n-1} | z_n).
        This is what the model *predicts* the posterior should be.
        """
        # get model predictions (epsilon and v)
        pred_epsilon, v = self._get_epsilon_and_var(model_output)

        # get estimate of x0 using the predicted epsilon
        est_x0 = self.estimate_x0(z_n, n, pred_epsilon)

        # get mean of p_theta (using *estimated* x0)
        alpha_bar_prev_n = batch_broadcast(self.alpha_bar_prev[n], z_n)
        alpha_n = batch_broadcast(self.alpha[n], z_n)
        beta_n = batch_broadcast(self.beta[n], z_n)
        alpha_bar_n = batch_broadcast(self.alpha_bar[n], z_n)

        # Unstable for when n=0
        # p_mean = ((torch.sqrt(alpha_bar_prev_n) * beta_n) / (1.0 - alpha_bar_prev_n)) * est_x0 + \
        #          ((torch.sqrt(alpha_n) * (1.0 - alpha_bar_prev_n)) / (1.0 - alpha_bar_n)) * z_n
        p_mean = (1.0 / torch.sqrt(alpha_n)) * (z_n - (beta_n / torch.sqrt(1.0 - alpha_bar_n)) * pred_epsilon)

        # get (log) variance of p_theta
        if self.var_schedule == "fixed":
            # variance is fixed to log(beta_tilde_n)
            p_log_var = batch_broadcast(self.log_beta_tilde[n], z_n)
        elif self.var_schedule == "learned":
            log_beta_n = batch_broadcast(self.log_beta[n], z_n)
            log_beta_tilde_n = batch_broadcast(self.log_beta_tilde[n], z_n)

            # the model output 'v' is a value from -1 to 1 (network output)
            # map to the range [log(beta_tilde), log(beta)]
            v = v.clamp(-1, 1)
            v_frac = (v + 1) / 2  # map from [-1, 1] to [0, 1]

            # interpolate between the min and max log-variances
            p_log_var = v_frac * log_beta_n + (1 - v_frac) * log_beta_tilde_n
        else:
            raise ValueError(f"Unknown var schedule {self.var_schedule}")

        return p_mean, p_log_var

    def _kl_divergence(self, q_mean, q_log_var, p_mean, p_log_var) -> torch.Tensor:
        """Compute the KL divergence between two Gaussian distributions"""
        return 0.5 * (
                p_log_var - q_log_var +
                (torch.exp(q_log_var) + (q_mean - p_mean) ** 2) / torch.exp(p_log_var)
                - 1.0
        )

    def _sample_t(
            self,
            batch_size: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Sample timesteps t and return sampling probabilities"""
        use_is = self.importance_sampling and self.loss_type == "vlb"

        is_warming_up = False
        if use_is:
            is_warming_up = (self.vlb_counts[1:] < self.warmup_samples).any()

        # sample uniformly if not using importance sampling or not in VLB mode
        if not use_is or is_warming_up:
            n = torch.randint(1, self.N, (batch_size,), device=device)
            return n, None

        # sample based on loss history
        vlb_weights = self.vlb_ema.clone()
        # set prob=0 for t=0 as it is not sampled
        vlb_weights[0] = 0.0

        # convert loss history to sampling probs
        probs = F.softmax(vlb_weights, dim=0)

        # sample t from the distribution
        n = torch.multinomial(probs, batch_size, replacement=True)

        # Fallback in case of numerical instability
        if torch.isnan(probs).any() or (probs.sum() == 0):
            n = torch.randint(1, self.N, (batch_size,), device=device)
            return n, None

        return n, probs

    @typechecked
    def loss(self, x0: ImageBatch) -> dict[str, torch.Tensor]:
        batch_size = x0.shape[0]

        # n = torch.randint(1, self.N, (batch_size,), device=x0.device)
        n, probs = self._sample_t(batch_size, x0.device)

        # create noisy image using forward process
        epsilon = torch.randn_like(x0)
        alpha_bar_n = self.alpha_bar[n].view(-1, 1, 1, 1)
        x_n = torch.sqrt(alpha_bar_n) * x0 + torch.sqrt(1 - alpha_bar_n) * epsilon

        # normalize the noise level from an integer to a float (0 to 1)
        normalized_n = n.float() / self.N

        # predict the noise using the model
        model_output = self.model(x_n, normalized_n)
        predicted_epsilon, _ = self._get_epsilon_and_var(model_output)

        # calculate L_simple (MSE loss)
        loss_simple = F.mse_loss(predicted_epsilon, epsilon, reduction="none")
        loss_simple_reduced = loss_simple.view(batch_size, -1).mean(dim=1)  # reduce per-sample

        losses = {"L_simple": loss_simple_reduced.mean()}  # always log L_simple
        total_loss = loss_simple_reduced  # default to L_simple

        # calculate L_vlb
        if self.loss_type in ["hybrid", "vlb"]:
            # get p_theta(z_{n-1} | z_n)
            p_mean, p_log_var = self._get_p_theta_mean_var(model_output, x_n, n)

            # get q(z_{n-1} | z_n, x0)
            q_mean, q_log_var = self._get_q_posterior_mean_var(x0, x_n, n)

            # stop gradient for L_vlb term only in hybrid loss, as per paper
            if self.loss_type == "hybrid":
                p_mean_detached = p_mean.detach()
            else:
                p_mean_detached = p_mean

            # calculate KL divergence (the L_vlb term)
            # use the (potentially detached) p_mean
            kl_loss = self._kl_divergence(q_mean, q_log_var, p_mean_detached, p_log_var)
            kl_loss_reduced = kl_loss.view(batch_size, -1).mean(dim=1) / math.log(2.0)  # VLB in bits/dim

            losses["L_vlb"] = kl_loss_reduced.mean()

            with torch.no_grad():
                for i, t in enumerate(n):
                    loss_val = kl_loss_reduced[i]

                    # update counts
                    self.vlb_counts[t] += 1
                    count = self.vlb_counts[t]

                    if count == 1:
                        self.vlb_ema[t] = loss_val
                    elif count < self.warmup_samples:
                        self.vlb_ema[t] = (self.vlb_ema[t] * (count - 1) + loss_val) / count
                    else:
                        self.vlb_ema[t] = self.vlb_ema[t] * self.ema_decay + loss_val * (1.0 - self.ema_decay)

            if self.loss_type == "hybrid":
                # L_hybrid = L_simple + lambda * L_vlb
                total_loss = (loss_simple_reduced + 0.001 * kl_loss_reduced)
            else:  # loss_type == "vlb"
                if self.importance_sampling and probs is not None:
                    # apply importance sampling weights
                    # L_vlb = E[L_t / p_t] / T  (where p_t is the probability)
                    prob_n = probs[n].clamp(min=1e-20)
                    weights = (1.0/self.N) / prob_n
                    total_loss = kl_loss_reduced * weights
                else:
                    total_loss = kl_loss_reduced

        losses["total_loss"] = total_loss.mean()
        return losses

    @typechecked
    def estimate_x0(
            self, z_n: ImageBatch, n: NoiseLevel, epsilon: ImageBatch
    ) -> ImageBatch:
        """Re-construct x_0 from z_n and epsilon."""
        # get alpha_bar at the current noise level n
        alpha_bar_n = self.alpha_bar[n]
        # reshape alpha_bar_n for broadcasting
        alpha_bar_n = alpha_bar_n.view(-1, 1, 1, 1)

        # apply formula to estimate x_0 from z_n and epsilon
        x0_estimate = (1.0 / torch.sqrt(alpha_bar_n)) * (z_n - torch.sqrt(1 - alpha_bar_n) * epsilon)
        return x0_estimate

    @typechecked
    def _p_sample(
            self,
            model_output: ModelOutputBatch,
            z_n: ImageBatch,
            n: NoiseLevel
    ) -> ImageBatch:
        """Sample z_{n-1} from p_theta(z_{n-1} | z_n)"""

        p_mean, p_log_var = self._get_p_theta_mean_var(model_output, z_n, n)

        # no noise is added at the last step.
        noise = torch.randn_like(z_n)
        nonzero_mask = (n > 0).float().view(-1, 1, 1, 1)

        # reparametrization trick instead of drawing a sample from Gaussian
        z_n_previous = p_mean + nonzero_mask * torch.exp(0.5 * p_log_var) * noise

        return z_n_previous

    @torch.no_grad()
    def sample(self, batch_size: int, device: torch.device, num_steps: int = -1) -> ImageBatch:
        """Sample new images from scratch by iteratively denoising pure noise.

        Args:
            batch_size: Number of images to generate
            device: Device to generate them on
            num_steps: Number of sampling steps (K). If -1 or >= self.N,
                       uses all T (self.N) steps.

        Returns:
            Generated images
        """
        image_shape = (self.in_channels, self.image_size, self.image_size)
        z_n = torch.randn((batch_size, *image_shape), device=device)

        if num_steps <= 0 or num_steps > self.N:
            timesteps = reversed(range(self.N))
            total_steps = self.N
            desc = f"Sampling (full {self.N} steps)..."
        else:
            # generate K evenly spaced timesteps from T-1 down to 0
            timesteps = torch.linspace(self.N - 1, 0, num_steps, device=device).round().long()
            total_steps = num_steps
            desc = f"Sampling ({num_steps} steps)..."

        # iterate backward through the timesteps
        for n_val in tqdm(timesteps, total=total_steps, desc=desc):
            n = n_val.item() if isinstance(n_val, torch.Tensor) else n_val

            n_tensor = torch.full((batch_size,), n, dtype=torch.long, device=device)

            # predict the noise using the model
            normalized_n = n_tensor.float() / self.N
            model_output = self.model(z_n, normalized_n)

            # sample the previous step z_{n-1} using the estimated x0
            z_n = self._p_sample(model_output, z_n, n_tensor)

        return z_n
