import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

import wandb

def extract(a, t, x_shape):
    """
    This function abstracts away the tedious indexing that would otherwise have
    to be done to properly compute the diffusion equations from lecture. This
    is necessary because we train data in batches, while the math taught in
    lecture only considers a single sample.
    
    To use this function, consider the example
        alpha_t * x
    To compute this in code, we would write
        extract(alpha, t, x.shape) * x

    Args:
        a: 1D tensor containing the value at each time step.
        t: 1D tensor containing a batch of time indices.
        x_shape: The reference shape.
    Returns:
        The extracted tensor.
    """
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def cosine_schedule(timesteps, s=0.008):
    """
    Passes the input timesteps through the cosine schedule for the diffusion process
    Args:
        timesteps: 1D tensor containing a batch of time indices.
        s: The strength of the schedule.
    Returns:
        1D tensor of the same shape as timesteps, with the computed alpha.
    """
    steps = timesteps + 1
    x = torch.linspace(0, steps, steps)
    alpha_bar = torch.cos(((x / steps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alpha_bar = alpha_bar / alpha_bar[0]
    alphas = alpha_bar[1:] / alpha_bar[:-1]
    return torch.clip(alphas, 0.001, 1)


# normalization functions
def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5


# DDPM implementation
class Diffusion(nn.Module):
    def __init__(
        self,
        model,
        *,
        image_size,
        channels=3,
        timesteps=1000,
    ):
        super().__init__()
        self.channels = channels
        self.image_size = image_size
        self.model = model
        self.num_timesteps = int(timesteps)

        """
        Initializes the diffusion process.
            1. Setup the schedule for the diffusion process.
            2. Define the coefficients for the diffusion process.
        Args:
            model: The model to use for the diffusion process.
            image_size: The size of the images.
            channels: The number of channels in the images.
            timesteps: The number of timesteps for the diffusion process.
        """
        ## TODO: Implement the initialization of the diffusion process ##
        # 1. define the scheduler here
        # 2. pre-compute the coefficients for the diffusion process
        # ###########################################################

        # per-step α values
        T = timesteps
        s = 0.008
        alphas = cosine_schedule(timesteps=T, s=s)  # shape: [T], 0 to T-1
        self.register_buffer('alphas', alphas)
        betas = 1 - alphas
        self.register_buffer('betas', betas)

        # cumulative product of α values for use in q_sample
        # steps = T + 1  # t = 0,...,T
        # x = torch.linspace(0, steps, steps)
        # alpha_bar_full = torch.cos(((x / steps) + s) / (1 + s) * torch.pi * 0.5) ** 2  # in new paper
        # alpha_bar_full = alpha_bar_full / alpha_bar_full[0]  # shape: [T+1], from -1, with 1 at [0]
        # alpha_bar = alpha_bar_full[1:]  # shape: [T]. For t = 1,...,T, product of all previous α up to t
        alpha_bar = torch.cumprod(alphas, dim=0)
        alpha_bar_prev = F.pad(alpha_bar[:-1], (1, 0), value=1.0)
        self.register_buffer('alpha_bar', alpha_bar)
        self.register_buffer('alpha_bar_prev', alpha_bar_prev)
        self.register_buffer('sqrt_alpha_bar', torch.sqrt(alpha_bar))
        self.register_buffer('sqrt_one_minus_alpha_bar', torch.sqrt(1 - alpha_bar))

        # Compute posterior variance for the reverse process:
        # For t >= 1, σₜ² = (1 - ᾱ₍ₜ₋₁₎) / (1 - ᾱₜ) * βₜ, with convention ᾱ₋₁ = 1.
        # q_var = torch.empty_like(betas)  # shape: [T]
        q_var = betas * (1 - alpha_bar_prev) / (1 - alpha_bar)
        # q_var[0] = 0.0  # at t==0
        # for t in range(1, T):
        #     # Note: alpha_bar_full[t] corresponds to ᾱₜ₋₁ and alpha_bar_full[t+1] to ᾱₜ.
        #     # q_var[t] = betas[t] * (1 - alpha_bar_full[t]) / (1 - alpha_bar_full[t+1])
        #     q_var[t] = betas[t] * (1 - alpha_bar[t-1]) / (1 - alpha_bar[t])
        self.register_buffer('q_var', q_var)

    def noise_like(self, shape, device):
        """
        Generates noise with the same shape as the input.
        Args:
            shape: The shape of the noise.
            device: The device on which to create the noise.
        Returns:
            The generated noise.
        """
        noise = lambda: torch.randn(shape, device=device)
        return noise()

    # backward diffusion
    @torch.no_grad()
    def p_sample(self, x, t, t_index):
        """
        Computes the (t_index)th sample from the (t_index + 1)th sample using
        the reverse diffusion process.
        Args:
            x: The sampled image at timestep t_index + 1.
            t: 1D tensor of the index of the time step.
            t_index: Scalar of the index of the time step.
        Returns:
            The sampled image at timestep t_index.
        """
        ####### TODO: Implement the p_sample function #######
        # sample x_{t-1} from the gaussian distribution wrt. posterior mean and posterior variance
        # Hint: use extract function to get the coefficients at time t
        # Hint: use self.noise_like function to generate noise. DO NOT USE torch.randn
        # Begin code here#######################################################################

        # Predict noise using the model, i.e. ϵθ(xₜ, t)
        eps_theta = self.model(x, t)
        # Estimate x₀: x₀_pred = (x - √(1 - ᾱₜ) * ϵθ) / √(ᾱₜ)
        sqrt_alpha_bar = extract(self.sqrt_alpha_bar, t, x.shape)
        sqrt_one_minus_alpha_bar = extract(self.sqrt_one_minus_alpha_bar, t, x.shape)
        x0_pred = (x - sqrt_one_minus_alpha_bar * eps_theta) / sqrt_alpha_bar
        x0_pred = torch.clamp(x0_pred, -1, 1)

        # Extract coefficients needed for the reverse process.
        beta_t = extract(self.betas, t, x.shape)
        alpha_t = extract(self.alphas, t, x.shape)
        alpha_bar_t = extract(self.alpha_bar, t, x.shape)
        # For t_index == 0, we set ᾱ₍ₜ₋₁₎ = 1.
        # if t_index == 0:
        #     alpha_bar_prev = torch.ones_like(alpha_bar_t)
        # else:
        #     alpha_bar_prev = extract(self.alpha_bar, t - 1, x.shape)

        # Compute posterior mean:
        # μ̃ₜ = √(αₜ) * ((1 - ᾱ₍ₜ₋₁₎) / (1 - ᾱₜ)) * xₜ +
        #       √(ᾱ₍ₜ₋₁₎) * ((1 - αₜ) / (1 - ᾱₜ)) * x₀_pred
        alpha_bar_prev = extract(self.alpha_bar_prev, t, x.shape)
        posterior_mean = (torch.sqrt(alpha_t) * (1 - alpha_bar_prev) / (1 - alpha_bar_t)) * x + \
                         (torch.sqrt(alpha_bar_prev) * (1 - alpha_t) / (1 - alpha_bar_t)) * x0_pred

        # Get posterior variance.
        q_var = extract(self.q_var, t, x.shape)

        # Add noise if not at the final step.
        if t_index == 0:
            noise = torch.zeros_like(x)
        else:
            noise = self.noise_like(x.shape, x.device)

        return posterior_mean + torch.sqrt(q_var) * noise
        # if t_index == 0:
        #     return ...
        # else:
        #     return ...
        # ####################################################

    @torch.no_grad()
    def p_sample_loop(self, img):
        """
        Passes noise through the entire reverse diffusion process to generate
        final image samples.
        Args:
            img: The initial noise that is randomly sampled from the noise distribution.
        Returns:
            The sampled images.
        """
        b = img.shape[0]
        #### TODO: Implement the p_sample_loop function ####
        # 1. loop through the time steps from the last to the first
        # 2. inside the loop, sample x_{t-1} from the reverse diffusion process
        # 3. clamp and unnormalize the generated image to valid pixel range
        # Hint: to get time index, you can use torch.full()
        # for i in reversed(range(self.num_timesteps)):
        for i in range(self.num_timesteps - 1, -1, -1):
            t = torch.full((b,), i, device=img.device, dtype=torch.long)
            img = self.p_sample(img, t, i)
        img = torch.clamp(img, -1, 1)
        img = unnormalize_to_zero_to_one(img)
        return img
        # ####################################################

    @torch.no_grad()
    def sample(self, batch_size):
        """
        Wrapper function for p_sample_loop.
        Args:
            batch_size: The number of images to sample.
        Returns:
            The sampled images.
        """
        self.model.eval()
        #### TODO: Implement the sample function ####
        # Hint: use self.noise_like function to generate noise. DO NOT USE torch.randn
        device = next(self.model.parameters()).device
        img = self.noise_like((batch_size, self.channels, self.image_size, self.image_size), device)
        img = self.p_sample_loop(img)
        # img = ...
        return img

    # forward diffusion
    def q_sample(self, x_0, t, noise):
        """
        Applies alpha interpolation between x_0 and noise to simulate sampling
        x_t from the noise distribution.
        Args:
            x_0: The initial images.
            t: 1D tensor containing a batch of time indices to sample at.
            noise: The noise tensor to sample from.
        Returns:
            The sampled images.
        """
        ###### TODO: Implement the q_sample function #######
        sqrt_alpha_bar = extract(self.sqrt_alpha_bar, t, x_0.shape)
        sqrt_one_minus_alpha_bar = extract(self.sqrt_one_minus_alpha_bar, t, noise.shape)
        x_t = sqrt_alpha_bar * x_0 + sqrt_one_minus_alpha_bar * noise
        # x_t = ...
        return x_t

    def p_losses(self, x_0, t, noise):
        """
        Computes the loss for the forward diffusion.
        Args:
            x_0: The initial images.
            t: 1D tensor containing a batch of time indices to compute the loss at.
            noise: The noise tensor to use.
        Returns:
            The computed loss.
        """
        ###### TODO: Implement the p_losses function #######
        # define loss function wrt. the model output and the target
        # Hint: you can use pytorch built-in loss functions: F.l1_loss
        x_t = self.q_sample(x_0, t, noise)
        predicted_noise = self.model(x_t, t)
        loss = F.l1_loss(predicted_noise, noise)
        # loss = ...
        return loss
        # ####################################################

    def forward(self, x_0, noise):
        """
        Acts as a wrapper for p_losses.
        Args:
            x_0: The initial images.
            noise: The noise tensor to use.
        Returns:
            The computed loss.
        """
        b, c, h, w, device, img_size, = *x_0.shape, x_0.device, self.image_size
        device = x_0.device
        assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
        ###### TODO: Implement the forward function #######
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        return self.p_losses(x_0, t, noise)
        # t = torch.randint(...)
        # return ...
