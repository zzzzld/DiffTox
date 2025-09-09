import torch

def sin_mask_ratio_adapter(beta_t_bar, max_deviation=0.2, center=0.5):
    adjusted = beta_t_bar * torch.pi * 0.5
    sine = torch.sin(adjusted)
    adjustment = sine * max_deviation
    mask_ratio = center + adjustment
    return mask_ratio.squeeze(1)