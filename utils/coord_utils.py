import torch

def contract_cube(x, a=1.0):
    """Contracts points from world to a radius 2 cube centered about the origin."""
    eps = torch.finfo(x.dtype).eps
    # Clamping to eps prevents non-finite gradients when x == 0.

    x_inf = torch.amax(torch.abs(x).clamp_min(eps), dim=-1, keepdim=True)

    z = torch.where(x_inf <= 1, a * x, (2 - (2 - a) / x_inf) * x / x_inf)
    return z

def inv_contract_cube(z, a=1.0):
    """The inverse of nerf_contract_inf_general()."""
    eps = torch.finfo(z.dtype).eps

    # Clamping to eps prevents non-finite gradients when z == 0.
    z_inf = torch.amax(torch.abs(z).clamp_min(eps), dim=-1, keepdim=True)

    x = torch.where(z_inf <= a, z / a, z * (2 - a) / (z_inf * (2 - z_inf)).clamp_min(eps))
    return x