import torch

def pattern_from_decision(z, decision):

    if decision == "react":
        return z * 1.4

    if decision == "investigate":
        return z * torch.linspace(0.5, 1.5, z.numel(), device=z.device)

    if decision == "focus":
        return z * 0.7

    if decision == "uncertain":
        return z * torch.sin(z)

    return z
