import torch
import numpy as np

class VoiceStyle:

    def __init__(self, dim=64):
        self.timbre = torch.randn(dim) * 0.2

    def apply_timbre(self, z):
        return z + self.timbre.to(z.device)

voice_style = VoiceStyle()
