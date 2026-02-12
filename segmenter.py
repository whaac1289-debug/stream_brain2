import torch

class LatentSegmenter:

    def __init__(self, thresh=0.35):
        self.prev_z = None
        self.thresh = thresh
        self.buffer = []

    def step(self, z):

        z = z.detach().cpu()

        if self.prev_z is None:
            self.prev_z = z
            self.buffer.append(z)
            return None

        dist = torch.norm(z - self.prev_z).item()

        self.prev_z = z
        self.buffer.append(z)

        # yangi segment topildi
        if dist > self.thresh and len(self.buffer) > 4:
            seg = torch.stack(self.buffer)
            self.buffer = []
            return seg

        return None


segmenter = LatentSegmenter()
