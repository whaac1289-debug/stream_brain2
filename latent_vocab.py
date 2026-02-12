import torch
import numpy as np

class LatentVocab:

    def __init__(self, max_clusters=32, dist_thresh=0.6):
        self.max_clusters = max_clusters
        self.dist_thresh = dist_thresh

        self.centroids = []
        self.counts = []
        self.seq_buffer = []

    # -------------------------
    # cluster topish / yaratish
    # -------------------------
    def get_cluster(self, z):

        if isinstance(z, torch.Tensor):
            z = z.detach().cpu().numpy()

        if len(self.centroids) == 0:
            self.centroids.append(z)
            self.counts.append(1)
            return 0

        dists = [np.linalg.norm(z - c) for c in self.centroids]
        idx = int(np.argmin(dists))

        if dists[idx] < self.dist_thresh:
            # centroid update (running mean)
            self.counts[idx] += 1
            lr = 1.0 / self.counts[idx]
            self.centroids[idx] = (1 - lr) * self.centroids[idx] + lr * z
            return idx

        if len(self.centroids) < self.max_clusters:
            self.centroids.append(z)
            self.counts.append(1)
            return len(self.centroids) - 1

        return idx

    # -------------------------
    # syllable param
    # -------------------------
    def cluster_to_sound(self, cid):
        base = 180 + cid * 35
        dur = 0.12 + (cid % 3) * 0.04
        amp = 0.2 + (cid % 5) * 0.05
        return base, dur, amp

    # -------------------------
    # sequence builder
    # -------------------------
    def push(self, cid):
        self.seq_buffer.append(cid)
        if len(self.seq_buffer) > 6:
            self.seq_buffer.pop(0)

    def get_word_pattern(self):
        if len(self.seq_buffer) < 3:
            return None
        return tuple(self.seq_buffer[-3:])
        

latent_vocab = LatentVocab()
