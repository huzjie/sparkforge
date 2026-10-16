"""Sampled-frame video encoder for screen recordings.

A screen recording is huge; we sample a handful of frames and encode each with a
shared vision encoder, then mean-pool into a compact video embedding.
"""
from ..tensor import Tensor
from .vision import VisionEncoder


class VideoEncoder:
    def __init__(self, n_frames=4, patch_size=8, img_size=32, dim=32):
        self.n_frames = n_frames
        self.vision = VisionEncoder(patch_size, img_size, dim, depth=1, heads=2)

    def encode(self, frames):
        """frames: list of n_frames images -> Tensor (n_frames x dim) mean-pooled."""
        per_frame = []
        for img in frames:
            emb = self.vision.encode(img)
            # mean pool over patches
            cols = len(emb.data[0])
            avg = [sum(row[c] for row in emb.data) / len(emb.data) for c in range(cols)]
            per_frame.append(avg)
        return Tensor(per_frame)
