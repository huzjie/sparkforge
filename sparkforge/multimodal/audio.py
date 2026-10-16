"""Frame-based audio encoder for voice notes / spoken requirements."""
from ..tensor import Tensor
from ..utils.hash import stable_vector


class AudioEncoder:
    def __init__(self, n_frames=8, frame_size=16, dim=32):
        self.n_frames = n_frames
        self.frame_size = frame_size
        self.dim = dim
        self.proj = Tensor([stable_vector("audio:proj:%d" % j, frame_size) for j in range(dim)])

    def encode(self, waveform):
        """waveform: list of n_frames*frame_size floats -> Tensor (n_frames x dim)."""
        frames = []
        for i in range(self.n_frames):
            seg = waveform[i * self.frame_size:(i + 1) * self.frame_size]
            if len(seg) < self.frame_size:
                seg = seg + [0.0] * (self.frame_size - len(seg))
            frames.append([sum(seg[k] * self.proj.data[j][k] for k in range(self.frame_size))
                           for j in range(self.dim)])
        return Tensor(frames)
