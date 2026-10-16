"""Vision / audio / video / interleave."""
import unittest

from sparkforge.multimodal.vision import VisionEncoder
from sparkforge.multimodal.audio import AudioEncoder
from sparkforge.multimodal.video import VideoEncoder
from sparkforge.multimodal.interleave import InterleavedSequence, Modality
from sparkforge.multimodal.fusion import GatedFusion


class TestMultimodal(unittest.TestCase):
    def test_vision_patches(self):
        enc = VisionEncoder(patch_size=8, img_size=16, dim=16, depth=1, heads=2)
        img = [[0.0] * 16 for _ in range(16)]
        emb = enc.encode(img)
        self.assertEqual(emb.shape, (4, 16))  # (16/8)^2 patches

    def test_audio_frames(self):
        enc = AudioEncoder(n_frames=4, frame_size=16, dim=16)
        emb = enc.encode([0.0] * (4 * 16))
        self.assertEqual(emb.shape, (4, 16))

    def test_video_frames(self):
        enc = VideoEncoder(n_frames=2, patch_size=8, img_size=16, dim=16)
        emb = enc.encode([[[0.0] * 16 for _ in range(16)] for _ in range(2)])
        self.assertEqual(emb.shape, (2, 16))

    def test_interleave_counts_media_tokens(self):
        seq = InterleavedSequence()
        seq.add_text("hello world")
        seq.add_media("image", "screenshot-1", 128)
        seq.add_text("fix this")
        self.assertEqual(seq.total_tokens(), 2 + 128 + 2)
        self.assertEqual(len(seq.media_tokens()), 1)

    def test_fusion_blends(self):
        from sparkforge.tensor import Tensor
        f = GatedFusion(16)
        out = f.fuse(Tensor([[0.1] * 16]), Tensor([[0.2] * 16]))
        self.assertEqual(out.shape, (1, 16))


if __name__ == "__main__":
    unittest.main()
