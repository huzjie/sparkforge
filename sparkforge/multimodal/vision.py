"""Patch-based vision encoder (ViT-style) for screenshots / UI designs.

Turns an image (list of rows of pixel values) into patch embeddings, then a tiny
attention stack. The mock uses deterministic projections; the point is the
pipeline shape (patch -> embed -> encode), not real learned weights.
"""
from ..tensor import Tensor, matmul, softmax_rows, layernorm
from ..utils.hash import stable_vector


class VisionEncoder:
    def __init__(self, patch_size=8, img_size=32, dim=32, depth=2, heads=4):
        self.patch_size = patch_size
        self.img_size = img_size
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.n_patches = (img_size // patch_size) ** 2
        # projection maps a patch (patch_size^2) -> dim, so: dim rows x patch_size^2 cols
        self.proj = Tensor([stable_vector("vision:proj:%d" % j, patch_size * patch_size) for j in range(dim)])

    def patchify(self, image):
        """image: list of img_size rows of img_size floats -> list of patch vectors."""
        ps = self.patch_size
        patches = []
        for r0 in range(0, self.img_size, ps):
            for c0 in range(0, self.img_size, ps):
                flat = []
                for r in range(r0, r0 + ps):
                    for c in range(c0, c0 + ps):
                        flat.append(image[r][c])
                patches.append(flat)
        return patches

    def encode(self, image):
        """image -> Tensor (n_patches x dim)."""
        patches = self.patchify(image)
        embs = []
        for p in patches:
            embs.append([sum(p[k] * self.proj.data[j][k] for k in range(len(p))) for j in range(self.dim)])
        x = Tensor(embs)
        for _ in range(self.depth):
            # simplified single-head attention over patches
            q = k = v = x
            scores = matmul(q, Tensor(list(zip(*k.data))))
            attn = softmax_rows(scores)
            ctx = matmul(attn, v)
            x = layernorm(x + ctx)
        return x
