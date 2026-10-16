"""Capacity planning: 816B on unified memory."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sparkforge.memory.bandwidth import memory_footprint_gb, quantized_footprint_gb, tokens_per_second

total = 816.0
print("fp16   : %.1f GB" % memory_footprint_gb(total, 16))
print("nvfp4  : %.1f GB" % quantized_footprint_gb(total, 4))
print("fits 512GB unified pool (nvfp4):", quantized_footprint_gb(total, 4) < 512.0)
# active slice 42B fp16 = 84GB; stream misses at 1.2 TB/s
for hit in (0.0, 0.9, 0.99):
    print("tps @ hit=%.2f : %.2f" % (hit, tokens_per_second(1200.0, 84e9, hit)))
