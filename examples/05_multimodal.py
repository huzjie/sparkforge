"""Build an interleaved multimodal sequence."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sparkforge.multimodal.interleave import InterleavedSequence

seq = InterleavedSequence()
seq.add_text("Here is the screenshot of the broken UI:")
seq.add_media("image", "screenshot-001", 128)
seq.add_text("and here is the voice note with the requirement:")
seq.add_media("audio", "voice-001", 96)
seq.add_text("Fix the layout.")
print("total tokens:", seq.total_tokens())
print("media spans:", [(m.modality.value, m.n_tokens) for m in seq.media_tokens()])
