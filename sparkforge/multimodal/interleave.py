"""Media-token interleaving into the 1M-token context.

Muse Spark Open's context includes *media tokens*: a screenshot or a voice note
occupies a span of the context, not a single slot. We model media spans with a
per-media token budget so the KV cache can be sized correctly for mixed
text/image/audio/video sequences.
"""
from dataclasses import dataclass, field
from enum import Enum


class Modality(Enum):
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"


@dataclass
class MediaToken:
    modality: Modality
    payload_id: str
    n_tokens: int


@dataclass
class InterleavedSequence:
    """A sequence of text runs and media spans in order."""
    items: list = field(default_factory=list)

    def add_text(self, text):
        self.items.append({"kind": "text", "text": text})

    def add_media(self, modality, payload_id, n_tokens):
        self.items.append(MediaToken(Modality(modality), payload_id, n_tokens))

    def total_tokens(self, text_tokenizer=None):
        """Count total tokens: text tokens + media token spans."""
        total = 0
        for it in self.items:
            if isinstance(it, MediaToken):
                total += it.n_tokens
            else:
                if text_tokenizer is not None:
                    total += len(text_tokenizer(it["text"]))
                else:
                    total += max(1, len(it["text"].split()))
        return total

    def media_tokens(self):
        return [it for it in self.items if isinstance(it, MediaToken)]
