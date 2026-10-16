"""Request/response dataclasses for the serving layer."""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ChatMessage:
    role: str
    content: str


@dataclass
class ChatCompletionRequest:
    messages: list = field(default_factory=list)
    max_tokens: int = 16
    temperature: float = 0.7


@dataclass
class ChatCompletionResponse:
    id: str
    content: str
    finish_reason: str = "stop"


@dataclass
class CodeRequest:
    task: str
    repo: str = "."
    symbols: list = field(default_factory=list)
