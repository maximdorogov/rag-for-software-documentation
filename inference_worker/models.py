from typing import Set
from dataclasses import dataclass

@dataclass
class LLMResponse:
    docs: Set[str]
    answer: str