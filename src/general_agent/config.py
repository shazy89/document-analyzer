from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


@dataclass
class GeneralAgentConfig:
    model_name: str = field(default_factory=lambda: os.getenv("TOGETHER_MODEL", "meta-llama/Llama-3.3-70B-Instruct-Turbo"))
    api_key: str = field(default_factory=lambda: os.getenv("TOGETHER_API_KEY", ""))
    api_base: str = field(default_factory=lambda: os.getenv("TOGETHER_API_BASE", "https://api.together.xyz/v1"))
    temperature: float = field(default_factory=lambda: float(os.getenv("TOGETHER_TEMPERATURE", "0.0")))

    @classmethod
    def from_env(cls) -> GeneralAgentConfig:
        return cls()
