"""Modular LLM provider abstraction for EduDocAI.

Supports:
1. OpenAI-compatible API (OpenAI, Groq, Together, DeepSeek, LocalAI, vLLM)
2. Local Ollama instance
3. Extractive zero-hallucination fallback (offline)
"""

import os
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import requests

from config.settings import settings
from src.generation.ollama_client import OllamaClient
from src.utils.logger import logger


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the provider is reachable and configured."""
        pass

    @abstractmethod
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> Optional[str]:
        """Generate text from prompt."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Display name of the provider."""
        pass


class OpenAICompatibleProvider(BaseLLMProvider):
    """Client for any OpenAI-compatible chat completion endpoint (OpenAI, Groq, Together, DeepSeek, etc.)."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: int = 45,
    ):
        raw_key = api_key or os.getenv("GROQ_API_KEY") or settings.OPENAI_API_KEY or os.getenv("OPENAI_API_KEY", "")
        self.api_key = raw_key.strip() if raw_key else ""

        # Auto-detect Groq keys
        is_groq = self.api_key.startswith("gsk_") or os.getenv("GROQ_API_KEY")
        if is_groq and not base_url:
            self.base_url = "https://api.groq.com/openai/v1"
            self.model = model or "llama-3.3-70b-versatile"
            self.provider_label = "Groq"
        else:
            self.base_url = (base_url or settings.OPENAI_BASE_URL or "https://api.openai.com/v1").rstrip("/")
            self.model = model or settings.OPENAI_MODEL or "gpt-4o-mini"
            self.provider_label = "OpenAI-Compatible"

        self.timeout = timeout

    @property
    def name(self) -> str:
        return f"{self.provider_label} ({self.model})"

    def is_available(self) -> bool:
        return bool(self.api_key and len(self.api_key.strip()) > 5)

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> Optional[str]:
        if not self.is_available():
            return None

        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key.strip()}",
            "Content-Type": "application/json",
        }
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": settings.LLM_TEMPERATURE,
        }

        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
            if resp.status_code == 200:
                data = resp.json()
                return data["choices"][0]["message"]["content"].strip()
            else:
                logger.warning(f"OpenAI API returned status {resp.status_code}: {resp.text[:200]}")
                return None
        except Exception as e:
            logger.warning(f"OpenAI API call failed: {e}")
            return None


class OllamaProvider(BaseLLMProvider):
    """Client for local Ollama server."""

    def __init__(self, client: Optional[OllamaClient] = None):
        self.client = client or OllamaClient()

    @property
    def name(self) -> str:
        return f"Ollama ({self.client.model_name})"

    def is_available(self) -> bool:
        return self.client.is_available()

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> Optional[str]:
        return self.client.generate(prompt=prompt, system_prompt=system_prompt)


def get_active_llm_provider() -> BaseLLMProvider:
    """Factory that returns the best available LLM provider in order of priority:
    1. OpenAI-compatible API (if API key is present)
    2. Local Ollama (if running)
    3. Null/fallback (extractive answering)
    """
    openai_prov = OpenAICompatibleProvider()
    if openai_prov.is_available():
        return openai_prov

    ollama_prov = OllamaProvider()
    if ollama_prov.is_available():
        return ollama_prov

    return None
