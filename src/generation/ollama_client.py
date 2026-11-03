"""Ollama local LLM API client for EduDocAI."""

from typing import List, Dict, Any, Optional
import requests

from config.settings import settings
from src.utils.logger import logger


class OllamaClient:
    """Client for local Ollama LLM inference with health-check and timeout safeguards."""

    _cached_available: Optional[bool] = None
    _last_check_time: float = 0.0

    def __init__(
        self,
        base_url: str = settings.OLLAMA_BASE_URL,
        model_name: str = settings.OLLAMA_DEFAULT_MODEL,
        timeout: int = settings.OLLAMA_TIMEOUT_SECONDS
    ):
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.timeout = timeout

    def is_available(self) -> bool:
        """Checks if local Ollama daemon is reachable (with fast 30s caching).

        Returns:
            bool: True if Ollama service responds, False otherwise.
        """
        import time as _t
        now = _t.time()
        # Cache probe for 30 seconds to prevent repeated connection timeout delays
        if OllamaClient._cached_available is not None and (now - OllamaClient._last_check_time) < 30.0:
            return OllamaClient._cached_available

        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=0.8)
            is_ok = response.status_code == 200
        except Exception:
            is_ok = False

        OllamaClient._cached_available = is_ok
        OllamaClient._last_check_time = now
        return is_ok

    def list_available_models(self) -> List[str]:
        """Fetches list of downloaded models in the local Ollama instance.

        Returns:
            List[str]: Model tags (e.g. ['llama3:8b', 'mistral:latest']).
        """
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=3)
            if response.status_code == 200:
                data = response.json()
                return [m.get("name", "") for m in data.get("models", []) if m.get("name")]
            return []
        except Exception as e:
            logger.debug(f"Could not retrieve Ollama models: {str(e)}")
            return []

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = settings.LLM_TEMPERATURE
    ) -> Optional[str]:
        """Calls Ollama generate endpoint synchronously.

        Args:
            prompt: User message including formatted retrieved context.
            system_prompt: Strict grounding system prompt.
            temperature: Sampling temperature (default 0.0 for deterministic grounding).

        Returns:
            Optional[str]: Generated text response, or None if failed.
        """
        payload: Dict[str, Any] = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature
            }
        }
        if system_prompt:
            payload["system"] = system_prompt

        try:
            logger.info(f"OllamaClient: Calling model '{self.model_name}' at {self.base_url}...")
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout
            )
            if response.status_code == 200:
                data = response.json()
                return data.get("response", "").strip()
            else:
                logger.warning(f"Ollama returned HTTP {response.status_code}: {response.text}")
                return None
        except requests.exceptions.Timeout:
            logger.error(f"Ollama request timed out after {self.timeout}s.")
            return None
        except requests.exceptions.ConnectionError:
            logger.info("Ollama is not running locally.")
            return None
        except Exception as e:
            logger.error(f"Error during Ollama generation: {str(e)}")
            return None
