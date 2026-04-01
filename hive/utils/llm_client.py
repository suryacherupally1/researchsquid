"""LLM client — OpenAI-compatible API wrapper for all HiveResearch LLM calls."""

import os
from typing import Any, Dict, List, Optional


class LLMClient:
    """
    Unified LLM client using OpenAI-compatible API.

    Supports any provider that implements the OpenAI chat completions format:
    - Anthropic (via OpenAI-compatible proxy)
    - OpenAI
    - Alibaba Qwen
    - Local models (Ollama, vLLM)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
    ):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY") or os.getenv("LLM_API_KEY", "")
        self.base_url = base_url or os.getenv("LLM_BASE_URL", "https://api.anthropic.com/v1")
        self.model = model or os.getenv("LLM_MODEL_NAME", "claude-haiku-4-5")
        self._client = None

    @property
    def client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
            )
        return self._client

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        model: Optional[str] = None,
    ) -> str:
        """
        Synchronous chat completion.

        Args:
            messages: [{"role": "system"|"user"|"assistant", "content": "..."}]
            temperature: 0.0-1.0
            max_tokens: max response tokens
            model: override default model

        Returns:
            Response text
        """
        response = self.client.chat.completions.create(
            model=model or self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content

    async def achat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        model: Optional[str] = None,
    ) -> str:
        """Async chat completion."""
        from openai import AsyncOpenAI
        client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )
        response = await client.chat.completions.create(
            model=model or self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content


# Singleton
_llm_client: Optional[LLMClient] = None


def get_llm_client() -> LLMClient:
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClient()
    return _llm_client
