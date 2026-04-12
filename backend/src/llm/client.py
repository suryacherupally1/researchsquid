"""
OpenAI-compatible LLM client.

Provides structured output (JSON mode with Pydantic parsing),
streaming, and budget tracking. Works with any OpenAI-compatible
endpoint — OpenAI, Anthropic via proxy, Ollama, vLLM, etc.

Supports per-persona model selection: agents with different model_tier
or model_name settings use different models through the same client.
"""

import json
import logging
import re
import time
from typing import Any, TypeVar

from openai import AsyncOpenAI
from pydantic import BaseModel

from src.config import Settings, settings as default_settings
from src.llm.pricing import get_pricing_manager

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class LLMClient:
    """
    Async LLM client with structured output support.

    Wraps the OpenAI client to provide:
    - Plain text completions
    - Structured JSON output parsed into Pydantic models
    - Budget tracking (call counting)
    - Configurable model and endpoint

    Usage:
        client = LLMClient(settings)
        response = await client.complete("What is 2+2?")
        parsed = await client.complete_structured(
            "Analyze this text",
            response_model=AnalysisResult,
        )
    """

    def __init__(self, config: Settings | None = None) -> None:
        self._config = config or default_settings
        self._client = AsyncOpenAI(
            api_key=self._config.openai_api_key,
            base_url=self._config.openai_api_base,
        )
        self._model = self._config.openai_model
        self._call_count = 0
        self._total_tokens = 0
        self._total_cost = 0.0
        self._last_usage: dict[str, Any] | None = None

    @property
    def call_count(self) -> int:
        """Number of LLM calls made through this client."""
        return self._call_count

    @property
    def total_tokens(self) -> int:
        """Total tokens used across all calls."""
        return self._total_tokens

    @property
    def total_cost(self) -> float:
        """Total cost in dollars (from OpenRouter total_cost or estimated)."""
        return self._total_cost

    @property
    def last_usage(self) -> dict[str, Any] | None:
        """Most recent usage data: tokens and cost."""
        return self._last_usage

    def reset_usage(self) -> None:
        """Reset all usage counters for a new session."""
        self._call_count = 0
        self._total_tokens = 0
        self._total_cost = 0.0
        self._last_usage = None

    async def _capture_usage(
        self, 
        response: Any, 
        usage_accumulator: dict[str, Any] | None = None,
        model_override: str | None = None
    ) -> dict[str, Any]:
        """Extract usage data from API response."""
        usage = response.usage
        if not usage:
            return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "cost": 0.0}

        # Resolve cost using the pricing manager
        manager = get_pricing_manager()
        model_to_price = model_override or self._model
        
        # Check if response has OpenRouter specific total_cost
        cost_val = getattr(usage, "total_cost", None)
        
        if cost_val is None:
            # Calculate estimate using ground truth or OR models API
            res = await manager.estimate_cost(
                model_name=model_to_price,
                usage_obj=usage,
                base_url=self._config.openai_api_base
            )
            cost_val = res.amount_usd

        cost = float(cost_val)
        
        usage_data = {
            "prompt_tokens": getattr(usage, "prompt_tokens", 0),
            "completion_tokens": getattr(usage, "completion_tokens", 0),
            "total_tokens": getattr(usage, "total_tokens", 0),
            "cost": cost,
        }
        
        self._total_tokens += usage_data["total_tokens"]
        self._total_cost += cost
        self._last_usage = usage_data

        if usage_accumulator is not None:
            usage_accumulator["prompt_tokens"] = usage_accumulator.get("prompt_tokens", 0) + usage_data["prompt_tokens"]
            usage_accumulator["completion_tokens"] = usage_accumulator.get("completion_tokens", 0) + usage_data["completion_tokens"]
            usage_accumulator["total_tokens"] = usage_accumulator.get("total_tokens", 0) + usage_data["total_tokens"]
            usage_accumulator["cost"] = usage_accumulator.get("cost", 0.0) + cost
        
        return usage_data

    def resolve_model_for_persona(self, persona: Any) -> str:
        """
        Pick the model name for a persona's tier/override.

        Resolution order:
        1. persona.model_name (explicit override, e.g. "gpt-4o")
        2. Tier mapping from config (fast_model, balanced_model, powerful_model)
        3. Default openai_model from config

        Args:
            persona: An AgentPersona (or any object with model_tier
                     and model_name attributes).

        Returns:
            The model name string to use for API calls.
        """
        # Explicit model override takes precedence
        if hasattr(persona, "model_name") and persona.model_name:
            return persona.model_name

        # Map tier to configured model
        tier = getattr(persona, "model_tier", "fast")
        tier_map = {
            "fast": self._config.fast_model or self._config.openai_model,
            "balanced": self._config.balanced_model or self._config.openai_model,
            "powerful": self._config.powerful_model or self._config.openai_model,
        }
        return tier_map.get(tier, self._config.openai_model)

    async def complete(
        self,
        prompt: str,
        system: str = "",
        temperature: float | None = None,
        max_tokens: int | None = None,
        usage_accumulator: dict[str, Any] | None = None,
        model_override: str | None = None,
    ) -> str:
        """
        Send a prompt and return the text response.

        Args:
            prompt: User message content.
            system: Optional system message for role/behavior guidance.
            temperature: Sampling temperature (0.0 = deterministic).
            max_tokens: Maximum response length.
            model_override: Use a specific model instead of the configured default.

        Returns:
            The assistant's text response.
        """
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        model = model_override or self._model
        response = await self._client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=(
                self._config.temperature_default_text
                if temperature is None else temperature
            ),
            max_tokens=(
                self._config.max_tokens_default
                if max_tokens is None else max_tokens
            ),
        )
        self._call_count += 1
        await self._capture_usage(response, usage_accumulator=usage_accumulator)
        return response.choices[0].message.content or ""

    async def complete_structured(
        self,
        prompt: str,
        response_model: type[T],
        system: str = "",
        temperature: float | None = None,
        max_tokens: int | None = None,
        max_retries: int | None = None,
        model_override: str | None = None,
        usage_accumulator: dict[str, Any] | None = None,
    ) -> T:
        """
        Send a prompt and parse the response into a Pydantic model.

        Uses JSON mode to get structured output from the LLM, then
        validates it against the provided Pydantic model. Retries
        with a conciseness instruction if the response is truncated.

        Args:
            prompt: User message content.
            response_model: Pydantic model class to parse the response into.
            system: Optional system message.
            temperature: Lower is better for structured output.
            max_tokens: Maximum response length.
            max_retries: Number of retry attempts on truncated/invalid JSON.

        Returns:
            An instance of response_model populated from the LLM's response.
        """
        schema_json = json.dumps(
            response_model.model_json_schema(), indent=2
        )
        structured_system = (
            f"{system}\n\n"
            f"You MUST respond with valid JSON matching this schema:\n"
            f"{schema_json}\n\n"
            f"IMPORTANT: Keep all string values concise (1-3 sentences max). "
            f"Lists should have at most 5 items. "
            f"Respond ONLY with the JSON object, no other text."
        )

        messages: list[dict[str, str]] = [
            {"role": "system", "content": structured_system},
            {"role": "user", "content": prompt},
        ]

        current_temperature = (
            self._config.temperature_default_structured
            if temperature is None else temperature
        )
        current_max_tokens = (
            self._config.max_tokens_default
            if max_tokens is None else max_tokens
        )
        retry_limit = (
            self._config.max_retries_structured
            if max_retries is None else max_retries
        )
        model = model_override or self._model

        last_error: Exception | None = None
        for attempt in range(1 + retry_limit):
            start_time = time.time()
            response = await self._client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=current_temperature,
                max_tokens=current_max_tokens,
                response_format={"type": "json_object"},
            )
            elapsed = time.time() - start_time
            self._call_count += 1
            await self._capture_usage(
                response, 
                usage_accumulator=usage_accumulator,
                model_override=model
            )

            raw = response.choices[0].message.content or "{}"

            # Check if response was truncated (hit max_tokens)
            finish_reason = response.choices[0].finish_reason

            try:
                normalized = self._normalize_structured_json(raw)
                parsed = response_model.model_validate(json.loads(normalized))
                logger.warning(
                    f"LLM structured call OK: model={model}, "
                    f"elapsed={elapsed:.2f}s, attempt={attempt+1}, "
                    f"finish_reason={finish_reason}, "
                    f"raw_len={len(raw)}"
                )
                return parsed
            except Exception as e:
                last_error = e
                logger.warning(
                    f"LLM structured parse FAILED: model={model}, "
                    f"attempt={attempt+1}/{retry_limit+1}, "
                    f"error={type(e).__name__}: {str(e)[:200]}, "
                    f"raw_preview={raw[:500]}"
                )
                if attempt < retry_limit:
                    retry_reason = (
                        "Your previous response was truncated."
                        if finish_reason == "length"
                        else "Your previous response was not valid bare JSON."
                    )
                    # Retry with stronger conciseness constraint
                    messages = [
                        {"role": "system", "content": structured_system},
                        {"role": "user", "content": (
                            f"{prompt}\n\n"
                            f"CRITICAL: {retry_reason} "
                            f"Return only a single raw JSON object with no markdown fences or explanation. "
                            f"Be MUCH more concise. Each string field max 2 sentences. "
                            f"Max 3 items per list. Keep total response under 1000 tokens."
                        )},
                    ]
                    # Increase token limit on retry
                    current_max_tokens = min(
                        current_max_tokens * self._config.llm_retry_token_multiplier,
                        self._config.max_tokens_retry_cap,
                    )

        raise last_error  # type: ignore[misc]

    async def complete_structured_for_persona(
        self,
        prompt: str,
        response_model: type[T],
        persona: Any,
        system: str = "",
        temperature: float | None = None,
        max_tokens: int | None = None,
        max_retries: int | None = None,
        usage_accumulator: dict[str, Any] | None = None,
    ) -> T:
        """
        Structured completion using the model resolved from a persona.

        Same as complete_structured but overrides the model based on
        the persona's model_tier or model_name. This lets different
        agents in the same session use different models (e.g., a
        "powerful" tier agent uses gpt-4o while "fast" agents use
        gpt-4o-mini).

        Args:
            prompt: User message content.
            response_model: Pydantic model class to parse into.
            persona: AgentPersona (needs model_tier, model_name attrs).
            system: Optional system message.
            temperature: Sampling temperature.
            max_tokens: Maximum response length.
            max_retries: Retry attempts on truncated/invalid JSON.

        Returns:
            An instance of response_model.
        """
        model = self.resolve_model_for_persona(persona)
        return await self.complete_structured(
            prompt=prompt,
            response_model=response_model,
            system=system,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=max_retries,
            model_override=model,
            usage_accumulator=usage_accumulator,
        )

    @staticmethod
    def _normalize_structured_json(raw: str) -> str:
        """
        Recover a JSON payload from common model formatting mistakes.

        Accepts:
        - bare JSON
        - fenced ```json ... ``` blocks
        - leading or trailing prose wrapped around the first object/array
        """
        text = (raw or "").strip()
        if not text:
            return "{}"

        fenced = re.match(
            r"^```(?:json)?\s*(.*?)\s*```$",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if fenced:
            text = fenced.group(1).strip()

        if text.startswith("{") or text.startswith("["):
            return text

        object_start = text.find("{")
        object_end = text.rfind("}")
        if object_start != -1 and object_end != -1 and object_end > object_start:
            return text[object_start:object_end + 1]

        array_start = text.find("[")
        array_end = text.rfind("]")
        if array_start != -1 and array_end != -1 and array_end > array_start:
            return text[array_start:array_end + 1]

        return text

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for a list of texts.

        Args:
            texts: Strings to embed.

        Returns:
            List of embedding vectors, one per input text.
        """
        logger.debug(f"Generating embeddings for {len(texts)} texts using {self._config.embedding_model}")
        response = await self._client.embeddings.create(
            model=self._config.embedding_model,
            input=texts,
        )
        results = [item.embedding for item in response.data]
        if not results:
            logger.warning(f"OpenAI returned empty embedding data for {len(texts)} inputs.")
        return results
