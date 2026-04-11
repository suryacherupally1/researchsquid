"""
LLM Pricing Manager.

Provides model-aware cost estimation by matching token usage against
a ground-truth pricing manifest or live provider metadata.

Logic ported from hermes-agent to ensure high-accuracy budget tracking
without relying on generic fallbacks.
"""

import json
import logging
import time
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, Optional, Literal

import httpx

logger = logging.getLogger(__name__)

# Constants
ONE_MILLION = Decimal("1000000")
ZERO = Decimal("0")

@dataclass(frozen=True)
class PricingEntry:
    input_cost_per_million: Optional[Decimal] = None
    output_cost_per_million: Optional[Decimal] = None
    cache_read_cost_per_million: Optional[Decimal] = None
    cache_write_cost_per_million: Optional[Decimal] = None
    source: str = "none"

@dataclass(frozen=True)
class CanonicalUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

@dataclass(frozen=True)
class CostResult:
    amount_usd: Decimal
    status: Literal["actual", "estimated", "unknown"]
    source: str

class PricingManager:
    """
    Manages LLM pricing lookups and cost estimation.
    """

    def __init__(self, manifest_path: Optional[Path] = None):
        if manifest_path is None:
            manifest_path = Path(__file__).parent / "pricing.json"
        self._manifest_path = manifest_path
        self._ground_truth: Dict[str, Dict[str, PricingEntry]] = {}
        self._openrouter_cache: Dict[str, Dict[str, Any]] = {}
        self._openrouter_cache_time: float = 0
        self._load_manifest()

    def _load_manifest(self):
        """Load the ground truth JSON manifest into PricingEntry objects."""
        try:
            if not self._manifest_path.exists():
                logger.warning(f"Pricing manifest not found at {self._manifest_path}")
                return

            with open(self._manifest_path, "r") as f:
                data = json.load(f)

            for provider, models in data.items():
                self._ground_truth[provider] = {}
                for model_slug, rates in models.items():
                    self._ground_truth[provider][model_slug] = PricingEntry(
                        input_cost_per_million=Decimal(rates.get("input", "0")),
                        output_cost_per_million=Decimal(rates.get("output", "0")),
                        cache_read_cost_per_million=Decimal(rates.get("cache_read", "0")) if "cache_read" in rates else None,
                        cache_write_cost_per_million=Decimal(rates.get("cache_write", "0")) if "cache_write" in rates else None,
                        source="ground_truth"
                    )
        except Exception as e:
            logger.error(f"Failed to load pricing manifest: {e}")

    async def fetch_openrouter_metadata(self, force: bool = False) -> Dict[str, Any]:
        """Fetch live metadata from OpenRouter to support dynamic models."""
        now = time.time()
        if not force and self._openrouter_cache and (now - self._openrouter_cache_time) < 3600:
            return self._openrouter_cache

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get("https://openrouter.ai/api/v1/models")
                resp.raise_for_status()
                data = resp.json()
                
                new_cache = {}
                for model in data.get("data", []):
                    model_id = model.get("id")
                    if model_id:
                        new_cache[model_id] = model
                
                self._openrouter_cache = new_cache
                self._openrouter_cache_time = now
                return new_cache
        except Exception as e:
            logger.warning(f"Failed to fetch OpenRouter metadata: {e}")
            return self._openrouter_cache

    async def resolve_pricing(
        self, 
        model_name: str, 
        base_url: Optional[str] = None
    ) -> Optional[PricingEntry]:
        """
        Resolve a model to its pricing entry using ground truth or local inference.
        """
        # 0. Check if the full model name (e.g. "google/gemma-4-26b-a4b-it") 
        #    is in the manifest under "openrouter" provider
        if "openrouter" in self._ground_truth:
            if model_name in self._ground_truth["openrouter"]:
                return self._ground_truth["openrouter"][model_name]
            # Fuzzy match within openrouter section
            for m_slug, entry in self._ground_truth["openrouter"].items():
                if m_slug in model_name or model_name in m_slug:
                    return entry

        # 1. Infer provider from model name or base_url
        provider = "unknown"
        clean_model = model_name
        
        if "/" in model_name:
            # Handle "provider/model" format (OpenRouter style)
            parts = model_name.split("/", 1)
            provider = parts[0].lower()
            clean_model = parts[1]
        elif base_url:
            if "openai.com" in base_url:
                provider = "openai"
            elif "anthropic.com" in base_url:
                provider = "anthropic"
            elif "googleapis.com" in base_url:
                provider = "google"
            elif "deepseek.com" in base_url:
                provider = "deepseek"

        # 2. Check ground truth
        if provider in self._ground_truth:
            if clean_model in self._ground_truth[provider]:
                return self._ground_truth[provider][clean_model]
            
            # Try fuzzy match (e.g. gpt-4o-2024-05-13 vs gpt-4o)
            for m_slug, entry in self._ground_truth[provider].items():
                if m_slug in clean_model:
                    return entry

        # 3. Check OpenRouter cache
        # Fetch live pricing if it's not cached yet or expired
        await self.fetch_openrouter_metadata()
        
        #    OpenRouter API returns pricing in per-token values
        if model_name in self._openrouter_cache:
            meta = self._openrouter_cache[model_name]
            pricing = meta.get("pricing", {})
            # OR prices are per-token; convert to per-million for our internal format
            prompt_per_token = Decimal(str(pricing.get("prompt", "0")))
            completion_per_token = Decimal(str(pricing.get("completion", "0")))
            return PricingEntry(
                input_cost_per_million=prompt_per_token * ONE_MILLION,
                output_cost_per_million=completion_per_token * ONE_MILLION,
                source="openrouter_api"
            )

        return None

    def normalize_usage(self, response_usage: Any) -> CanonicalUsage:
        """
        Extract token counts from a raw API usage object into CanonicalUsage.
        Handles OpenAI-style details if present (e.g. cached_tokens).
        """
        if not response_usage:
            return CanonicalUsage()

        prompt_total = getattr(response_usage, "prompt_tokens", 0)
        completion_tokens = getattr(response_usage, "completion_tokens", 0)
        
        # Extract cache from OpenAI "prompt_tokens_details"
        cache_read = 0
        details = getattr(response_usage, "prompt_tokens_details", None)
        if details:
            # Standard OpenAI field
            cache_read = getattr(details, "cached_tokens", 0)
        
        # Determine actual 'new' prompt tokens (subtracting cache if the API includes it in total)
        # For OpenAI, prompt_tokens IS the total (including cache).
        # We want to price them separately if our manifest supports it.
        prompt_new = max(0, prompt_total - cache_read)

        return CanonicalUsage(
            prompt_tokens=prompt_new,
            completion_tokens=completion_tokens,
            cache_read_tokens=cache_read
        )

    async def estimate_cost(
        self,
        model_name: str,
        usage_obj: Any,
        base_url: Optional[str] = None
    ) -> CostResult:
        """
        Calculate USD cost based on raw usage object and resolved pricing.
        """
        usage = self.normalize_usage(usage_obj)
        entry = await self.resolve_pricing(model_name, base_url)
        
        if not entry:
            return CostResult(amount_usd=ZERO, status="unknown", source="none")

        amount = ZERO
        
        if entry.input_cost_per_million is not None:
            amount += (Decimal(usage.prompt_tokens) * entry.input_cost_per_million) / ONE_MILLION
            
        if entry.output_cost_per_million is not None:
            amount += (Decimal(usage.completion_tokens) * entry.output_cost_per_million) / ONE_MILLION

        # Optional cache pricing
        if usage.cache_read_tokens and entry.cache_read_cost_per_million is not None:
            amount += (Decimal(usage.cache_read_tokens) * entry.cache_read_cost_per_million) / ONE_MILLION
            
        if usage.cache_write_tokens and entry.cache_write_cost_per_million is not None:
            amount += (Decimal(usage.cache_write_tokens) * entry.cache_write_cost_per_million) / ONE_MILLION

        return CostResult(
            amount_usd=amount,
            status="estimated",
            source=entry.source
        )

# Global instances for reuse
_manager = None

def get_pricing_manager() -> PricingManager:
    global _manager
    if _manager is None:
        _manager = PricingManager()
    return _manager
