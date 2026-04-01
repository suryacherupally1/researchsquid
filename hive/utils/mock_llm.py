"""Mock LLM client for testing — returns deterministic responses."""

from typing import Any, Dict, List, Optional


class MockLLMClient:
    """
    Mock LLM that returns deterministic responses for testing.
    No API calls, no dependencies.
    """

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        model: Optional[str] = None,
    ) -> str:
        last_msg = messages[-1].get("content", "") if messages else ""

        if "interview" in last_msg.lower() or "why" in last_msg.lower():
            return (
                "Based on my findings, I believe this because I found two independent "
                "tier-1 sources supporting this claim. My confidence is 0.85 based on "
                "peer-reviewed evidence. I would revise my position if a replicated RCT "
                "showed contradictory results."
            )

        if "outline" in last_msg.lower() or "plan" in last_msg.lower():
            return (
                "1. Executive Summary\n"
                "2. Key Findings\n"
                "3. Experiment Results\n"
                "4. Limitations\n"
                "5. Sources"
            )

        if "hypothesis" in last_msg.lower():
            return (
                "My current hypothesis is that naproxen is preferable for chronic pain "
                "due to its longer half-life (12-17h vs 2h for ibuprofen). "
                "This is supported by two PubMed sources."
            )

        return "Mock response: I have analyzed the available evidence and formed my conclusion based on the findings provided."

    async def achat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        model: Optional[str] = None,
    ) -> str:
        return self.chat(messages, temperature, max_tokens, model)
