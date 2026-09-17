"""AI advisory layer: cloud LLM client + prompt builder + assistant."""
from algotrading.ai.client import AIClient, RecommendationError, extract_json
from algotrading.ai.assistant import Assistant
from algotrading.ai.prompt_builder import build_prompt

# Import Hermes client conditionally to avoid hard dependency
try:
    from algotrading.hermes_ai.client import HermesAgentClient
    HERMES_AI_AVAILABLE = True
except ImportError:
    HERMES_AI_AVAILABLE = False
    HermesAgentClient = None  # type: ignore

__all__ = ["AIClient", "Assistant", "RecommendationError", "extract_json", "build_prompt"]

if HERMES_AI_AVAILABLE:
    __all__.append("HermesAgentClient")