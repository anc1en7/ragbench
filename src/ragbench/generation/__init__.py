"""Answer generation: versioned prompt + OpenAI-compatible chat call (Phase 4)."""

from ragbench.generation.prompt import PROMPT_VERSION, build_messages, generate_answer

__all__ = ["PROMPT_VERSION", "build_messages", "generate_answer"]
