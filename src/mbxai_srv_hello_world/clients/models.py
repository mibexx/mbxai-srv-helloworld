"""OpenRouter model definitions."""

from enum import Enum


class OpenRouterModel(Enum):
    """Available models from OpenRouter.
    
    This enum can be extended or modified to add custom models as needed.
    """
    GPT_4O = "openai/gpt-4o"
    GPT_41 = "openai/gpt-4.1"
    GPT_4O_MINI = "openai/gpt-4o-mini"
    
    # Additional models can be added here
    # ANTHROPIC_CLAUDE_3_OPUS = "anthropic/claude-3-opus-20240229"
    # ANTHROPIC_CLAUDE_3_SONNET = "anthropic/claude-3-sonnet-20240229"
    # ANTHROPIC_CLAUDE_3_HAIKU = "anthropic/claude-3-haiku-20240307" 