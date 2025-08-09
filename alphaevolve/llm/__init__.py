"""
LLM package for AlphaEvolveLite.

This package contains all LLM-related functionality including:
- LLM configuration classes
- Client pool for managing API connections
- LLM engine for conversation management and generation
"""

from .config import ModelCfg, LLMCfg
from .clients import ClientPool, global_client_pool
from .engine import LLMEngine, LLMAPIError

__all__ = [
    "ModelCfg",
    "LLMCfg", 
    "ClientPool",
    "global_client_pool",
    "LLMEngine",
    "LLMAPIError",
]