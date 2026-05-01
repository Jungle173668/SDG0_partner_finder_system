"""
LLM factory — returns a LangChain chat model based on environment config.

Supported providers:
  gemini — Google Gemini API (default, free tier). Set GOOGLE_API_KEY in .env
  groq   — Groq API (fast, generous free tier). Set GROQ_API_KEY in .env
  ollama — Local Ollama (offline). Run `ollama pull llama3.1:8b` first

Usage:
    from agent.llm import get_llm
    llm = get_llm()              # uses LLM_PROVIDER env var (default: gemini)
    llm = get_llm("groq")        # override

.env keys:
    LLM_PROVIDER=gemini
    GOOGLE_API_KEY=AIza...          # Get from aistudio.google.com/apikey (free)
    GEMINI_MODEL=gemini-2.0-flash   # or gemini-1.5-pro for higher quality

    GROQ_API_KEY=gsk_...            # Backup: console.groq.com (free)
    GROQ_MODEL=llama-3.1-8b-instant

    OLLAMA_MODEL=llama3.1:8b        # Local fallback
"""

import os
from typing import Optional
from functools import lru_cache


@lru_cache(maxsize=4)
def get_llm(provider: Optional[str] = None):
    """
    Return a cached LangChain chat model.

    Args:
        provider: "gemini" | "groq" | "ollama" | None
                  (reads LLM_PROVIDER env var, default "gemini")

    Returns:
        LangChain BaseChatModel
    """
    provider = provider or os.getenv("LLM_PROVIDER", "gemini")

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "GOOGLE_API_KEY not set. "
                "Get a free key at https://aistudio.google.com/apikey "
                "then add GOOGLE_API_KEY=... to .env"
            )
        return ChatGoogleGenerativeAI(
            model=os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
            google_api_key=api_key,
            temperature=0.3,
            max_output_tokens=512,
        )

    if provider == "groq":
        from langchain_groq import ChatGroq
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "GROQ_API_KEY not set. Get a free key at https://console.groq.com"
            )
        return ChatGroq(
            model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
            api_key=api_key,
            temperature=0.3,
            max_tokens=256,
        )

    if provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=os.getenv("OLLAMA_MODEL", "llama3.1:8b"),
            temperature=0.3,
            num_predict=512,
        )

    raise ValueError(
        f"Unknown LLM_PROVIDER '{provider}'. Set to 'gemini', 'groq', or 'ollama'."
    )
