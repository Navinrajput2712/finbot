"""
FinBot — backend/llm_loader.py
================================
NVIDIA NIM client initializer and connection tester.
Loads once at startup and reused across all requests.

Usage:
    from backend.llm_loader import get_nim_client, test_nim_connection
"""

import os
import logging
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

NVIDIA_API_KEY  = os.getenv("NVIDIA_API_KEY")
NVIDIA_BASE_URL = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
NVIDIA_MODEL    = os.getenv("NVIDIA_MODEL", "meta/llama-3.1-8b-instruct")


def get_nim_client() -> OpenAI:
    """
    Initialize and return NVIDIA NIM OpenAI-compatible client.

    Returns:
        OpenAI client pointed at NVIDIA NIM base URL

    Raises:
        ValueError: If NVIDIA_API_KEY is missing from .env
    """
    if not NVIDIA_API_KEY:
        raise ValueError(
            "NVIDIA_API_KEY not found!\n"
            "Add it to your .env file.\n"
            "Get free key at: https://build.nvidia.com"
        )

    client = OpenAI(
        base_url=NVIDIA_BASE_URL,
        api_key=NVIDIA_API_KEY
    )
    logger.info(f"NVIDIA NIM client initialized — model: {NVIDIA_MODEL}")
    return client


def test_nim_connection() -> bool:
    """
    Send a simple test message to verify NVIDIA NIM API is reachable.

    Returns:
        True if connection successful, False otherwise
    """
    try:
        client = get_nim_client()
        response = client.chat.completions.create(
            model=NVIDIA_MODEL,
            messages=[{"role": "user", "content": "Reply with: OK"}],
            max_tokens=10,
            temperature=0.1,
        )
        reply = response.choices[0].message.content.strip()
        logger.info(f"✅ NVIDIA NIM connection test passed — reply: {reply}")
        return True

    except Exception as e:
        logger.error(f"❌ NVIDIA NIM connection test failed: {str(e)}")
        return False
