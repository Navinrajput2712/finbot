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

def get_nim_client() -> OpenAI:
    """
    Initialize and return NVIDIA NIM OpenAI-compatible client.

    Returns:
        OpenAI client pointed at NVIDIA NIM base URL

    Raises:
        ValueError: If NVIDIA_API_KEY is missing from .env
    """
    api_key = os.getenv("NVIDIA_API_KEY")
    base_url = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
    model = os.getenv("NVIDIA_MODEL", "meta/llama-3.1-8b-instruct")

    if not api_key:
        raise ValueError(
            "NVIDIA_API_KEY not found!\n"
            "Add it to your environment variables or .env file.\n"
            "Get free key at: https://build.nvidia.com"
        )

    client = OpenAI(
        base_url=base_url,
        api_key=api_key
    )
    logger.info(f"NVIDIA NIM client initialized — model: {model}")
    return client


def test_nim_connection() -> bool:
    """
    Send a simple test message to verify NVIDIA NIM API is reachable.

    Returns:
        True if connection successful, False otherwise
    """
    try:
        client = get_nim_client()
        model = os.getenv("NVIDIA_MODEL", "meta/llama-3.1-8b-instruct")
        response = client.chat.completions.create(
            model=model,
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
