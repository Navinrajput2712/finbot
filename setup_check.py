"""
FinBot — setup_check.py
=======================
Run this after Hour 1 setup to verify everything is working.
Checks: folder structure, .env file, NVIDIA NIM API connection.

Usage:
    python setup_check.py
"""

import os
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

# ============================================================
# 1. CHECK FOLDER STRUCTURE
# ============================================================

def check_folders() -> bool:
    """
    Verify all required project folders exist.
    Creates missing folders automatically.
    
    Returns:
        bool: True if all folders exist or were created
    """
    required_folders = [
        "rag",
        "backend",
        "backend/routes",
        "frontend",
        "data/knowledge_base",
        "chroma_db",
        "evaluation",
    ]

    all_good = True
    for folder in required_folders:
        path = Path(folder)
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            logger.warning(f"📁 Created missing folder: {folder}")
        else:
            logger.info(f"✅ Folder exists: {folder}")

    return all_good


# ============================================================
# 2. CHECK .ENV FILE
# ============================================================

def check_env() -> bool:
    """
    Verify .env file exists and NVIDIA_API_KEY is set.
    
    Returns:
        bool: True if all required env vars are present
    """
    # Check .env file exists
    if not Path(".env").exists():
        logger.error("❌ .env file not found!")
        logger.error("   Run: cp .env.example .env")
        logger.error("   Then add your NVIDIA_API_KEY")
        return False
    else:
        logger.info("✅ .env file found")

    # Check required keys
    required_keys = [
        "NVIDIA_API_KEY",
        "NVIDIA_BASE_URL",
        "NVIDIA_MODEL",
        "CHROMA_DB_PATH",
        "KNOWLEDGE_BASE_PATH",
    ]

    all_good = True
    for key in required_keys:
        value = os.getenv(key)
        if not value:
            logger.error(f"❌ Missing env variable: {key}")
            all_good = False
        elif key == "NVIDIA_API_KEY" and value == "your_nvidia_nim_api_key_here":
            logger.error(f"❌ {key} is still the placeholder value — add your real key!")
            all_good = False
        else:
            # Mask API key for security
            display = value[:8] + "..." if "KEY" in key else value
            logger.info(f"✅ {key} = {display}")

    return all_good


# ============================================================
# 3. CHECK NVIDIA NIM API CONNECTION
# ============================================================

def check_nvidia_nim() -> bool:
    """
    Make a real test call to NVIDIA NIM API with llama-3.1-8b-instruct.
    Confirms API key is valid and model is reachable.
    
    Returns:
        bool: True if API call succeeds
    """
    try:
        from openai import OpenAI

        api_key = os.getenv("NVIDIA_API_KEY")
        base_url = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
        model = os.getenv("NVIDIA_MODEL", "meta/llama-3.1-8b-instruct")

        logger.info(f"🔗 Connecting to NVIDIA NIM API...")
        logger.info(f"   Model: {model}")
        logger.info(f"   Base URL: {base_url}")

        client = OpenAI(
            base_url=base_url,
            api_key=api_key
        )

        # Simple test message
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are FinBot, an AI financial advisor."
                },
                {
                    "role": "user",
                    "content": "Say 'FinBot is ready!' in exactly 5 words."
                }
            ],
            temperature=0.1,
            max_tokens=50,
        )

        answer = response.choices[0].message.content.strip()
        logger.info(f"✅ NVIDIA NIM API connected!")
        logger.info(f"   Model response: {answer}")
        return True

    except ImportError:
        logger.error("❌ openai library not installed!")
        logger.error("   Run: pip install openai")
        return False

    except Exception as e:
        logger.error(f"❌ NVIDIA NIM API connection failed: {str(e)}")
        logger.error("   Check your NVIDIA_API_KEY in .env file")
        logger.error("   Get free key at: https://build.nvidia.com")
        return False


# ============================================================
# 4. CHECK PYTHON PACKAGES
# ============================================================

def check_packages() -> bool:
    """
    Verify all critical Python packages are installed.
    
    Returns:
        bool: True if all packages are importable
    """
    packages = {
        "openai": "openai",
        "langchain": "langchain",
        "chromadb": "chromadb",
        "sentence_transformers": "sentence-transformers",
        "fastapi": "fastapi",
        "streamlit": "streamlit",
        "uvicorn": "uvicorn",
        "pydantic": "pydantic",
        "dotenv": "python-dotenv",
        "yfinance": "yfinance",
        "fitz": "pymupdf",
    }

    all_good = True
    for import_name, package_name in packages.items():
        try:
            __import__(import_name)
            logger.info(f"✅ Package installed: {package_name}")
        except ImportError:
            logger.error(f"❌ Package missing: {package_name}")
            logger.error(f"   Run: pip install {package_name}")
            all_good = False

    return all_good


# ============================================================
# 5. MAIN — RUN ALL CHECKS
# ============================================================

def main():
    """Run all setup checks and print final summary."""

    print("\n" + "="*60)
    print("   FINBOT — HOUR 1 SETUP CHECK")
    print("="*60 + "\n")

    results = {
        "Folder Structure": check_folders(),
        "Environment Variables": check_env(),
        "Python Packages": check_packages(),
        "NVIDIA NIM API": check_nvidia_nim(),
    }

    print("\n" + "="*60)
    print("   SETUP CHECK SUMMARY")
    print("="*60)

    all_passed = True
    for check_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status} — {check_name}")
        if not passed:
            all_passed = False

    print("="*60)

    if all_passed:
        print("\n  🎉 Hour 1 Complete — All systems ready!")
        print("  👉 Next: Start Hour 2 (RAG Ingestion Pipeline)\n")
        sys.exit(0)
    else:
        print("\n  ⚠️  Some checks failed. Fix errors above before Hour 2.\n")
        sys.exit(1)


if __name__ == "__main__":
    main()