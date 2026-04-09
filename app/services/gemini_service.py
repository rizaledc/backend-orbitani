"""
gemini_service.py
Round-Robin + BYOK Gemini AI service for Orbitani.

Model: gemini-2.5-flash

Strategi API Key:
  Prioritas 1 (BYOK): Jika user_api_key dikirim, gunakan key tersebut.
                       Jika gagal, kembalikan error (TIDAK fallback ke pool).
  Prioritas 2 (Round-Robin): Jika user_api_key = None, gunakan GEMINI_FALLBACK_KEYS.
                              Jika key saat ini terkena 429, geser ke key berikutnya.

Thread Safety:
  Setiap request membuat genai.Client() sendiri → tidak ada race condition antar request.

SDK: google-genai (baru) — menggunakan `from google import genai`
"""
import os
import logging
import threading
from typing import Optional

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model Fallback List (Prioritas Tertinggi → Terendah)
# ---------------------------------------------------------------------------
FALLBACK_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
]

# Model utama (untuk response metadata & backward-compat)
MODEL_NAME = FALLBACK_MODELS[0]

# Backward-compatible aliases
MODEL_FAST = MODEL_NAME
MODEL_DEEP = MODEL_NAME

# ---------------------------------------------------------------------------
# API Key Pool (Round-Robin)
# ---------------------------------------------------------------------------
GEMINI_FALLBACK_KEYS: list[str] = []
for i in range(1, 6):
    key = (os.getenv(f"GEMINI_API_KEY{i}") or "").strip()
    if key:
        GEMINI_FALLBACK_KEYS.append(key)

# Backward compat: juga cek GEMINI_API_KEY lama (tanpa angka)
_legacy_key = (os.getenv("GEMINI_API_KEY") or "").strip()
if _legacy_key and _legacy_key not in GEMINI_FALLBACK_KEYS:
    GEMINI_FALLBACK_KEYS.insert(0, _legacy_key)

logger.info("Gemini key pool loaded: %d keys available.", len(GEMINI_FALLBACK_KEYS))

# Thread-safe round-robin index
_rr_index = 0
_rr_lock = threading.Lock()


def _next_rr_index() -> int:
    """Geser indeks round-robin secara atomic."""
    global _rr_index
    with _rr_lock:
        idx = _rr_index
        _rr_index = (_rr_index + 1) % max(len(GEMINI_FALLBACK_KEYS), 1)
        return idx


# ---------------------------------------------------------------------------
# System Instruction (shared)
# ---------------------------------------------------------------------------
SYSTEM_INSTRUCTION = """Kamu adalah AI Agronomist dari Orbitani. Jawab pertanyaan pengguna dengan ramah dan profesional. ATURAN FORMATTING MUTLAK: Jangan pernah menggunakan format LaTeX, Markdown Math, atau simbol equation ($ atau $$) untuk angka, pecahan, atau rumus. Tuliskan semua angka, dosis, dan perhitungan menggunakan teks biasa yang mudah dibaca (contoh: gunakan '1/2' bukan pecahan LaTeX, gunakan 'derajat Celcius' atau 'C' biasa). Jangan buat tabel yang rumit, gunakan list/bullet point biasa."""


# ---------------------------------------------------------------------------
# Helper: Deteksi jenis error
# ---------------------------------------------------------------------------
def _is_server_error(e: Exception) -> bool:
    """Cek apakah exception adalah 5xx Server Error (503, 500, overloaded, dll.)."""
    err_str = str(e).lower()
    return (
        "503" in err_str
        or "500" in err_str
        or "overloaded" in err_str
        or "service unavailable" in err_str
        or "internal server error" in err_str
        or "internal_error" in err_str
    )


def _is_rate_limit_error(e: Exception) -> bool:
    """Cek apakah exception adalah 429 Rate Limit."""
    err_str = str(e).lower()
    return "429" in err_str or "resource_exhausted" in err_str or "rate limit" in err_str


# ---------------------------------------------------------------------------
# Core: Per-Request Client + Generate (dengan Model Fallback Berlapis)
# ---------------------------------------------------------------------------
def _call_gemini(api_key: str, prompt: str) -> str:
    """
    Membuat genai.Client baru per-request (thread-safe), lalu memanggil model.
    Jika model utama terkena 5xx (server overload), otomatis fallback ke model
    berikutnya dalam daftar FALLBACK_MODELS.
    Raise exception jika semua model gagal atau error bukan 5xx.
    """
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_INSTRUCTION,
    )

    last_error = None

    for model_name in FALLBACK_MODELS:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=config,
            )
            if model_name != FALLBACK_MODELS[0]:
                logger.info("Berhasil dengan model fallback: %s", model_name)
            return response.text

        except Exception as e:
            last_error = e
            if _is_server_error(e):
                # Model ini sedang down/overloaded → coba model berikutnya
                next_idx = FALLBACK_MODELS.index(model_name) + 1
                next_model = FALLBACK_MODELS[next_idx] if next_idx < len(FALLBACK_MODELS) else "NONE"
                logger.warning(
                    "%s down (%s), falling back to %s...",
                    model_name, type(e).__name__, next_model,
                )
                continue
            else:
                # Error non-5xx (429, invalid key, dll.) → lempar ke caller
                raise

    # Semua model dalam daftar gagal (5xx semua)
    logger.error(
        "Semua %d model fallback gagal (server overload). Last error: %s",
        len(FALLBACK_MODELS), last_error,
    )
    from fastapi import HTTPException
    raise HTTPException(
        status_code=503,
        detail="Server sedang sibuk, tunggu beberapa saat lagi",
    )


# ---------------------------------------------------------------------------
# Public API: ask_fast & ask_deep
# ---------------------------------------------------------------------------
async def ask_fast(prompt: str, user_api_key: Optional[str] = None) -> str:
    """
    Quick Q&A chat menggunakan gemini-3.1-flash-lite.
    Mendukung BYOK (Bring Your Own Key) dan Round-Robin fallback.
    """
    return await _ask(prompt, user_api_key)


async def ask_deep(prompt: str, user_api_key: Optional[str] = None) -> str:
    """
    Deep analysis menggunakan gemini-3.1-flash-lite.
    Mendukung BYOK (Bring Your Own Key) dan Round-Robin fallback.
    """
    return await _ask(prompt, user_api_key)


async def _ask(prompt: str, user_api_key: Optional[str] = None) -> str:
    """
    Logika utama pemanggilan Gemini dengan hierarki:
      1. BYOK: Jika user_api_key ada, gunakan langsung. Error = kembalikan pesan error.
      2. Round-Robin: Jika None, iterasi pool keys. 429 = geser ke key berikutnya.
    """
    # --- Prioritas 1: BYOK ---
    if user_api_key:
        user_api_key = user_api_key.strip()
        logger.info("BYOK mode: menggunakan API key dari user (len=%d)", len(user_api_key))
        try:
            return _call_gemini(user_api_key, prompt)
        except Exception as e:
            logger.error("BYOK key gagal: %s", e)
            return f"API Key Anda gagal: [{type(e).__name__}] {e}"

    # --- Prioritas 2: Round-Robin Pool ---
    if not GEMINI_FALLBACK_KEYS:
        return "Sistem AI tidak aktif (tidak ada API Key yang dikonfigurasi)."

    total_keys = len(GEMINI_FALLBACK_KEYS)
    start_idx = _next_rr_index()
    last_error = None

    for attempt in range(total_keys):
        idx = (start_idx + attempt) % total_keys
        key = GEMINI_FALLBACK_KEYS[idx]
        masked = key[:8] + "..." + key[-4:]
        logger.info("Round-Robin attempt %d/%d menggunakan key index=%d (%s)", attempt + 1, total_keys, idx, masked)

        try:
            result = _call_gemini(key, prompt)
            logger.info("Berhasil dengan key index=%d", idx)
            return result
        except Exception as e:
            last_error = e
            if _is_rate_limit_error(e):
                logger.warning("Key index=%d terkena rate limit (429). Geser ke key berikutnya...", idx)
                continue
            else:
                # Error non-429 (invalid key, network error, dll.) — langsung kembalikan
                logger.error("Key index=%d error non-429: %s", idx, e)
                return f"Maaf, terjadi kesalahan pada layanan AI [{type(e).__name__}]: {e}"

    # Semua key habis terkena 429
    logger.error("Semua %d API Key terkena rate limit! Last error: %s", total_keys, last_error)
    return f"Maaf, semua API Key sedang terbatas (rate limit). Silakan coba lagi dalam beberapa menit, atau gunakan API Key pribadi Anda."


# Backward-compatible alias
async def ask_agronomist(prompt: str) -> str:
    return await ask_fast(prompt)
