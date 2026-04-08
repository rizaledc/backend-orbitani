"""
gemini_service.py
Round-Robin + BYOK Gemini AI service for Orbitani.

Model: gemini-3.1-flash-lite

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
# Model Constant
# ---------------------------------------------------------------------------
MODEL_NAME = "gemini-3.1-flash-lite"

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
SYSTEM_INSTRUCTION = """Anda adalah Pakar Agronomi Senior dan Ilmuwan Data utama untuk Proyek Orbitani. 
Tugas Anda adalah memberikan analisis teknis, ringkas, dan praktis mengenai kondisi hara tanah dan rekomendasi pemupukan.

KONTEKS KHUSUS:
1. LOKASI: Fokus utama adalah Lahan Hibisc di Bogor, Jawa Barat. Pahami karakteristik tanah latosol/podsolik merah kuning yang umum di Bogor (curah hujan tinggi, pH cenderung asam).
2. TANAMAN: Fokus pada optimasi tanaman Kenaf (Hibiscus cannabinus). Kenaf membutuhkan Nitrogen tinggi untuk serat dan Kalium untuk kekuatan batang.
3. DATA: Anda akan menerima data hara (N, P, K), pH, dan iklim dari satelit Sentinel-2 (resolusi 10m) dengan suhu permukaan dari Landsat-8/9 L2 dan MODIS Terra.

ATURAN JAWABAN:
- Berikan analisis dalam 2 poin kritis saja (sesuai permintaan sistem).
- Gunakan bahasa profesional namun mudah dimengerti pengguna modern.
- Format jawaban wajib menggunakan Markdown yang rapi.
- Jika pH di bawah 6.0, selalu sarankan pemberian kapur dolomit.
- Jika Nitrogen (N) rendah, fokus pada efisiensi pupuk Urea atau ZA.
- Gunakan terminologi "Orbitani Smart Analysis"."""


# ---------------------------------------------------------------------------
# Core: Per-Request Client + Generate
# ---------------------------------------------------------------------------
def _call_gemini(api_key: str, prompt: str) -> str:
    """
    Membuat genai.Client baru per-request (thread-safe), lalu memanggil model.
    Raise exception jika gagal.
    """
    from google import genai

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config={
            "system_instruction": SYSTEM_INSTRUCTION,
        },
    )
    return response.text


def _is_rate_limit_error(e: Exception) -> bool:
    """Cek apakah exception adalah 429 Rate Limit."""
    err_str = str(e).lower()
    return "429" in err_str or "resource_exhausted" in err_str or "rate limit" in err_str


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
