"""
gemini_service.py
Dual-model Gemini AI service for Orbitani.
  - model_deep  : gemini-2.5-flash             → deep agronomist analysis (analyze-lahan)
  - model_fast  : gemini-flash-lite-latest → quick chat / Q&A (ask)

SDK: google-genai (baru) — menggunakan `from google import genai`
"""
import os
import logging
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# Strip whitespace — Azure env vars kadang mengandung spasi/newline tersembunyi
GEMINI_API_KEY = (os.getenv("GEMINI_API_KEY") or "").strip() or None

# ---------------------------------------------------------------------------
# System Instruction (shared by both models)
# ---------------------------------------------------------------------------
SYSTEM_INSTRUCTION = """Anda adalah Pakar Agronomi Senior dan Ilmuwan Data utama untuk Proyek Orbitani. 
Tugas Anda adalah memberikan analisis teknis, ringkas, dan praktis mengenai kondisi hara tanah dan rekomendasi pemupukan.

KONTEKS KHUSUS:
1. LOKASI: Fokus utama adalah Lahan Hibisc di Bogor, Jawa Barat. Pahami karakteristik tanah latosol/podsolik merah kuning yang umum di Bogor (curah hujan tinggi, pH cenderung asam).
2. TANAMAN: Fokus pada optimasi tanaman Kenaf (Hibiscus cannabinus). Kenaf membutuhkan Nitrogen tinggi untuk serat dan Kalium untuk kekuatan batang.
3. DATA: Anda akan menerima data hara (N, P, K), pH, dan iklim dari satelit Sentinel-2 (resolusi 10m) dengan suhu permukaan dari Landsat-9 L2.

ATURAN JAWABAN:
- Berikan analisis dalam 2 poin kritis saja (sesuai permintaan sistem).
- Gunakan bahasa profesional namun mudah dimengerti pengguna modern.
- Format jawaban wajib menggunakan Markdown yang rapi.
- Jika pH di bawah 6.0, selalu sarankan pemberian kapur dolomit.
- Jika Nitrogen (N) rendah, fokus pada efisiensi pupuk Urea atau ZA.
- Gunakan terminologi "Orbitani Smart Analysis"."""

# ---------------------------------------------------------------------------
# Inisialisasi Client (SDK baru: google-genai)
# ---------------------------------------------------------------------------
client = None

if GEMINI_API_KEY:
    try:
        from google import genai

        # Diagnostic: log masked API key
        masked = GEMINI_API_KEY[:8] + "..." + GEMINI_API_KEY[-4:]
        logger.info("GEMINI_API_KEY detected: %s (len=%d)", masked, len(GEMINI_API_KEY))

        client = genai.Client(api_key=GEMINI_API_KEY)
        logger.info("Gemini Client initialized (google-genai SDK).")
    except Exception as e:
        logger.error("Failed to initialize Gemini Client: %s", e)
        print(f"ERROR GEMINI INIT: {e}")
        client = None
else:
    logger.warning("GEMINI_API_KEY not found. Gemini AI service is disabled.")


# ---------------------------------------------------------------------------
# Fast Chat — gemini-flash-lite-latest
# ---------------------------------------------------------------------------
async def ask_fast(prompt: str) -> str:
    """Prompt ke gemini-flash-lite-latest — untuk konsultasi chat cepat."""
    if not client:
        logger.warning("ask_fast called but client is None (no API key).")
        return "Sistem AI saat ini tidak aktif (GEMINI_API_KEY tidak ditemukan)."
    try:
        logger.info("Calling gemini-flash-lite-latest...")
        response = client.models.generate_content(
            model="gemini-flash-lite-latest",
            contents=prompt,
            config={
                "system_instruction": SYSTEM_INSTRUCTION,
                "max_output_tokens": 512,
            },
        )
        return response.text
    except Exception as e:
        print(f"ERROR GEMINI FAST: {e}")
        logger.error("Error calling Gemini fast model (type=%s): %s", type(e).__name__, e)
        return f"Maaf, terjadi kesalahan pada layanan AI [{type(e).__name__}]: {e}"


# ---------------------------------------------------------------------------
# Deep Analysis — gemini-2.5-flash
# ---------------------------------------------------------------------------
async def ask_deep(prompt: str) -> str:
    """Prompt ke gemini-2.5-flash — untuk analisis lahan mendalam."""
    if not client:
        return "Sistem AI saat ini tidak aktif (GEMINI_API_KEY tidak ditemukan)."
    try:
        logger.info("Calling gemini-2.5-flash...")
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config={
                "system_instruction": SYSTEM_INSTRUCTION,
                "max_output_tokens": 1024,
            },
        )
        return response.text
    except Exception as e:
        print(f"ERROR GEMINI DEEP: {e}")
        logger.error("Error calling Gemini deep model (type=%s): %s", type(e).__name__, e)
        return f"Maaf, terjadi kesalahan pada layanan AI [{type(e).__name__}]: {e}"


# Backward-compatible alias
async def ask_agronomist(prompt: str) -> str:
    return await ask_fast(prompt)
