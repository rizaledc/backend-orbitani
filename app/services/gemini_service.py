"""
gemini_service.py
Dual-model Gemini AI service for Orbitani.
  - model_deep  : gemini-2.5-flash  → deep agronomist analysis (analyze-lahan)
  - model_fast  : gemini-1.5-flash  → quick chat / Q&A (ask)
"""
import os
import logging
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# ---------------------------------------------------------------------------
# System Instruction (shared by both models)
# ---------------------------------------------------------------------------
SYSTEM_INSTRUCTION = """Anda adalah Pakar Agronomi Senior dan Ilmuwan Data utama untuk Proyek Orbitani. 
Tugas Anda adalah memberikan analisis teknis, ringkas, dan praktis mengenai kondisi hara tanah dan rekomendasi pemupukan.

KONTEKS KHUSUS:
1. LOKASI: Fokus utama adalah Lahan Hibisc di Bogor, Jawa Barat. Pahami karakteristik tanah latosol/podsolik merah kuning yang umum di Bogor (curah hujan tinggi, pH cenderung asam).
2. TANAMAN: Fokus pada optimasi tanaman Kenaf (Hibiscus cannabinus). Kenaf membutuhkan Nitrogen tinggi untuk serat dan Kalium untuk kekuatan batang.
3. DATA: Anda akan menerima data hara (N, P, K), pH, dan iklim dari satelit Landsat 8.

ATURAN JAWABAN:
- Berikan analisis dalam 2 poin kritis saja (sesuai permintaan sistem).
- Gunakan bahasa profesional namun mudah dimengerti pengguna modern.
- Format jawaban wajib menggunakan Markdown yang rapi.
- Jika pH di bawah 6.0, selalu sarankan pemberian kapur dolomit.
- Jika Nitrogen (N) rendah, fokus pada efisiensi pupuk Urea atau ZA.
- Gunakan terminologi "Orbitani Smart Analysis"."""

model_deep = None   # gemini-2.5-flash — for /analyze-lahan
model_fast = None   # gemini-1.5-flash — for /ask

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

    # Deep Analysis model — accurate, slower
    model_deep = genai.GenerativeModel(
        model_name="gemini-2.5-flash",
        system_instruction=SYSTEM_INSTRUCTION,
        generation_config=genai.GenerationConfig(max_output_tokens=1024),
    )

    # Fast Chat model — quick response for Q&A
    model_fast = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        system_instruction=SYSTEM_INSTRUCTION,
        generation_config=genai.GenerationConfig(max_output_tokens=512),
    )

    logger.info("Gemini AI service initialized (deep=gemini-2.5-flash, fast=gemini-1.5-flash).")
else:
    logger.warning("GEMINI_API_KEY not found. Gemini AI service is disabled.")


async def ask_fast(prompt: str) -> str:
    """Prompt ke gemini-1.5-flash — untuk konsultasi chat cepat."""
    if not model_fast:
        return "Sistem AI saat ini tidak aktif (GEMINI_API_KEY tidak ditemukan)."
    try:
        response = await model_fast.generate_content_async(prompt)
        return response.text
    except Exception as e:
        logger.error("Error calling Gemini fast model: %s", e)
        return f"Maaf, terjadi kesalahan pada layanan AI: {e}"


async def ask_deep(prompt: str) -> str:
    """Prompt ke gemini-2.5-flash — untuk analisis lahan mendalam."""
    if not model_deep:
        return "Sistem AI saat ini tidak aktif (GEMINI_API_KEY tidak ditemukan)."
    try:
        response = await model_deep.generate_content_async(prompt)
        return response.text
    except Exception as e:
        logger.error("Error calling Gemini deep model: %s", e)
        return f"Maaf, terjadi kesalahan pada layanan AI: {e}"


# Backward-compatible alias (used by existing code if any)
async def ask_agronomist(prompt: str) -> str:
    return await ask_fast(prompt)
