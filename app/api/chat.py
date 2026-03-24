"""
chat.py
AI Agronomist endpoints for Orbitani.

Dual-Model Strategy:
  POST /ask            → gemini-flash-lite-latest (fast, ~2-3s)  — Q&A chat
  POST /analyze-lahan  → gemini-2.5-flash (deep, ~15-20s) — satellite analysis

Rate Limiting:
  Role 'user' : 5 RPM  |  Role 'admin'/'superadmin' : unlimited
"""
import logging
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from supabase import Client

from app.db.database import get_supabase
from app.core.security import get_current_user
from app.core.rate_limiter import check_rate_limit
from app.services.gemini_service import ask_fast, ask_deep

logger = logging.getLogger(__name__)
router = APIRouter()


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=1000)


class AnalyzeLahanRequest(BaseModel):
    lahan_id: int


# ---------------------------------------------------------------
# POST /ask — Quick Q&A (gemini-flash-lite-latest)
# ---------------------------------------------------------------
@router.post("/ask")
async def ask_agronomist_api(
    req: ChatRequest,
    current_user: dict = Depends(get_current_user),
):
    """
    Konsultasi tanya jawab bebas dengan AI Agronomist.
    Menggunakan gemini-flash-lite-latest untuk respons cepat (~2-3 detik).
    Rate limit: 5 RPM untuk role 'user'.
    """
    check_rate_limit(current_user)

    logger.info("User '%s' asking fast AI: %s...", current_user["username"], req.message[:60])
    answer = await ask_fast(req.message)
    return {"status": "success", "model": "gemini-flash-lite-latest", "answer": answer}


# ---------------------------------------------------------------
# POST /analyze-lahan — Deep Analysis (gemini-2.5-flash)
# ---------------------------------------------------------------
@router.post("/analyze-lahan")
async def analyze_lahan_api(
    req: AnalyzeLahanRequest,
    db: Client = Depends(get_supabase),
    current_user: dict = Depends(get_current_user),
):
    """
    Menganalisis hasil satelit terbaru dari suatu lahan menggunakan
    gemini-2.5-flash untuk analisis mendalam (~15-20 detik).
    Rate limit: 5 RPM untuk role 'user'.
    """
    check_rate_limit(current_user)

    # 1. Ambil data satelit terbaru dari lahan ini
    result = (
        db.table("satellite_results")
        .select("*")
        .eq("lahan_id", req.lahan_id)
        .order("extracted_at", desc=True)
        .limit(1)
        .execute()
    )

    if not result.data:
        raise HTTPException(
            status_code=404,
            detail="Belum ada data satelit untuk lahan ini. Silakan sinkronisasi GEE terlebih dahulu.",
        )

    sat_data = result.data[0]

    # 2. Susun prompt agronomis dengan data satelit aktual
    prompt = f"""Tolong berikan Orbitani Smart Analysis untuk data lahan satelit terbaru ini:

- Nitrogen (N)      : {sat_data.get('n_value', 'N/A')}
- Fosfor (P)        : {sat_data.get('p_value', 'N/A')}
- Kalium (K)        : {sat_data.get('k_value', 'N/A')}
- pH Tanah          : {sat_data.get('ph', 'N/A')}
- Suhu Lahan        : {sat_data.get('temperature', 'N/A')} °C
- Curah Hujan (1th) : {sat_data.get('rainfall', 'N/A')} mm

Berikan 2 poin aksi yang harus segera dilakukan petani untuk memaksimalkan hasil panen Kenaf."""

    logger.info(
        "User '%s' requesting deep analysis for lahan_id=%d",
        current_user["username"], req.lahan_id,
    )
    answer = await ask_deep(prompt)

    return {
        "status": "success",
        "model": "gemini-2.5-flash",
        "satellite_data": sat_data,
        "ai_analysis": answer,
    }
