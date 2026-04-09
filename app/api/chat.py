"""
chat.py
AI Agronomist endpoints for Orbitani.
Model: gemini-2.5-flash (Round-Robin + BYOK)
Endpoints:
  POST /ask            → quick Q&A chat
  POST /analyze-lahan  → satellite sync + ML prediction + deep AI analysis

Flow analyze-lahan:
  1. Ambil koordinat lahan dari DB
  2. Trigger GEE Hybrid (Sentinel-2 + Landsat-8/9 + MODIS) → simpan data fresh ke DB
  3. ML Random Forest prediction → simpan rekomendasi ke DB
  4. Kirim data + prediksi ML ke Gemini 2.5 Flash untuk analisis mendalam

Rate Limiting:
  Role 'user' : 5 RPM  |  Role 'admin'/'superadmin' : unlimited
"""
import logging
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from supabase import Client

from app.db.database import get_supabase
from app.core.security import get_current_user
from app.core.rate_limiter import check_rate_limit
from app.services.gemini_service import ask_fast, ask_deep, MODEL_FAST, MODEL_DEEP

logger = logging.getLogger(__name__)
router = APIRouter()


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=1000)
    user_api_key: Optional[str] = None  # BYOK: Bring Your Own Key


class AnalyzeLahanRequest(BaseModel):
    lahan_id: int
    user_api_key: Optional[str] = None  # BYOK: Bring Your Own Key


# ---------------------------------------------------------------
# POST /ask — Quick Q&A (gemini-2.5-flash)
# ---------------------------------------------------------------
@router.post("/ask")
async def ask_agronomist_api(
    req: ChatRequest,
    current_user: dict = Depends(get_current_user),
):
    """
    Konsultasi tanya jawab bebas dengan AI Agronomist.
    Menggunakan gemini-2.5-flash dengan Round-Robin key pool.
    Mendukung BYOK (user_api_key opsional).
    Rate limit: 5 RPM untuk role 'user'.
    """
    check_rate_limit(current_user)

    logger.info("User '%s' asking fast AI: %s...", current_user["username"], req.message[:60])
    answer = await ask_fast(req.message, user_api_key=req.user_api_key)
    return {"status": "success", "model": MODEL_FAST, "answer": answer}


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
    Menganalisis lahan menggunakan data satelit FRESH dari GEE Hybrid
    lalu dilempar ke Gemini 2.5 Flash.
    Mendukung BYOK (user_api_key opsional).

    Flow:
      1. Ambil koordinat centroid lahan dari tabel lahan.
      2. Trigger sinkronisasi GEE (Sentinel-2 + Landsat-8/9 + MODIS).
      3. ML Random Forest prediction → simpan ke DB.
      4. Data fresh dikirim ke Gemini 2.5 Flash untuk analisis.

    Rate limit: 5 RPM untuk role 'user'.
    """
    check_rate_limit(current_user)

    # 1. Ambil data lahan + pastikan milik user
    lahan_res = db.table("lahan").select("*").eq("id", req.lahan_id).execute()
    if not lahan_res.data:
        raise HTTPException(status_code=404, detail="Lahan tidak ditemukan.")

    lahan = lahan_res.data[0]
    if lahan["created_by"] != current_user["id"] and current_user["role"] != "superadmin":
        raise HTTPException(status_code=403, detail="Akses ditolak ke lahan ini.")

    # 2. Hitung centroid dari koordinat lahan untuk titik sinkronisasi GEE
    koordinat = lahan.get("koordinat", {})
    coords = koordinat.get("coordinates", [[]])[0] if isinstance(koordinat, dict) else []

    if not coords:
        raise HTTPException(
            status_code=400,
            detail="Koordinat lahan tidak valid. Tidak bisa menentukan titik GEE."
        )

    # Centroid sederhana: rata-rata semua titik koordinat
    avg_lon = sum(c[0] for c in coords) / len(coords)
    avg_lat = sum(c[1] for c in coords) / len(coords)

    logger.info(
        "[METRIC_ANALYZE_LAHAN] lahan_id=%d | username=%s | lat=%f | lon=%f",
        req.lahan_id, current_user["username"], avg_lat, avg_lon,
    )

    # 3. Trigger sinkronisasi GEE Hybrid → data + ML prediction disimpan ke DB
    try:
        from app.services.gee_service import process_point_satellite_data

        gee_result = process_point_satellite_data(req.lahan_id, avg_lat, avg_lon)

        if "error" in gee_result:
            raise HTTPException(
                status_code=500,
                detail=f"Gagal sinkronisasi satelit: {gee_result['message']}"
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error GEE sync for analyze-lahan: %s", e)
        raise HTTPException(
            status_code=500,
            detail=f"Gagal memproses data satelit: {str(e)}"
        )

    # 4. Ambil data FRESH yang baru saja disimpan (record terbaru)
    fresh_result = (
        db.table("satellite_results")
        .select("*")
        .eq("lahan_id", req.lahan_id)
        .order("created_at", desc=True)
        .limit(1)
        .execute()
    )

    if not fresh_result.data:
        raise HTTPException(
            status_code=500,
            detail="Data satelit fresh gagal ditemukan setelah sinkronisasi."
        )

    sat_data = fresh_result.data[0]

    # 5. Ambil rekomendasi ML yang sudah disimpan oleh gee_service
    ml_recommendation = sat_data.get("ai_recommendation", "Pending Analysis")

    # 6. Susun prompt agronomis dengan data satelit FRESH + prediksi ML
    prompt = f"""Tolong berikan Orbitani Smart Analysis untuk data lahan satelit TERBARU ini:

Sumber Data: Sentinel-2 SR Harmonized (10m) + Landsat-8/9 L2 (30m) + MODIS Terra (1km fallback)
Waktu Ekstraksi: {sat_data.get('created_at', 'N/A')}

- Nitrogen (N)      : {sat_data.get('n', 'N/A')}
- Fosfor (P)        : {sat_data.get('p', 'N/A')}
- Kalium (K)        : {sat_data.get('k', 'N/A')}
- pH Tanah          : {sat_data.get('ph', 'N/A')}
- Suhu Lahan        : {sat_data.get('temperature', 'N/A')} °C
- Kelembapan (NDTI) : {sat_data.get('humidity', 'N/A')}
- Curah Hujan (1th) : {sat_data.get('rainfall', 'N/A')} mm

Model Random Forest kami merekomendasikan tanaman: **{ml_recommendation}**.

Berikan analisis lanjutan berdasarkan rekomendasi ini:
1. Apakah rekomendasi tanaman tersebut sesuai dengan kondisi hara dan iklim di atas?
2. Berikan 2 poin aksi yang harus segera dilakukan petani untuk memaksimalkan hasil panen."""

    logger.info(
        "[METRIC_DEEP_ANALYSIS] lahan_id=%d | username=%s | recommendation=%s | model=%s",
        req.lahan_id, current_user["username"], ml_recommendation, MODEL_DEEP,
    )
    answer = await ask_deep(prompt, user_api_key=req.user_api_key)

    return {
        "status": "success",
        "model": MODEL_DEEP,
        "ml_recommendation": ml_recommendation,
        "satellite_data": sat_data,
        "ai_analysis": answer,
    }
