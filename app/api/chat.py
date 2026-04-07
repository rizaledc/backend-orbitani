"""
chat.py
AI Agronomist endpoints for Orbitani.

Dual-Model Strategy:
  POST /ask            → gemini-2.5-flash (fast, ~2-3s)  — Q&A chat
  POST /analyze-lahan  → gemini-2.5-flash (deep, ~15-20s)      — satellite analysis

Flow analyze-lahan:
  1. Ambil koordinat lahan dari DB
  2. Trigger GEE Hybrid (Sentinel-2 + Landsat-9) → simpan data fresh ke DB
  3. Kirim data fresh ke Gemini 2.5 Flash untuk analisis mendalam

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
from app.services.gemini_service import ask_fast, ask_deep, MODEL_FAST, MODEL_DEEP

logger = logging.getLogger(__name__)
router = APIRouter()


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=1000)


class AnalyzeLahanRequest(BaseModel):
    lahan_id: int


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
    Menggunakan gemini-2.5-flash untuk respons cepat (~2-3 detik).
    Rate limit: 5 RPM untuk role 'user'.
    """
    check_rate_limit(current_user)

    logger.info("User '%s' asking fast AI: %s...", current_user["username"], req.message[:60])
    answer = await ask_fast(req.message)
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
    (Sentinel-2 + Landsat-9) lalu dilempar ke Gemini 2.5 Flash.

    Flow:
      1. Ambil koordinat centroid lahan dari tabel lahan.
      2. Trigger sinkronisasi GEE (Sentinel-2 optik + Landsat-9 termal).
      3. Data fresh disimpan ke DB oleh gee_service.
      4. Data fresh tersebut dikirim ke Gemini 2.5 Flash untuk analisis.

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
        "User '%s' analyze-lahan: lahan_id=%d, triggering fresh GEE sync at [%.6f, %.6f]",
        current_user["username"], req.lahan_id, avg_lat, avg_lon,
    )

    # 3. Trigger sinkronisasi GEE Hybrid (Sentinel-2 + Landsat-9) → data disimpan ke DB
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
        .order("extracted_at", desc=True)
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
    ml_recommendation = sat_data.get("recommendation", "Pending Analysis")

    # 6. Susun prompt agronomis dengan data satelit FRESH + prediksi ML
    prompt = f"""Tolong berikan Orbitani Smart Analysis untuk data lahan satelit TERBARU ini:

Sumber Data: Sentinel-2 SR Harmonized (10m) + Landsat-9 L2 TIRS-2 (30m)
Waktu Ekstraksi: {sat_data.get('extracted_at', 'N/A')}

- Nitrogen (N)      : {sat_data.get('n_value', 'N/A')}
- Fosfor (P)        : {sat_data.get('p_value', 'N/A')}
- Kalium (K)        : {sat_data.get('k_value', 'N/A')}
- pH Tanah          : {sat_data.get('ph', 'N/A')}
- Suhu Lahan        : {sat_data.get('temperature', 'N/A')} °C
- Kelembapan (NDTI) : {sat_data.get('humidity', 'N/A')}
- Curah Hujan (1th) : {sat_data.get('rainfall', 'N/A')} mm

Model Random Forest kami merekomendasikan tanaman: **{ml_recommendation}**.

Berikan analisis lanjutan berdasarkan rekomendasi ini:
1. Apakah rekomendasi tanaman tersebut sesuai dengan kondisi hara dan iklim di atas?
2. Berikan 2 poin aksi yang harus segera dilakukan petani untuk memaksimalkan hasil panen."""

    logger.info(
        "User '%s' sending fresh satellite data + ML prediction (%s) to %s for lahan_id=%d",
        current_user["username"], ml_recommendation, MODEL_DEEP, req.lahan_id,
    )
    answer = await ask_deep(prompt)

    return {
        "status": "success",
        "model": MODEL_DEEP,
        "ml_recommendation": ml_recommendation,
        "satellite_data": sat_data,
        "ai_analysis": answer,
    }
