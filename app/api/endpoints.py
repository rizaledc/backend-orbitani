"""
endpoints.py
ML & GEE endpoints for Orbitani.

ARSITEKTUR LAZY LOADING:
  - Model .pkl TIDAK diimport di top-level.
  - Semua model dimuat di dalam handler → prediksi → cleanup.
  - Gemini (di router /chat) berjalan di ruang RAM yang bersih dari .pkl.
"""
import logging
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel
from app.core.security import get_current_user

logger = logging.getLogger(__name__)
router = APIRouter()


# Schema Input untuk lokasi
class LocationRequest(BaseModel):
    lahan_id: int
    latitude: float
    longitude: float


@router.post("/analyze-location")
async def analyze_location(
    req: LocationRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user),
):
    """
    Endpoint utama untuk analisis lokasi (GPS/Klik/Manual).
    1. Cek Geofencing.
    2. Jalankan GEE extraction (disimpan ke DB).
    3. Trigger Background ML Checker.
    """
    # Lazy import agar modul GEE tidak memakan RAM saat idle
    from app.services.gee_service import is_inside_hibisc, process_point_satellite_data

    # 1. Validasi Pagar Gaib
    if not is_inside_hibisc(req.latitude, req.longitude):
        raise HTTPException(
            status_code=400,
            detail="Lokasi di luar area! Pastikan Anda berada di area Lahan Hibisc Cilacap.",
        )

    # 2. Proses GEE & Simpan ke DB
    logger.info(
        "[METRIC_ANALYZE_LOCATION] lahan_id=%d | username=%s | lat=%f | lon=%f",
        req.lahan_id, current_user.get("username"), req.latitude, req.longitude,
    )
    result = process_point_satellite_data(req.lahan_id, req.latitude, req.longitude)

    if "error" in result:
        logger.error("[METRIC_ANALYZE_FAIL] lahan_id=%d | reason=%s", req.lahan_id, result["message"])
        raise HTTPException(status_code=500, detail=result["message"])

    # Trigger Pengecekan Auto-Retrain di Background
    from app.services.retrain_service import check_and_trigger_retrain
    background_tasks.add_task(check_and_trigger_retrain)

    return {
        "status": "success",
        "message": "Data satelit berhasil diekstrak dan disimpan.",
        "data": result["data"],
    }


@router.get("/sync-satellite")
def sync_satellite(lahan_id: int, lat: float, lon: float):
    """Versi GET untuk kemudahan testing atau sinkronisasi cepat."""
    from app.services.gee_service import process_point_satellite_data

    return process_point_satellite_data(lahan_id, lat, lon)


@router.post("/predict")
def predict_crop(data: dict):
    """
    Prediksi manual dari input user.
    Mendukung payload huruf kecil (n, p, k) maupun huruf besar (N, P, K).
    Model .pkl dimuat LAZY di dalam fungsi ini → prediksi → RAM dibersihkan.
    """
    try:
        # Mapping dictionary eksplisit agar payload (n, p, k) huruf kecil diubah ke N, P, K huruf besar
        normalised = {
            "N":           float(data.get("N", data.get("n", 0))),
            "P":           float(data.get("P", data.get("p", 0))),
            "K":           float(data.get("K", data.get("k", 0))),
            "temperature": float(data.get("temperature", 0)),
            "humidity":    float(data.get("humidity", 0)),
            "ph":          float(data.get("ph", 0)),
            "rainfall":    float(data.get("rainfall", 0)),
        }

        from app.services.ml_service import predict
        result = predict(normalised)
        return {
            "status":            "success",
            "ai_recommendation": result["ai_recommendation"],
            "calibrated_data":   result["calibrated_data"],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/admin/llm-status", tags=["Admin"])
async def get_llm_status():
    """Mengembalikan status Round-Robin pooling API Key Gemini."""
    # Ambil dari gemini_service.py
    from app.services.gemini_service import _rr_index, GEMINI_FALLBACK_KEYS
    
    # Masking key demi keamanan (tampilkan 8 huruf pertama saja)
    masked_keys = [f"{k[:8]}...***" for k in GEMINI_FALLBACK_KEYS]
    
    current_active = masked_keys[_rr_index] if masked_keys else None

    return {
        "status": "success",
        "active_index": _rr_index,
        "total_keys_in_pool": len(GEMINI_FALLBACK_KEYS),
        "pool_keys": masked_keys,
        "current_active_key": current_active
    }