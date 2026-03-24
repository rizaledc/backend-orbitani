"""
endpoints.py
ML & GEE endpoints for Orbitani.

ARSITEKTUR LAZY LOADING:
  - Model .pkl TIDAK diimport di top-level.
  - Semua model dimuat di dalam handler → prediksi → cleanup.
  - Gemini (di router /chat) berjalan di ruang RAM yang bersih dari .pkl.
"""
import logging
from fastapi import APIRouter, HTTPException, Depends
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
    current_user: dict = Depends(get_current_user),
):
    """
    Endpoint utama untuk analisis lokasi (GPS/Klik/Manual).
    1. Cek Geofencing.
    2. Jalankan GEE extraction.
    3. Lazy-load model ML → prediksi → cleanup RAM.
    4. Simpan & Kembalikan hasil.
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
        "User '%s' memulai analisis lokasi: Lahan %d pada [%f, %f]",
        current_user.get("username"), req.lahan_id, req.latitude, req.longitude,
    )
    result = process_point_satellite_data(req.lahan_id, req.latitude, req.longitude)

    if "error" in result:
        raise HTTPException(status_code=500, detail=result["message"])

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
    Model .pkl dimuat LAZY di dalam fungsi ini → prediksi → RAM dibersihkan.
    """
    try:
        # Lazy import — ml_service.predict() akan load → predict → del → gc.collect()
        from app.services.ml_service import predict

        result = predict(data)
        return {
            "status": "success",
            "recommendation": result["recommendation"],
            "calibrated_data": result["calibrated_data"],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))