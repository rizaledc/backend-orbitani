import logging
from fastapi import APIRouter, Depends, HTTPException, status
from supabase import Client

from app.db.database import get_supabase
from app.core.security import require_roles, get_current_user

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/")
def get_history(
    db: Client = Depends(get_supabase),
    current_user: dict = Depends(get_current_user),
):
    """
    Mengambil riwayat pengecekan satelit (satellite_results join lahan).
    longitude & latitude diambil langsung dari tabel satellite_results (*),
    bukan dari tabel lahan (yang hanya punya kolom koordinat GeoJSON poligon).
    """
    try:
        res = db.table("satellite_results").select(
            "*, lahan(id, nama, deskripsi)"
        ).order("created_at", desc=True).execute()

        mapped_data = []
        for record in res.data:
            lahan_data = record.get("lahan") or {}
            mapped_data.append({
                "id":        record.get("id"),
                "created_at": record.get("created_at"),
                "longitude": record.get("longitude", 0.0),   # dari satellite_results
                "latitude":  record.get("latitude", 0.0),    # dari satellite_results
                "nitrogen":  record.get("n_value"),
                "fosfor":    record.get("p_value"),
                "kalium":    record.get("k_value"),
                "ph":        record.get("ph"),
                "tci":       record.get("temperature"),
                "ndti":      record.get("humidity"),
                "rainfall":  record.get("rainfall"),
                "label":     record.get("recommendation"),
                "lahan":     lahan_data,
            })

        return {"data": mapped_data}

    except Exception as e:
        logger.error("Error fetching history: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Gagal mendapatkan history data",
        )
