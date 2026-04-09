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
    Mengambil riwayat pengecekan satelit (satellite_results JOIN lahan).
    Semua key output menggunakan nama kolom DB final:
      - n, p, k (bukan n_value/nitrogen/fosfor/kalium)
      - ai_recommendation (bukan recommendation/label)
      - temperature, humidity (bukan tci/ndti)
      - created_at (bukan extracted_at)
    longitude & latitude diambil dari tabel satellite_results bukan dari lahan.
    """
    try:
        res = db.table("satellite_results").select(
            "*, lahan(id, nama, deskripsi)"
        ).order("created_at", desc=True).execute()

        mapped_data = []
        for record in res.data:
            lahan_data = record.get("lahan") or {}
            mapped_data.append({
                "id":                record.get("id"),
                "created_at":        record.get("created_at"),
                "longitude":         record.get("longitude", 0.0),
                "latitude":          record.get("latitude", 0.0),
                # Kolom NPK — nama kolom DB final (huruf kecil tunggal)
                "n":                 record.get("n"),
                "p":                 record.get("p"),
                "k":                 record.get("k"),
                "ph":                record.get("ph"),
                "temperature":       record.get("temperature"),
                "humidity":          record.get("humidity"),
                "rainfall":          record.get("rainfall"),
                # Kolom rekomendasi — nama kolom DB final
                "ai_recommendation": record.get("ai_recommendation"),
                "lahan_id":          record.get("lahan_id"),
                "lahan":             lahan_data,
            })

        return {"data": mapped_data}

    except Exception as e:
        logger.error("Error fetching history: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Gagal mendapatkan history data",
        )
