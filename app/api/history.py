import logging
from fastapi import APIRouter, Depends, HTTPException, status
from supabase import Client

from app.db.database import get_supabase
from app.core.security import require_roles

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get("/")
def get_history(
    db: Client = Depends(get_supabase),
    _current_user: dict = Depends(require_roles(["superadmin", "admin", "user"])),
):
    """
    Mengambil semua riwayat pengecekan (satellite_results join dengan lahan).
    Kontrak: return object `{"data": [...]}` dengan key spesifik.
    """
    try:
        # Fetch all satellite results. Join with lahan to get longitude/latitude
        # Supabase syntax for inner join: select="*, lahan(longitude, latitude)"
        res = db.table("satellite_results").select(
            "id",
            "created_at",
            "n_value",
            "p_value",
            "k_value",
            "ph",
            "temperature",
            "humidity",
            "rainfall",
            "recommendation",
            "lahan_id",
            "lahan(longitude, latitude)"
        ).order("created_at", desc=True).execute()

        mapped_data = []
        for record in res.data:
            lahan_data = record.get("lahan") or {}
            
            # Map sesuai requirement Frontend API Contract
            mapped_data.append({
                "id": record.get("id"),
                "created_at": record.get("created_at"),
                "longitude": lahan_data.get("longitude", 0.0),
                "latitude": lahan_data.get("latitude", 0.0),
                "nitrogen": record.get("n_value"),
                "fosfor": record.get("p_value"),
                "kalium": record.get("k_value"),
                "ph": record.get("ph"),
                "tci": record.get("temperature"),
                "ndti": record.get("humidity"),    
                "rainfall": record.get("rainfall"), 
                "label": record.get("recommendation"),  
            })

        return {"data": mapped_data}

    except Exception as e:
        logger.error(f"Error fetching history: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Gagal mendapatkan history data"
        )
