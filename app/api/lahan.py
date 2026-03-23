import logging
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Optional
from supabase import Client

from app.db.database import get_supabase
from app.core.security import get_current_user

logger = logging.getLogger(__name__)
router = APIRouter()

class LahanCreate(BaseModel):
    nama: str
    deskripsi: Optional[str] = None
    koordinat: dict # GeoJSON Polygon

@router.get("/")
def get_my_lahan(
    db: Client = Depends(get_supabase),
    current_user: dict = Depends(get_current_user)
):
    """Mendapatkan daftar semua lahan milik user yang sedang login."""
    result = db.table("lahan").select("*").eq("created_by", current_user["id"]).execute()
    return {"status": "success", "data": result.data}

@router.post("/")
def create_lahan(
    data: LahanCreate,
    db: Client = Depends(get_supabase),
    current_user: dict = Depends(get_current_user)
):
    """Membuat area lahan pantauan baru (WebGIS Poligon)."""
    payload = {
        "nama": data.nama,
        "deskripsi": data.deskripsi,
        "koordinat": data.koordinat,
        "created_by": current_user["id"]
    }
    result = db.table("lahan").insert(payload).execute()
    return {"status": "success", "data": result.data[0]}

@router.get("/{lahan_id}/data")
def get_lahan_satellite_data(
    lahan_id: int,
    db: Client = Depends(get_supabase),
    current_user: dict = Depends(get_current_user)
):
    """Mengambil semua titik hasil satelit GEE pada lahan tertentu untuk diplot di Leaflet."""
    # Pastikan lahan milik user (atau admin public)
    # Ini dicek secara otomatis oleh RLS Supabase (bila diatur ketat), 
    # atau kita cek eksplisit
    lahan = db.table("lahan").select("*").eq("id", lahan_id).execute()
    if not lahan.data:
        raise HTTPException(status_code=404, detail="Lahan tidak ditemukan")
        
    if lahan.data[0]["created_by"] != current_user["id"] and current_user["role"] != "superadmin":
        raise HTTPException(status_code=403, detail="Akses ditolak ke lahan ini")

    sat_data = db.table("satellite_results").select("*").eq("lahan_id", lahan_id).order("extracted_at", desc=True).execute()
    return {"status": "success", "lahan": lahan.data[0], "satellite_data": sat_data.data}
