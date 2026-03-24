import logging
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
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

@router.get("/analytics")
def get_lahan_analytics(
    db: Client = Depends(get_supabase),
    current_user: dict = Depends(get_current_user)
):
    """Mengambil data tren analitik lahan historis (Rata-rata NPK, iklim per tanggal)."""
    lahan_res = db.table("lahan").select("id").eq("created_by", current_user["id"]).execute()
    lahan_ids = [str(l["id"]) for l in lahan_res.data]
    if not lahan_ids:
        return {"data": []}
    
    sat_res = db.table("satellite_results").select("*").in_("lahan_id", lahan_ids).order("extracted_at").execute()
    
    from collections import defaultdict
    trends = defaultdict(list)
    for record in sat_res.data:
        date_str = record.get("extracted_at", "")[:10] if record.get("extracted_at") else "Unknown"
        if date_str != "Unknown":
            trends[date_str].append(record)
    
    data = []
    for date_str, records in trends.items():
        count = len(records)
        data.append({
            "date": date_str,
            "nitrogen": round(sum(r.get("n_value") or 0 for r in records) / count, 2),
            "fosfor": round(sum(r.get("p_value") or 0 for r in records) / count, 2),
            "kalium": round(sum(r.get("k_value") or 0 for r in records) / count, 2),
            "ph": round(sum(r.get("ph") or 0 for r in records) / count, 2),
            "tci": round(sum(r.get("temperature") or 0 for r in records) / count, 2),
            "ndti": round(sum(r.get("humidity") or 0 for r in records) / count, 2),
            "rainfall": round(sum(r.get("rainfall") or 0 for r in records) / count, 2)
        })
        
    data.sort(key=lambda x: x["date"])
    return {"data": data}

@router.get("/{lahan_id}/data")
def get_lahan_satellite_data(
    lahan_id: int,
    background_tasks: BackgroundTasks,
    lat: Optional[float] = None,
    lon: Optional[float] = None,
    db: Client = Depends(get_supabase),
    current_user: dict = Depends(get_current_user)
):
    """
    Eksplorasi Real-Time: Jika lat & lon dikirim, ambil dari GEE secara live.
    Jika tidak, ambil dari history database.
    """
    # Pastikan lahan milik user (atau admin public)
    lahan = db.table("lahan").select("*").eq("id", lahan_id).execute()
    if not lahan.data:
        raise HTTPException(status_code=404, detail="Lahan tidak ditemukan")
        
    if lahan.data[0]["created_by"] != current_user["id"] and current_user["role"] != "superadmin":
        raise HTTPException(status_code=403, detail="Akses ditolak ke lahan ini")

    if lat is not None and lon is not None:
        # Trigger GEE extraction & ML Prediction real-time
        try:
            from app.services.gee_service import process_point_satellite_data
            gee_result = process_point_satellite_data(lahan_id, lat, lon)
            if "error" in gee_result:
                raise HTTPException(status_code=500, detail=gee_result["message"])
            # Hasil dari process_point_satellite_data otomatis disimpan ke DB
            
            # Trigger Background auto-retrain checker
            from app.services.retrain_service import check_and_trigger_retrain
            background_tasks.add_task(check_and_trigger_retrain)
        except Exception as e:
            logger.error(f"Error real-time GEE: {e}")
            raise HTTPException(status_code=500, detail=f"Gagal memproses satelit: {str(e)}")

    sat_data = db.table("satellite_results").select("*").eq("lahan_id", lahan_id).order("extracted_at", desc=True).execute()
    return {"status": "success", "lahan": lahan.data[0], "satellite_data": sat_data.data}
