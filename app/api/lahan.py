import logging
from collections import defaultdict
from typing import Optional, Any

from fastapi import APIRouter, Depends, HTTPException, status
from supabase import Client

from app.db.database import get_supabase
from app.core.security import get_current_user, require_roles
from app.models.schemas import LahanCreate, LahanUpdate

logger = logging.getLogger(__name__)
router = APIRouter()


# ---------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------
def _get_org_id(current_user: dict) -> Optional[int]:
    """Ambil organization_id dari user yang sedang login."""
    return current_user.get("organization_id")


def _check_lahan_access(lahan: dict, current_user: dict):
    """
    Periksa apakah user berhak mengakses/mengubah lahan ini.
    - superadmin : akses ke semua lahan
    - admin/user : hanya lahan dalam organisasinya ATAU lahan miliknya sendiri
    """
    role = current_user.get("role")
    if role == "superadmin":
        return
    org_id = current_user.get("organization_id")
    # Izinkan jika lahan dalam org yang sama ATAU milik user itu sendiri
    if lahan.get("organization_id") == org_id:
        return
    if lahan.get("created_by") == current_user.get("id"):
        return
    raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Akses ditolak ke lahan ini.")


def _build_lahan_query(db: Client, current_user: dict):
    """Kembalikan base query lahan sesuai hak akses user."""
    role = current_user.get("role")
    if role == "superadmin":
        return db.table("lahan").select("*")
    org_id = _get_org_id(current_user)
    if org_id:
        return db.table("lahan").select("*").eq("organization_id", org_id)
    # Fallback: user tanpa org, hanya lihat lahan milik sendiri
    return db.table("lahan").select("*").eq("created_by", current_user["id"])


# ================================================================
# STATIC ROUTES — Di atas semua route dinamis (/{lahan_id})
# ================================================================

# ---------------------------------------------------------------
# GET / — Daftar lahan milik user / organisasi
# ---------------------------------------------------------------
@router.get("/")
def get_my_lahan(
    db: Client = Depends(get_supabase),
    current_user: dict = Depends(get_current_user),
):
    """Mendapatkan daftar lahan sesuai tenant/organisasi user yang login."""
    result = _build_lahan_query(db, current_user).order("created_at", desc=True).execute()
    return {"status": "success", "data": result.data}


# ---------------------------------------------------------------
# POST / — Buat lahan baru
# ---------------------------------------------------------------
@router.post("/", status_code=status.HTTP_201_CREATED)
def create_lahan(
    data: LahanCreate,
    db: Client = Depends(get_supabase),
    current_user: dict = Depends(get_current_user),
):
    """Membuat area lahan pantauan baru (WebGIS Poligon)."""
    # Normalisasi koordinat: terima dict GeoJSON atau raw array
    koordinat_val: Any = data.koordinat
    if isinstance(koordinat_val, list):
        # FE kirim array langsung → bungkus jadi GeoJSON Polygon
        koordinat_val = {"type": "Polygon", "coordinates": koordinat_val}

    payload = {
        "nama":            data.nama,
        "deskripsi":       data.keterangan,   # FE pakai 'keterangan', DB pakai 'deskripsi'
        "koordinat":       koordinat_val,
        "created_by":      current_user["id"],
        "organization_id": _get_org_id(current_user),
    }
    result = db.table("lahan").insert(payload).execute()
    if not result.data:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Gagal menyimpan lahan")
    logger.info("Lahan baru dibuat: '%s' oleh user ID %d", data.nama, current_user["id"])
    return {"status": "success", "data": result.data[0]}


# ---------------------------------------------------------------
# GET /analytics — Tren historis (HARUS sebelum /{lahan_id})
# ---------------------------------------------------------------
@router.get("/analytics")
def get_lahan_analytics(
    db: Client = Depends(get_supabase),
    current_user: dict = Depends(get_current_user),
):
    """Mengambil data tren analitik lahan historis (Rata-rata NPK, iklim per tanggal)."""
    lahan_res = _build_lahan_query(db, current_user).execute()
    lahan_ids = [str(l["id"]) for l in lahan_res.data]

    if not lahan_ids:
        return {"data": []}

    sat_res = (
        db.table("satellite_results")
        .select("*")
        .in_("lahan_id", lahan_ids)
        .order("created_at")
        .execute()
    )

    trends: dict = defaultdict(list)
    for record in sat_res.data:
        raw_date = record.get("created_at") or ""
        date_str = raw_date[:10] if raw_date else None
        if date_str:
            trends[date_str].append(record)

    data = []
    for date_str, records in trends.items():
        count = len(records)
        data.append({
            "date":     date_str,
            "nitrogen": round(sum(r.get("n") or 0 for r in records) / count, 2),
            "fosfor":   round(sum(r.get("p") or 0 for r in records) / count, 2),
            "kalium":   round(sum(r.get("k") or 0 for r in records) / count, 2),
            "ph":       round(sum(r.get("ph") or 0 for r in records) / count, 2),
            "tci":      round(sum(r.get("temperature") or 0 for r in records) / count, 2),
            "ndti":     round(sum(r.get("humidity") or 0 for r in records) / count, 2),
            "rainfall": round(sum(r.get("rainfall") or 0 for r in records) / count, 2),
        })

    data.sort(key=lambda x: x["date"])
    return {"data": data}


# ================================================================
# DYNAMIC ROUTES — Mengandung /{lahan_id}, wajib di bawah static
# ================================================================

# ---------------------------------------------------------------
# POST /{lahan_id}/analyze — Analisis Spasial & Rekomendasi Tanaman
# ---------------------------------------------------------------
@router.post("/{lahan_id}/analyze")
def analyze_lahan(
    lahan_id: int,
    db: Client = Depends(get_supabase),
    current_user: dict = Depends(get_current_user),
):
    """
    Melakukan analisis spasial pada lahan:
      1. Sampling 10 titik acak di dalam poligon lahan.
      2. Prediksi tanaman di tiap titik menggunakan model ML (Random Forest).
      3. Agregasi hasil dengan Majority Voting → Top-3 rekomendasi (%).
      4. Simpan hasil ke kolom `hasil_rekomendasi` dan `terakhir_dianalisis`.
      5. Kembalikan data lahan yang sudah diperbarui.
    """
    from datetime import datetime, timezone
    from app.services.spatial_analysis_service import run_spatial_analysis

    # -- 1. Ambil & validasi lahan --
    lahan_res = db.table("lahan").select("*").eq("id", lahan_id).execute()
    if not lahan_res.data:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Lahan tidak ditemukan")

    lahan = lahan_res.data[0]
    _check_lahan_access(lahan, current_user)

    koordinat = lahan.get("koordinat")
    if not koordinat:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Lahan tidak memiliki data koordinat. Tambahkan poligon terlebih dahulu.",
        )



    # -- 2. Jalankan analisis spasial (Live GEE) --
    try:
        hasil_rekomendasi, data_lengkap_titik = run_spatial_analysis(
            koordinat=koordinat,
        )
    except ValueError as e:
        logger.error("Koordinat lahan %d tidak valid: %s", lahan_id, e)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Koordinat lahan tidak valid: {e}",
        )
    except RuntimeError as e:
        logger.error("Analisis spasial lahan %d gagal: %s", lahan_id, e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analisis spasial gagal: {e}",
        )

    # -- 3. Simpan hasil 10 titik ke satellite_results sebagai log historis --
    terakhir_dianalisis = datetime.now(timezone.utc).isoformat()
    
    try:
        sat_payloads = []
        for point_data in data_lengkap_titik:
            sat_payloads.append({
                "lahan_id":          lahan_id,
                "longitude":         point_data["lon"],
                "latitude":          point_data["lat"],
                "n":                 point_data["n"],
                "p":                 point_data["p"],
                "k":                 point_data["k"],
                "ph":                point_data["ph"],
                "temperature":       point_data["temperature"],
                "humidity":          point_data["humidity"],
                "rainfall":          point_data["rainfall"],
                "ai_recommendation": point_data["ai_recommendation"],
                "created_at":        terakhir_dianalisis,
            })
        if sat_payloads:
            db.table("satellite_results").insert(sat_payloads).execute()
            logger.info("Berhasil menyimpan %d log historis titik satelit untuk lahan %d", len(sat_payloads), lahan_id)
    except Exception as e:
        logger.error("Gagal menyimpan log historis satelit lahan %d: %s", lahan_id, e)
        # Tidak melempar error agar proses analisis tetap selesai (hanya gagal save log)

    # -- 4. Update tabel lahan dengan agregasi Majority Voting --
    update_payload = {
        "hasil_rekomendasi":  hasil_rekomendasi,
        "terakhir_dianalisis": terakhir_dianalisis,
    }

    updated_res = db.table("lahan").update(update_payload).eq("id", lahan_id).execute()
    if not updated_res.data:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Analisis selesai, tetapi gagal menyimpan hasil ke database lahan.",
        )

    updated_lahan = updated_res.data[0]

    # Hitung rata-rata fitur (dari nilai terkalibrasi) untuk disertakan di response
    rata_rata_fitur = {}
    if data_lengkap_titik:
        n_pts = len(data_lengkap_titik)
        cal_list = [pt.get("calibrated", {}) for pt in data_lengkap_titik]
        rata_rata_fitur = {
            "n":           round(sum(c.get("N", 0) for c in cal_list) / n_pts, 2),
            "p":           round(sum(c.get("P", 0) for c in cal_list) / n_pts, 2),
            "k":           round(sum(c.get("K", 0) for c in cal_list) / n_pts, 2),
            "ph":          round(sum(c.get("ph", 0) for c in cal_list) / n_pts, 2),
            "temperature": round(sum(c.get("temperature", 0) for c in cal_list) / n_pts, 2),
            "humidity":    round(sum(c.get("humidity", 0) for c in cal_list) / n_pts, 2),
            "rainfall":    round(sum(c.get("rainfall", 0) for c in cal_list) / n_pts, 2),
        }

    logger.info(
        "Analisis spasial lahan %d selesai oleh user %d. Top rekomendasi: %s",
        lahan_id,
        current_user["id"],
        hasil_rekomendasi[0]["tanaman"] if hasil_rekomendasi else "N/A",
    )

    return {
        "status":  "success",
        "message": f"Analisis spasial selesai. {len(hasil_rekomendasi)} rekomendasi tanaman ditemukan.",
        "data":    updated_lahan,
        "rata_rata_fitur": rata_rata_fitur,
    }


# ---------------------------------------------------------------
# GET /{lahan_id}/data — Data satelit lahan tertentu
# ---------------------------------------------------------------
@router.get("/{lahan_id}/data")
def get_lahan_satellite_data(
    lahan_id: int,
    lat: Optional[float] = None,
    lon: Optional[float] = None,
    db: Client = Depends(get_supabase),
    current_user: dict = Depends(get_current_user),
):
    """
    Eksplorasi real-time atau history.
    Jika lat & lon dikirim → ambil dari GEE secara live.
    Jika tidak → ambil dari history database.
    """
    lahan_res = db.table("lahan").select("*").eq("id", lahan_id).execute()
    if not lahan_res.data:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Lahan tidak ditemukan")

    lahan = lahan_res.data[0]
    _check_lahan_access(lahan, current_user)

    if lat is not None and lon is not None:
        try:
            from app.services.gee_service import process_point_satellite_data
            gee_result = process_point_satellite_data(lahan_id, lat, lon)
            if "error" in gee_result:
                raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=gee_result.get("message", "GEE error"))

            from app.services.retrain_service import check_and_trigger_retrain
            check_and_trigger_retrain()
        except HTTPException:
            raise
        except Exception as e:
            logger.error("Error real-time GEE lahan %d: %s", lahan_id, e)
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Gagal memproses satelit: {str(e)}")

    sat_data = (
        db.table("satellite_results")
        .select("*")
        .eq("lahan_id", lahan_id)
        .order("created_at", desc=True)
        .execute()
    )
    return {"status": "success", "lahan": lahan, "satellite_data": sat_data.data}


# ---------------------------------------------------------------
# PUT /{lahan_id} — Update lahan (nama, keterangan, atau koordinat)
# ---------------------------------------------------------------
@router.put("/{lahan_id}")
def update_lahan(
    lahan_id: int,
    data: LahanUpdate,
    db: Client = Depends(get_supabase),
    current_user: dict = Depends(get_current_user),
):
    """Mengubah nama, keterangan, atau koordinat lahan. User hanya bisa update lahan miliknya."""
    lahan_res = db.table("lahan").select("*").eq("id", lahan_id).execute()
    if not lahan_res.data:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Lahan tidak ditemukan")

    lahan = lahan_res.data[0]
    _check_lahan_access(lahan, current_user)

    update_payload: dict = {}
    if data.nama is not None:
        update_payload["nama"] = data.nama
    if data.keterangan is not None:
        update_payload["deskripsi"] = data.keterangan
    if data.koordinat is not None:
        koordinat_val = data.koordinat
        if isinstance(koordinat_val, list):
            koordinat_val = {"type": "Polygon", "coordinates": koordinat_val}
        update_payload["koordinat"] = koordinat_val

    if not update_payload:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Tidak ada field yang dikirim untuk diperbarui",
        )

    result = db.table("lahan").update(update_payload).eq("id", lahan_id).execute()
    if not result.data:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Lahan tidak ditemukan setelah update")

    logger.info("Lahan ID %d diperbarui oleh user ID %d: %s", lahan_id, current_user["id"], list(update_payload.keys()))
    return {"status": "success", "data": result.data[0]}


# ---------------------------------------------------------------
# DELETE /{lahan_id} — Hapus lahan
# ---------------------------------------------------------------
@router.delete("/{lahan_id}", status_code=status.HTTP_200_OK)
def delete_lahan(
    lahan_id: int,
    db: Client = Depends(get_supabase),
    current_user: dict = Depends(get_current_user),
):
    """Menghapus lahan berdasarkan ID. User hanya bisa hapus lahan miliknya."""
    lahan_res = db.table("lahan").select("id, nama, created_by, organization_id").eq("id", lahan_id).execute()
    if not lahan_res.data:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Lahan tidak ditemukan")

    lahan = lahan_res.data[0]
    _check_lahan_access(lahan, current_user)

    db.table("lahan").delete().eq("id", lahan_id).execute()
    logger.info("Lahan '%s' (ID %d) dihapus oleh user ID %d", lahan.get("nama"), lahan_id, current_user["id"])
    return {"detail": f"Lahan '{lahan.get('nama')}' berhasil dihapus"}
