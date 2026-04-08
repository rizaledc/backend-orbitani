import logging
from fastapi import APIRouter, Depends, HTTPException, status
from supabase import Client

from app.db.database import get_supabase
from app.models.schemas import UserOut, RoleUpdate
from app.core.security import get_current_user, require_roles

logger = logging.getLogger(__name__)
router = APIRouter()


# ---------------------------------------------------------------
# GET /stats — Statistik user (current user)
# ---------------------------------------------------------------
@router.get("/stats")
def user_stats(
    db: Client = Depends(get_supabase),
    current_user: dict = Depends(get_current_user),
):
    """Mendapatkan statistik penggunaan (jumlah ekstraksi satelit) oleh user."""
    lahan_res = db.table("lahan").select("id").eq("created_by", current_user["id"]).execute()
    lahan_ids = [str(l["id"]) for l in lahan_res.data]

    total_queries = 0
    if lahan_ids:
        sat_res = (
            db.table("satellite_results")
            .select("id", count="exact")
            .in_("lahan_id", lahan_ids)
            .execute()
        )
        total_queries = sat_res.count if sat_res.count is not None else len(sat_res.data)

    return {
        "status": "success",
        "username": current_user["username"],
        "role": current_user["role"],
        "organization_id": current_user.get("organization_id"),
        "total_lahan": len(lahan_ids),
        "total_satellite_extractions": total_queries,
    }


# ---------------------------------------------------------------
# GET / — Daftar semua user (superadmin & admin, tenant-aware)
# ---------------------------------------------------------------
@router.get("/", response_model=list[UserOut])
def list_users(
    db: Client = Depends(get_supabase),
    current_user: dict = Depends(require_roles(["superadmin", "admin"])),
):
    """
    superadmin: melihat semua user.
    admin: hanya melihat user dalam organisasinya sendiri.
    """
    if current_user["role"] == "superadmin":
        result = db.table("users").select("id, username, role, name, email, organization_id").execute()
    else:
        org_id = current_user.get("organization_id")
        if not org_id:
            return []
        result = (
            db.table("users")
            .select("id, username, role, name, email, organization_id")
            .eq("organization_id", org_id)
            .execute()
        )
    return result.data


# ---------------------------------------------------------------
# PUT /{user_id}/role — Ubah role user (superadmin only)
# ---------------------------------------------------------------
@router.put("/{user_id}/role", response_model=UserOut)
def update_user_role(
    user_id: int,
    data: RoleUpdate,
    db: Client = Depends(get_supabase),
    _current_user: dict = Depends(require_roles(["superadmin"])),
):
    """Mengubah role user menjadi 'admin' atau 'user'. Hanya superadmin."""
    if data.role not in ("admin", "user"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Role harus 'admin' atau 'user'",
        )

    existing = db.table("users").select("id, username, role").eq("id", user_id).execute()
    if not existing.data:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User tidak ditemukan")
    if existing.data[0]["role"] == "superadmin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Tidak bisa mengubah role superadmin")

    result = db.table("users").update({"role": data.role}).eq("id", user_id).execute()
    logger.info("Role user ID %d diubah menjadi '%s'", user_id, data.role)
    return result.data[0]


# ---------------------------------------------------------------
# PUT /{user_id}/organization — Assign user ke organisasi (superadmin)
# ---------------------------------------------------------------
@router.put("/{user_id}/organization")
def assign_organization(
    user_id: int,
    organization_id: int,
    db: Client = Depends(get_supabase),
    _current_user: dict = Depends(require_roles(["superadmin"])),
):
    """Menetapkan user ke organisasi tertentu. Hanya superadmin."""
    existing = db.table("users").select("id, username, role").eq("id", user_id).execute()
    if not existing.data:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User tidak ditemukan")
    if existing.data[0]["role"] == "superadmin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Superadmin tidak terikat organisasi")

    # Verifikasi org ada
    org_res = db.table("organizations").select("id").eq("id", organization_id).execute()
    if not org_res.data:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Organisasi tidak ditemukan")

    db.table("users").update({"organization_id": organization_id}).eq("id", user_id).execute()
    logger.info("User ID %d di-assign ke organization_id=%d", user_id, organization_id)
    return {"detail": f"User berhasil di-assign ke organisasi ID {organization_id}"}


# ---------------------------------------------------------------
# DELETE /{user_id} — Hapus user (superadmin & admin)
# ---------------------------------------------------------------
@router.delete("/{user_id}", status_code=status.HTTP_200_OK)
def delete_user(
    user_id: int,
    db: Client = Depends(get_supabase),
    current_user: dict = Depends(require_roles(["superadmin", "admin"])),
):
    """Menghapus user berdasarkan ID. superadmin (semua) atau admin (hanya dalam orgnya)."""
    existing = db.table("users").select("id, username, role, organization_id").eq("id", user_id).execute()
    if not existing.data:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User tidak ditemukan")

    target = existing.data[0]
    if target["role"] == "superadmin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Tidak bisa menghapus superadmin")

    # Admin hanya boleh hapus user dalam organisasinya
    if current_user["role"] == "admin":
        if target.get("organization_id") != current_user.get("organization_id"):
            raise HTTPException(status_code=403, detail="Akses ditolak — user di luar organisasi Anda")

    db.table("users").delete().eq("id", user_id).execute()
    logger.info("User '%s' dihapus", target["username"])
    return {"detail": f"User '{target['username']}' berhasil dihapus"}
