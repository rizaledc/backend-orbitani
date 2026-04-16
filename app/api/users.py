import logging
import re
from fastapi import APIRouter, Depends, HTTPException, status
from supabase import Client

from app.db.database import get_supabase
from app.models.schemas import UserOut, RoleUpdate, ProfileUpdate
from app.core.security import get_current_user, require_roles

logger = logging.getLogger(__name__)
router = APIRouter()

# Pola valid untuk username (sama seperti UserCreate)
_USERNAME_RE = re.compile(r"^[a-zA-Z0-9_]+$")


# ================================================================
# STATIC ROUTES — HARUS DI ATAS SEMUA DYNAMIC ROUTE (/{user_id})
# ================================================================

# ---------------------------------------------------------------
# PATCH /me — Update profil user yang sedang login
# ---------------------------------------------------------------
@router.patch("/me", response_model=UserOut)
def update_my_profile(
    data: ProfileUpdate,
    db: Client = Depends(get_supabase),
    current_user: dict = Depends(get_current_user),
):
    """Mengubah profil dasar (name, username, description) milik user yang sedang login."""
    update_payload: dict = {}

    # name: izinkan string kosong (untuk "menghapus" nama)
    if data.name is not None:
        update_payload["name"] = data.name

    # description: izinkan string kosong
    if data.description is not None:
        update_payload["description"] = data.description

    # username: validasi format di sini (bukan di Pydantic) supaya "" tidak 422
    if data.username is not None:
        stripped = data.username.strip()
        if stripped:
            # Hanya divalidasi jika bukan string kosong
            if len(stripped) < 3:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Username minimal 3 karakter",
                )
            if not _USERNAME_RE.match(stripped):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Username hanya boleh mengandung huruf, angka, dan underscore",
                )
            # Cek keunikan username (kecuali milik diri sendiri)
            conflict = (
                db.table("users")
                .select("id")
                .eq("username", stripped)
                .neq("id", current_user["id"])
                .execute()
            )
            if conflict.data:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Username '{stripped}' sudah digunakan oleh pengguna lain",
                )
            update_payload["username"] = stripped
        # Jika string kosong (""), username tidak diupdate (diabaikan dengan aman)

    if not update_payload:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Tidak ada field yang dikirim untuk diperbarui",
        )

    result = (
        db.table("users")
        .update(update_payload)
        .eq("id", current_user["id"])
        .execute()
    )

    if not result.data:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User tidak ditemukan")

    logger.info(
        "Profil user '%s' (ID %d) diperbarui: %s",
        current_user["username"],
        current_user["id"],
        list(update_payload.keys()),
    )
    return result.data[0]


# ---------------------------------------------------------------
# GET /stats — Statistik user (current user yang login)
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
@router.get("/")
def list_users(
    db: Client = Depends(get_supabase),
    current_user: dict = Depends(require_roles(["superadmin", "admin"])),
):
    """
    superadmin: melihat semua user beserta nama organisasinya.
    admin: hanya melihat user dalam organisasinya sendiri.
    Response menyertakan field `organization_name` untuk kebutuhan UI.
    """
    if current_user["role"] == "superadmin":
        result = db.table("users").select("id, username, role, name, email, description, organization_id").execute()
    else:
        org_id = current_user.get("organization_id")
        if not org_id:
            return []
        result = (
            db.table("users")
            .select("id, username, role, name, email, description, organization_id")
            .eq("organization_id", org_id)
            .execute()
        )

    users = result.data

    # Kumpulkan semua org_id unik lalu ambil sekali (efisien, hindari N+1 query)
    org_ids = list({u["organization_id"] for u in users if u.get("organization_id")})
    org_map: dict[int, str] = {}
    if org_ids:
        org_res = db.table("organizations").select("id, name").in_("id", org_ids).execute()
        org_map = {o["id"]: o["name"] for o in org_res.data}

    # Gabungkan organization_name ke setiap user row
    for u in users:
        u["organization_name"] = org_map.get(u.get("organization_id"))

    return users


# ================================================================
# DYNAMIC ROUTES — Di bawah semua static route
# ================================================================

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
    # Guard: tidak boleh hapus diri sendiri
    if user_id == current_user["id"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Tidak dapat menghapus akun Anda sendiri",
        )

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
