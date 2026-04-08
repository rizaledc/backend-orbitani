"""
organizations.py
Organization management endpoints (superadmin only).
"""
import logging
from fastapi import APIRouter, Depends, HTTPException, status
from supabase import Client

from app.db.database import get_supabase
from app.core.security import require_roles
from app.models.schemas import OrganizationCreate, OrganizationOut

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/", response_model=OrganizationOut, status_code=status.HTTP_201_CREATED)
def create_organization(
    data: OrganizationCreate,
    db: Client = Depends(get_supabase),
    _current_user: dict = Depends(require_roles(["superadmin"])),
):
    """Membuat organisasi baru. Hanya superadmin."""
    existing = db.table("organizations").select("id").eq("name", data.name).execute()
    if existing.data:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Organisasi dengan nama '{data.name}' sudah ada.",
        )
    result = db.table("organizations").insert({"name": data.name}).execute()
    logger.info("Organisasi baru dibuat: %s", data.name)
    return result.data[0]


@router.get("/", response_model=list[OrganizationOut])
def list_organizations(
    db: Client = Depends(get_supabase),
    _current_user: dict = Depends(require_roles(["superadmin"])),
):
    """Menampilkan semua organisasi. Hanya superadmin."""
    result = db.table("organizations").select("*").order("created_at").execute()
    return result.data


@router.get("/{org_id}/users")
def get_org_users(
    org_id: int,
    db: Client = Depends(get_supabase),
    _current_user: dict = Depends(require_roles(["superadmin"])),
):
    """Menampilkan semua user dalam satu organisasi. Hanya superadmin."""
    result = (
        db.table("users")
        .select("id, username, role, name, email, organization_id")
        .eq("organization_id", org_id)
        .execute()
    )
    return {"status": "success", "data": result.data}


@router.delete("/{org_id}", status_code=status.HTTP_200_OK)
def delete_organization(
    org_id: int,
    db: Client = Depends(get_supabase),
    _current_user: dict = Depends(require_roles(["superadmin"])),
):
    """Menghapus organisasi berdasarkan ID. Hanya superadmin."""
    existing = db.table("organizations").select("id, name").eq("id", org_id).execute()
    if not existing.data:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Organisasi tidak ditemukan.")
    name = existing.data[0]["name"]
    db.table("organizations").delete().eq("id", org_id).execute()
    logger.info("Organisasi '%s' (id=%d) dihapus", name, org_id)
    return {"detail": f"Organisasi '{name}' berhasil dihapus."}
