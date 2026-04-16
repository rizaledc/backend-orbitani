import logging
from fastapi import APIRouter, Depends, HTTPException, status
from supabase import Client

from app.db.database import get_supabase
from app.models.schemas import UserCreate, UserLogin, Token, PasswordUpdate, UserOut
from app.core.security import hash_password, verify_password, create_access_token, get_current_user

logger = logging.getLogger(__name__)
router = APIRouter()


# ---------------------------------------------------------------
# POST /register
# ---------------------------------------------------------------
@router.post("/register", response_model=Token, status_code=status.HTTP_201_CREATED)
def register(data: UserCreate, db: Client = Depends(get_supabase)):
    """Mendaftarkan user baru. Username harus unik."""
    # Validasi username unik
    existing = db.table("users").select("id").eq("username", data.username).execute()
    if existing.data:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Username '{data.username}' sudah terdaftar",
        )

    # Buat user baru (sertakan nama, email, dan organization_id jika ada)
    new_user = db.table("users").insert({
        "username":        data.username,
        "password_hash":   hash_password(data.password),
        "role":            "user",
        "name":            data.name,
        "email":           data.email,
        "organization_id": data.organization_id,
    }).execute()

    user = new_user.data[0]
    logger.info("User baru terdaftar: %s (role=%s)", user["username"], user["role"])

    # Langsung kembalikan token
    token = create_access_token(data={"sub": user["username"], "role": user["role"]})
    return Token(access_token=token, role=user["role"])


# ---------------------------------------------------------------
# POST /login
# ---------------------------------------------------------------
@router.post("/login", response_model=Token)
def login(data: UserLogin, db: Client = Depends(get_supabase)):
    """Login dengan username & password, mengembalikan JWT token + role."""
    result = db.table("users").select("username, password_hash, role").eq("username", data.username).execute()
    if not result.data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Username atau password salah",
        )

    user = result.data[0]
    if not verify_password(data.password, user["password_hash"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Username atau password salah",
        )

    token = create_access_token(data={"sub": user["username"], "role": user["role"]})
    logger.info("User login: %s (role=%s)", user["username"], user["role"])
    return Token(access_token=token, role=user["role"])


# ---------------------------------------------------------------
# GET /me
# ---------------------------------------------------------------
@router.get("/me", response_model=UserOut)
def get_me(
    db: Client = Depends(get_supabase),
    current_user: dict = Depends(get_current_user)
):
    """Mendapatkan profil lengkap user yang sedang login."""
    res = (
        db.table("users")
        .select("id, username, role, name, email, description, organization_id")
        .eq("id", current_user["id"])
        .execute()
    )
    if not res.data:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User tidak ditemukan")

    return res.data[0]


# ---------------------------------------------------------------
# PUT /update-password
# ---------------------------------------------------------------
@router.put("/update-password")
def update_password(
    data: PasswordUpdate,
    db: Client = Depends(get_supabase),
    current_user: dict = Depends(get_current_user),
):
    """Mengganti password user yang sedang login. Wajib menyertakan password lama."""
    # Ambil hash password saat ini dari database
    result = (
        db.table("users")
        .select("password_hash")
        .eq("id", current_user["id"])
        .execute()
    )
    if not result.data:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User tidak ditemukan")

    stored_hash = result.data[0]["password_hash"]

    # Verifikasi password lama
    if not verify_password(data.old_password, stored_hash):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Password lama tidak cocok",
        )

    # Hash password baru dan update
    new_hash = hash_password(data.new_password)
    db.table("users").update({"password_hash": new_hash}).eq("id", current_user["id"]).execute()

    logger.info("Password user '%s' (ID %d) berhasil diubah", current_user["username"], current_user["id"])
    return {"detail": "Password berhasil diperbarui"}
