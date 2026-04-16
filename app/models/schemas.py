from pydantic import BaseModel, Field
from typing import Optional


# ---------------------------------------------------------------
# ML Schemas
# ---------------------------------------------------------------
class LahanInput(BaseModel):
    N: float
    P: float
    K: float
    temperature: float
    humidity: float
    ph: float
    rainfall: float


# ---------------------------------------------------------------
# Auth Schemas
# ---------------------------------------------------------------
class UserCreate(BaseModel):
    username: str = Field(
        ...,
        min_length=3,
        max_length=50,
        pattern=r"^[a-zA-Z0-9_]+$",
        description="Username hanya boleh huruf, angka, dan underscore (3-50 karakter)",
    )
    password: str = Field(
        ...,
        min_length=8,
        max_length=72,
        description="Password minimal 8 karakter, maksimal 72 (batas bcrypt)",
    )
    name: str | None = Field(default=None, description="Nama Lengkap")
    email: str | None = Field(default=None, description="Alamat Email")
    organization_id: int | None = Field(default=None, description="ID Organisasi (wajib untuk role admin/user)")


class UserLogin(BaseModel):
    username: str = Field(..., min_length=1, max_length=50)
    password: str = Field(..., min_length=1, max_length=72)


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    role: str


class UserOut(BaseModel):
    id: int
    username: str
    role: str
    name: str | None = None
    email: str | None = None
    description: str | None = None
    organization_id: int | None = None

    model_config = {"from_attributes": True}


class RoleUpdate(BaseModel):
    role: str = Field(..., description="Role baru: admin atau user")


class ProfileUpdate(BaseModel):
    # Semua field Optional — frontend boleh kirim sebagian atau seluruhnya.
    # Nilai None  = field tidak dikirim (tidak diupdate).
    # Nilai ""    = field dikirim kosong (disimpan sebagai "" di DB).
    name: str | None = Field(default=None, max_length=100, description="Nama lengkap")
    username: str | None = Field(
        default=None,
        max_length=50,
        description="Username baru (huruf, angka, underscore, 3-50 karakter). Validasi format dilakukan di endpoint.",
    )
    description: str | None = Field(default=None, max_length=500, description="Bio / deskripsi singkat")


class PasswordUpdate(BaseModel):
    old_password: str = Field(..., min_length=1, max_length=72, description="Password lama")
    new_password: str = Field(
        ...,
        min_length=8,
        max_length=72,
        description="Password baru (minimal 8 karakter, maksimal 72)",
    )


# ---------------------------------------------------------------
# Organization Schemas
# ---------------------------------------------------------------
class OrganizationCreate(BaseModel):
    name: str = Field(..., min_length=2, max_length=100)


class OrganizationOut(BaseModel):
    id: int
    name: str
    created_at: str | None = None

    model_config = {"from_attributes": True}


# ---------------------------------------------------------------
# ML Feedback Schemas
# ---------------------------------------------------------------
class MlFeedbackCreate(BaseModel):
    lahan_id: int
    n: float
    p: float
    k: float
    temperature: float
    humidity: float
    ph: float
    rainfall: float
    ai_recommendation: Optional[str] = None
    actual_crop: str = Field(..., description="Tanaman yang sebenarnya ditanam (Ground Truth)")


class MlFeedbackOut(BaseModel):
    id: int
    lahan_id: int | None = None
    n: float | None = None
    p: float | None = None
    k: float | None = None
    temperature: float | None = None
    humidity: float | None = None
    ph: float | None = None
    rainfall: float | None = None
    ai_recommendation: str | None = None
    actual_crop: str
    submitted_by: int | None = None
    created_at: str | None = None

    model_config = {"from_attributes": True}


# ---------------------------------------------------------------
# Retrain Schemas
# ---------------------------------------------------------------
class RetrainResponse(BaseModel):
    status: str
    message: str
    model_path: str | None = None
