from pydantic import BaseModel, Field


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

    model_config = {"from_attributes": True}


class RoleUpdate(BaseModel):
    role: str = Field(..., description="Role baru: admin atau user")
