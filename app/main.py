import logging
from contextlib import asynccontextmanager

from dotenv import load_dotenv
load_dotenv()  # Load .env sebelum import lainnya

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.db.database import supabase
from app.core.security import hash_password

from app.api.endpoints import router as predict_router
from app.api.auth import router as auth_router
from app.api.users import router as users_router
from app.api.chat import router as chat_router
from app.api.chat_live import router as chat_live_router
from app.api.lahan import router as lahan_router
from app.api.history import router as history_router

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------
# Lifespan: startup & shutdown events
# ---------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- STARTUP ---
    # Seed akun superadmin jika tabel users masih kosong
    try:
        result = supabase.table("users").select("id").limit(1).execute()
        if not result.data:
            supabase.table("users").insert({
                "username": "superadmin",
                "password_hash": hash_password("Superadmin123!"),
                "role": "superadmin",
            }).execute()
            logger.info("Superadmin default berhasil dibuat (username: superadmin)")
        else:
            logger.info("Tabel users sudah berisi data, skip seeding")
    except Exception as e:
        logger.warning("Gagal seed superadmin: %s", e)

    logger.info("Orbitani backend startup complete")
    yield  # Aplikasi berjalan

    # --- SHUTDOWN ---
    logger.info("Orbitani backend shutting down")


# ---------------------------------------------------------------
# 1. Inisialisasi Aplikasi
# ---------------------------------------------------------------
app = FastAPI(
    title="Orbitani Backend ML API",
    description="Production-ready FastAPI backend for ML prediction (Lahan Hibisc)",
    version="1.0.0",
    lifespan=lifespan,
)

# ---------------------------------------------------------------
# 2. CORS Middleware
# ---------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------
# 3. Mendaftarkan semua Router
# ---------------------------------------------------------------
app.include_router(predict_router, prefix="/api", tags=["Machine Learning"])
app.include_router(auth_router, prefix="/api/auth", tags=["Authentication"])
app.include_router(users_router, prefix="/api/users", tags=["User Management"])
app.include_router(chat_router, prefix="/api/chat", tags=["AI Agronomist"])
app.include_router(chat_live_router, prefix="/api/chat-live", tags=["Live Chat Human"])
app.include_router(lahan_router, prefix="/api/lahan", tags=["WebGIS Lahan"])
app.include_router(history_router, prefix="/api/history", tags=["History Ledger"])

# ---------------------------------------------------------------
# 4. Endpoint Halaman Depan
# ---------------------------------------------------------------
@app.get("/")
def root():
    return {"message": "Welcome to Orbitani ML API"}