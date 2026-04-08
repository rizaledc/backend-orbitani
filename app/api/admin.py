"""
admin.py
Superadmin endpoints for ML Feedback collection and model retraining.

Endpoints:
  POST /feedback        — Submit ground truth correction (any authenticated user)
  GET  /feedback        — List all feedback (superadmin only)
  POST /train-model     — Trigger background model retrain (superadmin only)
  GET  /llm-status      — Round-Robin key pool status (superadmin only)
"""
import logging
from fastapi import APIRouter, Depends, BackgroundTasks, HTTPException, status
from supabase import Client

from app.db.database import get_supabase
from app.core.security import get_current_user, require_roles
from app.models.schemas import MlFeedbackCreate, MlFeedbackOut, RetrainResponse

logger = logging.getLogger(__name__)
router = APIRouter()


# ---------------------------------------------------------------
# POST /feedback — Submit ground truth dari petani/admin
# ---------------------------------------------------------------
@router.post("/feedback", response_model=MlFeedbackOut, status_code=status.HTTP_201_CREATED)
def submit_feedback(
    data: MlFeedbackCreate,
    db: Client = Depends(get_supabase),
    current_user: dict = Depends(get_current_user),
):
    """
    User/admin mengirimkan koreksi prediksi (ground truth) berupa tanaman
    yang sebenarnya ditanam di lahan tersebut.
    """
    # Pastikan lahan ada dan user berhak mengaksesnya
    lahan_res = db.table("lahan").select("id, created_by").eq("id", data.lahan_id).execute()
    if not lahan_res.data:
        raise HTTPException(status_code=404, detail="Lahan tidak ditemukan.")
    lahan = lahan_res.data[0]
    if lahan["created_by"] != current_user["id"] and current_user["role"] not in ("admin", "superadmin"):
        raise HTTPException(status_code=403, detail="Akses ditolak ke lahan ini.")

    payload = {
        "lahan_id":           data.lahan_id,
        "n":                  data.n,
        "p":                  data.p,
        "k":                  data.k,
        "temperature":        data.temperature,
        "humidity":           data.humidity,
        "ph":                 data.ph,
        "rainfall":           data.rainfall,
        "ai_recommendation":  data.ai_recommendation,
        "actual_crop":        data.actual_crop,
        "submitted_by":       current_user["id"],
    }
    result = db.table("ml_feedback").insert(payload).execute()
    logger.info(
        "ML Feedback diterima dari user '%s' untuk lahan_id=%d — actual_crop=%s",
        current_user["username"], data.lahan_id, data.actual_crop,
    )
    return result.data[0]


# ---------------------------------------------------------------
# GET /feedback — Daftar semua feedback (superadmin)
# ---------------------------------------------------------------
@router.get("/feedback", response_model=list[MlFeedbackOut])
def list_feedback(
    db: Client = Depends(get_supabase),
    _current_user: dict = Depends(require_roles(["superadmin"])),
):
    """Menampilkan semua ground truth feedback. Hanya superadmin."""
    result = db.table("ml_feedback").select("*").order("created_at", desc=True).execute()
    return result.data


# ---------------------------------------------------------------
# POST /train-model — Hot-swap retrain (superadmin)
# ---------------------------------------------------------------
@router.post("/train-model", response_model=RetrainResponse)
async def trigger_retrain(
    background_tasks: BackgroundTasks,
    _current_user: dict = Depends(require_roles(["superadmin"])),
):
    """
    Trigger retraining model Random Forest di background.
    Menggabungkan dataset CSV awal + data ml_feedback terbaru.
    Hot-swap: model aktif diganti tanpa restart Uvicorn.
    """
    from app.services.retrain_service import retrain_model_full

    background_tasks.add_task(retrain_model_full)
    logger.info(
        "Superadmin '%s' memicu model retraining di background.",
        _current_user["username"],
    )
    return RetrainResponse(
        status="accepted",
        message="Proses retraining dimulai di background. Periksa log server untuk hasilnya.",
    )
