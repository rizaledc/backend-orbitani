"""
chat.py
AI Agronomist endpoints for Orbitani.
Model: gemini-2.5-flash (Round-Robin + BYOK)
Endpoints:
  POST /ask            → quick Q&A chat
  POST /analyze-lahan  → satellite sync + ML prediction + deep AI analysis

Flow analyze-lahan:
  1. Ambil koordinat lahan dari DB
  2. Trigger GEE Hybrid (Sentinel-2 + Landsat-8/9 + MODIS) → simpan data fresh ke DB
  3. ML Random Forest prediction → simpan rekomendasi ke DB
  4. Kirim data + prediksi ML ke Gemini 2.5 Flash untuk analisis mendalam

Rate Limiting:
  Role 'user' : 5 RPM  |  Role 'admin'/'superadmin' : unlimited
"""
import logging
from typing import Literal, Optional
from fastapi import APIRouter, Depends, HTTPException, Header
from pydantic import BaseModel, Field
from supabase import Client

from app.db.database import get_supabase
from app.core.security import get_current_user
from app.core.rate_limiter import check_rate_limit
from app.services.gemini_service import ask_fast, ask_deep, MODEL_FAST, MODEL_DEEP

logger = logging.getLogger(__name__)
router = APIRouter()


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=1000)
    user_api_key: Optional[str] = None  # BYOK: Bring Your Own Key


class AnalyzeLahanRequest(BaseModel):
    lahan_id: int
    user_api_key: Optional[str] = None  # BYOK: Bring Your Own Key


class AiChatHistoryCreate(BaseModel):
    role: Literal["user", "ai"] = Field(..., description="Pengirim pesan: 'user' atau 'ai'")
    content: str = Field(..., min_length=1, max_length=10000)
    session_id: Optional[str] = Field(None, description="ID Sesi obrolan (UUID)")
    session_title: Optional[str] = Field(None, description="Judul sesi (dihasilkan frontend/AI)")


class SessionRenameRequest(BaseModel):
    session_title: str = Field(..., min_length=1, max_length=200, description="Judul sesi baru")


# ---------------------------------------------------------------
# POST /ask — Quick Q&A (gemini-2.5-flash)
# ---------------------------------------------------------------
@router.post("/ask")
async def ask_agronomist_api(
    req: ChatRequest,
    x_custom_gemini_key: Optional[str] = Header(None, alias="X-Custom-Gemini-Key"),
    current_user: dict = Depends(get_current_user),
):
    """
    Konsultasi tanya jawab bebas dengan AI Agronomist.
    Menggunakan gemini-2.5-flash dengan Round-Robin key pool.
    Mendukung BYOK (user_api_key opsional via body atau header).
    Prioritas key: Header X-Custom-Gemini-Key > Body user_api_key > Pool round-robin.
    Rate limit: 5 RPM untuk role 'user'.
    """
    check_rate_limit(current_user)

    # Prioritas: Header > Body > Pool (round-robin)
    effective_key = x_custom_gemini_key or req.user_api_key

    logger.info("User '%s' asking fast AI: %s...", current_user["username"], req.message[:60])
    answer = await ask_fast(req.message, user_api_key=effective_key)
    return {"status": "success", "model": MODEL_FAST, "answer": answer}


# ---------------------------------------------------------------
# POST /analyze-lahan — Deep Analysis (gemini-2.5-flash)
# ---------------------------------------------------------------
@router.post("/analyze-lahan")
async def analyze_lahan_api(
    req: AnalyzeLahanRequest,
    x_custom_gemini_key: Optional[str] = Header(None, alias="X-Custom-Gemini-Key"),
    db: Client = Depends(get_supabase),
    current_user: dict = Depends(get_current_user),
):
    """
    Menganalisis lahan menggunakan data satelit FRESH dari GEE Hybrid
    lalu dilempar ke Gemini 2.5 Flash.
    Mendukung BYOK (user_api_key opsional via body atau header).
    Prioritas key: Header X-Custom-Gemini-Key > Body user_api_key > Pool round-robin.

    Flow:
      1. Ambil koordinat centroid lahan dari tabel lahan.
      2. Trigger sinkronisasi GEE (Sentinel-2 + Landsat-8/9 + MODIS).
      3. ML Random Forest prediction → simpan ke DB.
      4. Data fresh dikirim ke Gemini 2.5 Flash untuk analisis.

    Rate limit: 5 RPM untuk role 'user'.
    """
    check_rate_limit(current_user)

    # Prioritas: Header > Body > Pool (round-robin)
    effective_key = x_custom_gemini_key or req.user_api_key

    # 1. Ambil data lahan + pastikan milik user
    lahan_res = db.table("lahan").select("*").eq("id", req.lahan_id).execute()
    if not lahan_res.data:
        raise HTTPException(status_code=404, detail="Lahan tidak ditemukan.")

    lahan = lahan_res.data[0]
    if lahan["created_by"] != current_user["id"] and current_user["role"] != "superadmin":
        raise HTTPException(status_code=403, detail="Akses ditolak ke lahan ini.")

    # 2. Hitung centroid dari koordinat lahan untuk titik sinkronisasi GEE
    koordinat = lahan.get("koordinat", {})
    coords = koordinat.get("coordinates", [[]])[0] if isinstance(koordinat, dict) else []

    if not coords:
        raise HTTPException(
            status_code=400,
            detail="Koordinat lahan tidak valid. Tidak bisa menentukan titik GEE."
        )

    # Centroid sederhana: rata-rata semua titik koordinat
    avg_lon = sum(c[0] for c in coords) / len(coords)
    avg_lat = sum(c[1] for c in coords) / len(coords)

    logger.info(
        "[METRIC_ANALYZE_LAHAN] lahan_id=%d | username=%s | lat=%f | lon=%f",
        req.lahan_id, current_user["username"], avg_lat, avg_lon,
    )

    # 3. Trigger sinkronisasi GEE Hybrid → data + ML prediction disimpan ke DB
    try:
        from app.services.gee_service import process_point_satellite_data

        gee_result = process_point_satellite_data(req.lahan_id, avg_lat, avg_lon)

        if "error" in gee_result:
            raise HTTPException(
                status_code=500,
                detail=f"Gagal sinkronisasi satelit: {gee_result['message']}"
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error GEE sync for analyze-lahan: %s", e)
        raise HTTPException(
            status_code=500,
            detail=f"Gagal memproses data satelit: {str(e)}"
        )

    # 4. Ambil data FRESH yang baru saja disimpan (record terbaru)
    fresh_result = (
        db.table("satellite_results")
        .select("*")
        .eq("lahan_id", req.lahan_id)
        .order("created_at", desc=True)
        .limit(1)
        .execute()
    )

    if not fresh_result.data:
        raise HTTPException(
            status_code=500,
            detail="Data satelit fresh gagal ditemukan setelah sinkronisasi."
        )

    sat_data = fresh_result.data[0]

    # 5. Ambil rekomendasi ML yang sudah disimpan oleh gee_service
    ml_recommendation = sat_data.get("ai_recommendation", "Pending Analysis")

    # 6. Susun prompt agronomis dengan data satelit FRESH + prediksi ML
    prompt = f"""Tolong berikan Orbitani Smart Analysis untuk data lahan satelit TERBARU ini:

Sumber Data: Sentinel-2 SR Harmonized (10m) + Landsat-8/9 L2 (30m) + MODIS Terra (1km fallback)
Waktu Ekstraksi: {sat_data.get('created_at', 'N/A')}

- Nitrogen (N)      : {sat_data.get('n', 'N/A')}
- Fosfor (P)        : {sat_data.get('p', 'N/A')}
- Kalium (K)        : {sat_data.get('k', 'N/A')}
- pH Tanah          : {sat_data.get('ph', 'N/A')}
- Suhu Lahan        : {sat_data.get('temperature', 'N/A')} °C
- Kelembapan (NDTI) : {sat_data.get('humidity', 'N/A')}
- Curah Hujan (1th) : {sat_data.get('rainfall', 'N/A')} mm

Model Random Forest kami merekomendasikan tanaman: **{ml_recommendation}**.

Berikan analisis lanjutan berdasarkan rekomendasi ini:
1. Apakah rekomendasi tanaman tersebut sesuai dengan kondisi hara dan iklim di atas?
2. Berikan 2 poin aksi yang harus segera dilakukan petani untuk memaksimalkan hasil panen."""

    logger.info(
        "[METRIC_DEEP_ANALYSIS] lahan_id=%d | username=%s | recommendation=%s | model=%s",
        req.lahan_id, current_user["username"], ml_recommendation, MODEL_DEEP,
    )
    answer = await ask_deep(prompt, user_api_key=effective_key)

    return {
        "status": "success",
        "model": MODEL_DEEP,
        "ml_recommendation": ml_recommendation,
        "satellite_data": sat_data,
        "ai_analysis": answer,
    }


# ---------------------------------------------------------------
# GET /history — Ambil riwayat AI Chat milik user yang login
# ---------------------------------------------------------------
@router.get("/history")
def get_ai_chat_history(
    session_id: Optional[str] = None,
    db: Client = Depends(get_supabase),
    current_user: dict = Depends(get_current_user),
):
    """
    Mengambil seluru riwayat obrolan AI Agronomist milik current_user.
    Bila session_id dikirimkan, hanya mengembalikan chat untuk sesi tersebut.
    Diurutkan berdasarkan created_at secara Ascending (lama ke baru).
    """
    try:
        query = (
            db.table("ai_chat_history")
            .select("id, role, content, created_at, session_id, session_title")
            .eq("user_id", current_user["id"])
        )
        if session_id:
            query = query.eq("session_id", session_id)
            
        res = query.order("created_at", desc=False).execute()
        return {"status": "success", "data": res.data}
    except Exception as e:
        logger.error("Error fetching AI chat history: %s", e)
        raise HTTPException(status_code=500, detail="Gagal mengambil riwayat chat AI.")


# ---------------------------------------------------------------
# GET /sessions — Mengambil daftar sesi unik milik user
# ---------------------------------------------------------------
@router.get("/sessions")
def get_ai_chat_sessions(
    db: Client = Depends(get_supabase),
    current_user: dict = Depends(get_current_user),
):
    """
    Mengambil daftar riwayat sesi terbaru (dikembalikan list of session_id & session_title).
    Group by session_id dihitung di backend karena max row hanya 25 (sangat ringan).
    """
    try:
        res = (
            db.table("ai_chat_history")
            .select("session_id, session_title, created_at")
            .eq("user_id", current_user["id"])
            .order("created_at", desc=True)
            .execute()
        )
        
        seen_sessions = set()
        sessions = []
        for row in res.data:
            sid = row.get("session_id")
            # Skip messages without session tracking
            if not sid:
                continue
            if sid not in seen_sessions:
                seen_sessions.add(sid)
                sessions.append({
                    "session_id": sid,
                    "session_title": row.get("session_title", "Chat Baru"),
                    "last_active": row.get("created_at")
                })
                
        return {"status": "success", "data": sessions}
    except Exception as e:
        logger.error("Error fetching AI sessions: %s", e)
        raise HTTPException(status_code=500, detail="Gagal mengambil daftar sesi AI.")


# ---------------------------------------------------------------
# PATCH /sessions/{session_id} — Mengubah judul sesi
# ---------------------------------------------------------------
@router.patch("/sessions/{session_id}")
def rename_ai_chat_session(
    session_id: str,
    req: SessionRenameRequest,
    db: Client = Depends(get_supabase),
    current_user: dict = Depends(get_current_user),
):
    """
    Mengubah judul sesi pada semua pesan yang terangkum dalam session_id yang diberikan.
    Hanya bisa dilakukan bila user_id chat sesuai dengan token pengguna.
    """
    try:
        updated = (
            db.table("ai_chat_history")
            .update({"session_title": req.session_title})
            .eq("user_id", current_user["id"])
            .eq("session_id", session_id)
            .execute()
        )
        
        if not updated.data:
            raise HTTPException(status_code=404, detail="Sesi tidak ditemukan atau tidak memiliki akses.")
            
        return {"status": "success", "message": "Judul sesi berhasil diperbarui", "session_title": req.session_title}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error renaming AI session: %s", e)
        raise HTTPException(status_code=500, detail="Gagal merename sesi AI.")


# ---------------------------------------------------------------
# DELETE /sessions/{session_id} — Menghapus obrolan dalam sesi
# ---------------------------------------------------------------
@router.delete("/sessions/{session_id}")
def delete_ai_chat_session(
    session_id: str,
    db: Client = Depends(get_supabase),
    current_user: dict = Depends(get_current_user),
):
    """
    Menghapus semua baris pesan (riwayat) yang terkait dengan session_id yang diberikan.
    Hanya bisa dilakukan bila user_id chat sesuai dengan token pengguna (cross-user protection).
    """
    try:
        deleted = (
            db.table("ai_chat_history")
            .delete()
            .eq("user_id", current_user["id"])
            .eq("session_id", session_id)
            .execute()
        )
        
        # Jika deleted.data kosong, berarti tidak ada baris yang dihapus
        # (Namun Supabase v2 Python client mengembalikan hasil pada .data)
        if not deleted.data:
            raise HTTPException(status_code=404, detail="Sesi tidak ditemukan atau tidak memiliki akses untuk dihapus.")
            
        return {"status": "success", "message": "Sesi berhasil dihapus."}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error deleting AI session: %s", e)
        raise HTTPException(status_code=500, detail="Gagal menghapus sesi AI.")




# ---------------------------------------------------------------
# POST /history — Simpan 1 baris pesan baru ke ai_chat_history
# Retensi: max 25 pesan per user (admin/user). Superadmin unlimited.
# ---------------------------------------------------------------
MAX_CHAT_HISTORY = 25

@router.post("/history")
def save_ai_chat_history(
    req: AiChatHistoryCreate,
    db: Client = Depends(get_supabase),
    current_user: dict = Depends(get_current_user),
):
    """
    Menyimpan 1 baris pesan baru ke tabel ai_chat_history.
    Dipanggil oleh frontend setiap kali:
      - User mengirim pesan (role='user')
      - Gemini selesai menjawab (role='ai')
    user_id diambil otomatis dari token JWT (Integer).

    Kebijakan Retensi (Auto-Pruning):
      - Superadmin: Unlimited (tidak ada limit).
      - Admin / User: Maksimal 25 pesan. Jika sudah penuh,
        pesan tertua dihapus sebelum pesan baru disimpan.
    """
    user_id = current_user["id"]
    user_role = current_user.get("role", "user")

    try:
        # --- Auto-Pruning: hanya untuk role non-superadmin ---
        if user_role != "superadmin":
            # 1. Hitung total pesan milik user ini
            count_res = (
                db.table("ai_chat_history")
                .select("id", count="exact")
                .eq("user_id", user_id)
                .execute()
            )
            total_messages = count_res.count if count_res.count is not None else len(count_res.data)

            # 2. Jika sudah mencapai limit, hapus pesan tertua
            if total_messages >= MAX_CHAT_HISTORY:
                oldest_res = (
                    db.table("ai_chat_history")
                    .select("id")
                    .eq("user_id", user_id)
                    .order("created_at", desc=False)
                    .limit(1)
                    .execute()
                )
                if oldest_res.data:
                    oldest_id = oldest_res.data[0]["id"]
                    db.table("ai_chat_history").delete().eq("id", oldest_id).execute()
                    logger.info(
                        "Auto-pruned oldest chat message (id=%s) for user_id=%d (total was %d)",
                        oldest_id, user_id, total_messages,
                    )

        # --- Simpan pesan baru ---
        new_row = {
            "user_id": user_id,
            "role": req.role,
            "content": req.content,
            "session_id": req.session_id,
            "session_title": req.session_title,
        }
        saved = db.table("ai_chat_history").insert(new_row).execute()
        return {"status": "success", "data": saved.data[0]}

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error saving AI chat history: %s", e)
        raise HTTPException(status_code=500, detail="Gagal menyimpan pesan chat AI.")
