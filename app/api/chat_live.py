import logging
from typing import List, Dict
from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field
from supabase import Client

from app.db.database import get_supabase
from app.core.security import get_current_user

logger = logging.getLogger(__name__)
router = APIRouter()

# ---------------------------------------------------------------------------
# In-Memory WebSocket Connection Manager
# ---------------------------------------------------------------------------
class ConnectionManager:
    def __init__(self):
        # Menyimpan active connections berdasarkan user_id: {user_id: [WebSocket, ...]}
        self.active_connections: Dict[int, List[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, user_id: int):
        await websocket.accept()
        if user_id not in self.active_connections:
            self.active_connections[user_id] = []
        self.active_connections[user_id].append(websocket)
        logger.info(f"WebSocket connected for user {user_id}")

    def disconnect(self, websocket: WebSocket, user_id: int):
        if user_id in self.active_connections:
            if websocket in self.active_connections[user_id]:
                self.active_connections[user_id].remove(websocket)
            if not self.active_connections[user_id]:
                del self.active_connections[user_id]
            logger.info(f"WebSocket disconnected for user {user_id}")

    async def send_personal_message(self, message: dict, user_id: int):
        """Kirim event websocket langsung ke koneksi user_id yang aktif."""
        if user_id in self.active_connections:
            for connection in self.active_connections[user_id]:
                try:
                    await connection.send_json(message)
                except Exception as e:
                    logger.warning(f"Failed to send to {user_id}: {e}")

manager = ConnectionManager()

# ---------------------------------------------------------------------------
# REST API Schemas
# ---------------------------------------------------------------------------
class MessageSend(BaseModel):
    receiver_id: int
    message_text: str = Field(..., min_length=1)

# ---------------------------------------------------------------------------
# 1. GET Contacts (Daftar Lawan Bicara)
# ---------------------------------------------------------------------------
@router.get("/contacts")
def get_contacts(
    db: Client = Depends(get_supabase),
    current_user: dict = Depends(get_current_user)
):
    my_role = current_user["role"]
    my_id = current_user["id"]
    
    if my_role == "user":
        # User HANYA boleh melihat admin (tidak bisa melihat superadmin atau user lain)
        res = db.table("users").select("id, username, role, name, description").eq("role", "admin").execute()
    else:
        # Admin / Superadmin boleh melihat semuanya KECUALI dirinya sendiri
        res = db.table("users").select("id, username, role, name, description").neq("id", my_id).execute()
        
    return res.data

# ---------------------------------------------------------------------------
# 2. GET Message History
# ---------------------------------------------------------------------------
@router.get("/messages/{contact_id}")
def get_message_history(
    contact_id: int,
    db: Client = Depends(get_supabase),
    current_user: dict = Depends(get_current_user)
):
    my_id = current_user["id"]
    
    # Validasi Hak Akses Arahan Chat (User -> Admin limit)
    if current_user["role"] == "user":
        contact_res = db.table("users").select("role").eq("id", contact_id).execute()
        if not contact_res.data or contact_res.data[0]["role"] != "admin":
            raise HTTPException(status_code=403, detail="Anda hanya dapat bertukar pesan dengan Admin.")
            
    # Ambil riwayat chat antara my_id dan contact_id terurut berdasarkan waktu
    res = db.table("messages").select("*").or_(
        f"and(sender_id.eq.{my_id},receiver_id.eq.{contact_id}),"
        f"and(sender_id.eq.{contact_id},receiver_id.eq.{my_id})"
    ).order("timestamp", desc=False).execute()
    
    return res.data

# ---------------------------------------------------------------------------
# 3. POST Kirim Pesan (REST Fallback)
# ---------------------------------------------------------------------------
@router.post("/messages")
async def send_message_rest(
    payload: MessageSend,
    db: Client = Depends(get_supabase),
    current_user: dict = Depends(get_current_user)
):
    my_id = current_user["id"]
    receiver_id = payload.receiver_id
    
    # Validasi Hak Akses
    if current_user["role"] == "user":
        contact_res = db.table("users").select("role").eq("id", receiver_id).execute()
        if not contact_res.data or contact_res.data[0]["role"] != "admin":
            raise HTTPException(status_code=403, detail="Anda hanya dapat mengirim pesan ke Admin.")
            
    # Simpan ke Database
    new_msg = {
        "sender_id": my_id,
        "receiver_id": receiver_id,
        "message_text": payload.message_text
    }
    
    saved = db.table("messages").insert(new_msg).execute()
    result_data = saved.data[0]
    
    # Broadcast via WebSocket secara realtime jika lawan bicara sedang online
    ws_payload = {
        "event": "new_message",
        "data": result_data
    }
    await manager.send_personal_message(ws_payload, receiver_id)
    
    return result_data

# ---------------------------------------------------------------------------
# 4. WebSocket Endpoint -> /api/chat-live/ws/{client_id}
# ---------------------------------------------------------------------------
@router.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: int):
    # Idealnya, gunakan token dari client untuk get_current_user. 
    # Karena API Websocket agak tricky dengan Header Authorization, kita
    # passing client_id pada URL (pastikan Frontend menjaga id-nya).
    await manager.connect(websocket, client_id)
    try:
        while True:
            # Tetap terbuka dan mendengarkan (walaupun client mungkin mengirim pasif payload)
            # Jika user send via REST, ini akan didengarkan.
            # Jika user send via WebSocket:
            data = await websocket.receive_json()
            # Asumsi: data harus { "receiver_id": 1, "message_text": "halo" }
            # Kita bisa memicu REST saving logic di sini, namun direkomendasikan
            # frontend POST via REST saja, dan WebSocket hanya untuk Listen Broadcast,
            # agar lebih mudah me-manage dependency (seperti `get_supabase`).
            pass
            
    except WebSocketDisconnect:
        manager.disconnect(websocket, client_id)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket, client_id)
