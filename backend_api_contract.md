# 📜 Orbitani Backend API Contract / Source of Truth

Dokumen ini adalah acuan **mutlak (100% presisi)** untuk tim Frontend dalam melakukan integrasi dengan backend Orbitani (FastAPI). Aturan di bawah ini tidak boleh dilanggar (terutama masalah *Trailing Slashes* dan *Header Autentikasi*).

---

## 1. Peta Endpoint & Trailing Slashes (SANGAT KRITIS)

Secara default, FastAPI sangat ketat terhadap keberadaan garis miring (slash) di akhir URL. Jika URL di bawah *tidak* memiliki slash di akhir, maka mengirimkan slash akan menghasilkan `307 Temporary Redirect` (sering diblokir oleh CORS di preflight).

| Endpoint | Method | Path yang BENAR (Copy-Paste) | Keterangan Slash |
|----------|--------|------------------------------|------------------|
| **Root** | GET | `/` | Wajib slash (Root) |
| **Auth** | POST | `/api/auth/login` | ❌ **TANPA** slash |
| **Auth** | POST | `/api/auth/register` | ❌ **TANPA** slash |
| **Auth** | GET | `/api/auth/me` | ❌ **TANPA** slash |
| **Chat** | POST | `/api/chat/ask` | ❌ **TANPA** slash |
| **Chat** | POST | `/api/chat/analyze-lahan` | ❌ **TANPA** slash |
| **Lahan**| GET | `/api/lahan/` | ✅ **WAJIB** slash |
| **Lahan**| POST | `/api/lahan/` | ✅ **WAJIB** slash |
| **Lahan**| GET | `/api/lahan/{lahan_id}/data` | ❌ **TANPA** slash |
| **ML** | POST | `/api/predict` | ❌ **TANPA** slash |

> ⚠️ **Peringatan**: Perhatikan rute `/api/lahan/`. Rute ini dideklarasikan dengan `@router.get("/")` pada prefix `/api/lahan`. Artinya, pemanggilan **wajib** menyertakan garis miring di kalimat terakhir: `fetch('/api/lahan/')`. Jika menggunakan `fetch('/api/lahan')`, akan terkena 307 Redirect dan error CORS.

---

## 2. Matriks Autentikasi (Public vs Protected)

### 🟢 Public Routes (DILARANG MENGIRIM `Authorization: Bearer <token>`)
Mengirim token (apalagi Bearer `null` atau `undefined`) ke rute di bawah ini dapat memicu penolakan tambahan dari middleware keamanan:
- `GET /`
- `POST /api/auth/login`
- `POST /api/auth/register`

### 🔴 Protected Routes (WAJIB MENGIRIM `Authorization: Bearer <token>`)
Semua rute di bawah ini dikawal oleh `Depends(get_current_user)`. Jika header tidak ada, null, atau tidak valid, backend akan mengembalikan **401 Unauthorized**.
- `GET /api/auth/me`
- `POST /api/chat/ask` (Limit: 5 RPM untuk Role User)
- `POST /api/chat/analyze-lahan` (Limit: 5 RPM untuk Role User)
- `GET /api/lahan/`
- `POST /api/lahan/`
- `GET /api/lahan/{lahan_id}/data`

---

## 3. Konfigurasi CORS Middleware (main.py)

Backend saat ini dikonfigurasi untuk menerima *Cross-Origin Requests* **hanya** dari daftar domain yang secara eksplisit diizinkan. Mode *wildcard* (`allow_origins=["*"]`) sudah dinonaktifkan demi keamanan.

Konfigurasi mutlak (saat ini):
- `allow_origins`: `["http://localhost:5173", "http://127.0.0.1:5173"]` *(Harus sama persis termasuk portnya, tanpa karakter slash / di akhir)*
- `allow_credentials`: `True` *(Mengizinkan pengiriman Cookie dan Origin Header lintas domain)*
- `allow_methods`: `["*"]` *(Semua method GET, POST, PUT, DELETE diizinkan)*
- `allow_headers`: `["*"]` *(Semua header seperti Authorization dan Content-Type diizinkan)*

---

## 4. Pydantic Schemas (Payload & Response)

Backend **TIDAK** menggunakan standar `OAuth2PasswordRequestForm` (multipart/form-data) untuk rute login. Semua rute mengharapkan **murni JSON** (Header `Content-Type: application/json`).

### A. Auth Login & Register
**Payload JSON Pydantic (`UserLogin` & `UserCreate`):**
```json
{
  "username": "contoh_user",
  "password": "contoh_password123"
}
```
**Response Sukses (FastAPI mengonversi ke `Token` model):**
```json
{
  "access_token": "eyJhbGci...<panjang>",
  "token_type": "bearer",
  "role": "user" // Atau "superadmin"
}
```

### B. Chat & Agronomist
**Payload POST `/api/chat/ask` (`ChatRequest`):**
```json
{
  "message": "Pertanyaan untuk AI Agronomist"
}
```
**Payload POST `/api/chat/analyze-lahan` (`AnalyzeLahanRequest`):**
```json
{
  "lahan_id": 12
}
```
**Response Sukses Chat:**
```json
{
  "status": "success",
  "model": "gemini-3.1-flash",
  "answer": "Jawaban dari AI..."
}
```

### C. Lahan (WebGIS)
**Payload POST `/api/lahan/` (`LahanCreate`):**
```json
{
  "nama": "Lahan Hibisc A",
  "deskripsi": "Keterangan opsional",
  "koordinat": {
    "type": "Polygon",
    "coordinates": [[[...]]] 
  }
}
```

---

## 5. Struktur Error HTTP (Exception Details)

Karena menggunakan FastAPI, error parsing internal (Pydantic) dan custom throw akan memiliki bentuk respons JSON standar dengan key `"detail"`.

**A. Error Logika Aplikasi (400, 401, 403, 404, 429)**
Dilempar secara manual menggunakan `raise HTTPException`. Nilai `detail` berupa **string**.
```json
// Contoh 401 Unauthorized
{
  "detail": "Username atau password salah"
}

// Contoh 429 Too Many Requests
{
  "detail": "Batas penggunaan tercapai: Maksimal 5 permintaan per 60 detik. Silakan coba lagi nanti."
}
```

**B. Error Validasi Pydantic (422 Unprocessable Entity)**
Dilempar otomatis oleh Pydantic (misal jika format JSON tidak lengkap atau tipe data salah). Nilai `detail` berupa **Array of Objects**.
```json
{
  "detail": [
    {
      "type": "missing",
      "loc": ["body", "message"],
      "msg": "Field required",
      "input": {}
    }
  ]
}
```

> **Pro Tip untuk Frontend:** 
> Saat menangkap error `axios`, parsing pesannya dengan cara:
> `const errorMsg = err.response?.data?.detail;`
> Pastikan untuk mengecek apakah `errorMsg` itu berupa `string` (401/400) atau `Array` (422) sebelum merendernya ke pengguna.
