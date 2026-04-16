# Laporan Serah Terima Teknis: Backend ke Frontend

**Tujuan Dokumen:** Panduan integrasi API Backend Orbitani untuk Tim/Agen Frontend (Vercel/Next.js)

---

## 1. Informasi Dasar & Autentikasi

### Base URL Production
- **Azure Environment:** `https://<app-name>.azurewebsites.net` *(Harap sesuaikan dengan URL production Azure yang aktif)*
- **Staging/Local:** `http://localhost:8000`

### Mekanisme Autentikasi
Semua endpoint yang membutuhkan otorisasi wajib menyertakan token JWT pada header HTTP.
```http
Authorization: Bearer <access_token>
```

### RBAC (Role-Based Access Control) & Multi-Tenancy
Sistem backend menerapkan multi-tenancy berbasis `organization_id` yang membatasi visibilitas data:
- **Tenant Isolation:** Data lahan, histori chat, dan analitik hanya dapat diakses oleh user yang berada di dalam `organization_id` yang sama.
- **Roles:** Akses CRUD terhadap management user atau lahan terbatas pada *Role* tertentu. Akses baca/tulis dibatasi secara ketat oleh sistem RBAC ini. Frontend tidak perlu melakukan filter data lahan secara manual di sisi client; cukup panggil endpoint dan backend hanya akan me-return data spesifik milik organisasi user yang login.

---

## 2. Struktur Payload Spesifik (PENTING)

> **PENGUMUMAN PENTING:** Tolong ikuti standar format berikut, karena ketidaksesuaian akan menyebabkan HTTP 422 Unprocessable Entity atau API error.

### `POST /api/lahan/`
**Aturan:** Data koordinat **WAJIB** menggunakan format standar **GeoJSON Polygon**.
**Catatan:** Array multi-dimensi dimana titik awal dan akhir pada koordinat harus ditutup di lokasi yang sama.
```json
{
  "name": "Lahan Bawang",
  "location": {
    "type": "Polygon",
    "coordinates": [
      [
        [109.0, -7.7],
        [109.1, -7.7],
        [109.1, -7.6],
        [109.0, -7.6],
        [109.0, -7.7]  // Titik awal dan akhir harus SAMA
      ]
    ]
  }
}
```

### `POST /api/chat/ask` & `POST /api/chat/analyze-lahan`
**Aturan:** Format balasan dari AI sudah dibersihkan dari sintaks LaTeX (seperti `\[...\]` atau `\(...\)`).
**Tindakan Frontend:** Tidak perlu menggunakan library `math-parser` atau `KaTeX` khusus. Cukup render *response text* menggunakan standar Markdown rendering biasa (misalnya menggunakan `react-markdown`).

### `GET /api/lahan/analytics`
**Aturan:** Endpoint ini dirancang khusus agar langsung kompatibel dengan library *chart/graph* (seperti **Recharts**).
**Format Output:** Merupakan Array of Objects secara langsung.
```json
[
  { "date": "2026-04-01", "NDVI": 0.82, "nitrogen": 45 },
  { "date": "2026-04-02", "NDVI": 0.85, "nitrogen": 48 }
]
```

---

## 3. Panduan Penanganan Latency (Loading State)

**INSTRUKSI TEGAS:** Agen Frontend diwajibkan untuk membuat UI Loading State (seperti *Spinner* atau *Skeleton*) yang memadai pada endpoint berikut. Timeout dari framework frontend juga harus disesuaikan.

1. **`POST /api/chat/analyze-lahan`**
   - **Potensi Latensi:** 10 - 30 detik.
   - **Penyebab:** Melakukan orkestrasi pipeline AI mulai dari pengambilan data Google Earth Engine (GEE), prediksi ML, hingga prompt generative text ke Gemini LLM.

2. **`GET /api/lahan/{lahan_id}/data` (Mode Hybrid)**
   - **Kondisi Khusus:** Jika dikirim dengan parameter `lat` & `lon`.
   - **Potensi Latensi:** Lebih lama dari request GET standar.
   - **Penyebab:** Menyebabkan *live fetch* langsung ke server satelit Google Earth Engine, bukan membaca *cache* dari database secara instan.

---

## 4. Daftar API Utama yang Telah Teruji (Checklist)

Berikut adalah tabel ringkasan endpoint untuk memandu alur integrasi frontend:

| Kategori | Method | Endpoint | Deskripsi Singkat |
| :--- | :--- | :--- | :--- |
| **Auth** | POST | `/api/auth/login` | Endpoint login untuk mendapatkan JWT Token. |
| **Auth** | POST | `/api/auth/register` | Membuat akun baru. |
| **Auth** | GET | `/api/auth/me` | Mengambil data pengguna yang sedang terautentikasi. |
| **User & Org Management** | GET/POST | `/api/organizations/` | Mengelola data perusahaan/organisasi (Role Admin). |
| **User & Org Management** | GET/POST | `/api/users/` | Mengelola akun user di dalam tenant/organisasi tertentu. |
| **Lahan & Analytics** | GET/POST | `/api/lahan/` | Menyimpan dan mengambil daftar lahan (memerlukan payload GeoJSON). |
| **Lahan & Analytics** | GET | `/api/lahan/analytics` | Mengambil time-series data analitik lahan untuk Charting. |
| **AI Chat** | POST | `/api/chat/ask` | Bertanya instruksi general (Tanya Jawab biasa) ke AI. |
| **AI Chat** | POST | `/api/chat/analyze-lahan` | Chat AI diiringi oleh konteks data satelit Lahan dan ML. |
| **Live Chat** | GET/POST | `/api/chat-live/messages`<br>`/api/chat-live/contacts` | Interaksi chat manusia sesama pengguna via API REST. |
| **MLOps** | POST | `/api/admin/feedback` | Memberikan rating / Ground-truth aktual untuk model ML. |
| **MLOps** | POST | `/api/admin/train-model` | Triggers retraining model ML secara dinamis di background. |

---

## 5. Catatan Khusus Fallback Model

Di *backend*, kami telah menangani masalah *Rate-Limit* maupun server AI yang *down* (Error HTTP 500/503 dari server layanan Gemini). Sistem sudah memakai mekanisme rotasi *API Key pool* dan juga implementasi *Fallback Model* agar chat dapat tetap beroperasi secara *seamless* di mayoritas waktu.

**Pesan untuk Agen/Dev Frontend:**
Anda tidak perlu menambahkan logic retry/polling yang rumit untuk mem-bypass *error* AI dari frontend. Jika Anda menerima balasan HTTP `503 Service Unavailable` atau `500` dari backend kami pada route `/api/chat/*`, hal ini mengindikasikan seluruh pool AI dan *rate-limit* benar-benar *exhausted*. 

Frontend **cukup** menangani *error handling* standar dengan menampilkan pesan ke user (seperti lewat *toast* atau *snackbar*):
> *"Server sedang sibuk. Silakan coba kembali beberapa saat lagi."*
