# Frontend Requirements Gap Analysis & Implementation Plan

Berdasarkan rencana fitur frontend yang sangat komprehensif, backend Orbitani saat ini sudah memiliki pondasi yang kuat, **namun belum 100% siap** untuk mendukung semua fitur secara langsung. Ada beberapa "GAP" (celah) endpoint API yang harus ditambahkan.

Berikut adalah analisis dukungan backend saat ini dan rencana implementasinya.

## 1. Sistem Pemetaan Interaktif (WebGIS)
- **Status Selesai (🟢)**: Sinkronisasi satelit & ekstraksi GEE (`POST /api/analyze-location`).
- **GAP (❌)**: Frontend butuh endpoint untuk mengambil data poligon lahan dan data historis titik satelit untuk di-*plot* ke peta Leaflet.
- **Rencana Implementasi**:
  - `GET /api/lahan`: List semua lahan milik user.
  - `POST /api/lahan`: Create lahan baru (menyimpan poligon GeoJSON baru).
  - `GET /api/lahan/{id}/data`: Mengambil semua titik history satelit (`satellite_results`) di lahan tersebut untuk ditampilkan markernya.

## 2. Analisis Agronomi & Machine Learning
- **Status Selesai (🟢)**: 100% didukung. Prediksi ML Random Forest, kalibrasi NPK, pH, cuaca, rekomendasi tanaman siap pakai.

## 3. Integrasi Pakar Agronomi AI (Gemini)
- **Status Selesai (❌)**: Belum ada sama sekali.
- **Rencana Implementasi**:
  - `POST /api/chat/ask`: Endpoint proxy yang meneruskan pertanyaan user ke model **Google Gemini Pro** (via library `google-generativeai`).
  - AI prompt akan di-*inject* dengan konteks bahwa agen adalah "Pakar Agronomi Orbitani" dan diberikan input data spesifik (contoh: *Berikan resep pupuk untuk Lahan A dengan N=10, pH=6*).

## 4. Manajemen Pengguna (RBAC) & Fitur Pendukung
- **Status Selesai (🟡)**: JWT Token, role [user](file:///c:/backend-orbitani/app/api/users.py#16-24)/[admin](file:///c:/backend-orbitani/app/core/security.py#71-79)/[superadmin](file:///c:/backend-orbitani/app/core/security.py#71-79), dan CRUD User (hanya superadmin) sudah siap.
- **GAP (❌)**: Frontend butuh endpoint profil *Current User* dan *Statistik* jumlah pemakaian.
- **Rencana Implementasi**:
  - `GET /api/auth/me`: Mengembalikan data profil user yang sedang login beserta role-nya.
  - `GET /api/users/stats`: Membaca jumlah total baris `satellite_results` yang pernah di-query oleh user tersebut.

---

## Ringkasan Rencana Kerja (Next Steps)

### A. Komponen Baru
1. [NEW] `app/services/gemini_service.py` — Integrasi Google Gemini API.
2. [NEW] `app/api/lahan.py` — Router khusus Lahan & WebGIS.
3. [NEW] `app/api/chat.py` — Router khusus konsultasi AI.

### B. Modifikasi Komponen Lama
1. [MODIFY] [app/api/auth.py](file:///c:/backend-orbitani/app/api/auth.py) — Tambah endpoint `/me`.
2. [MODIFY] [app/main.py](file:///c:/backend-orbitani/app/main.py) — Register router baru (`chat_router`, `lahan_router`).
3. [MODIFY] [requirements.txt](file:///c:/backend-orbitani/requirements.txt) — Tambah library `google-generativeai`.

> [!NOTE] User Review
> Apakah Anda sudah memiliki **Gemini API Key** (dari Google AI Studio) untuk mengaktifkan fitur Pakar Agronomi AI? Jika ya, kita bisa siapkan variabel [.env](file:///c:/backend-orbitani/.env) baru untuk itu.
> 
> Silakan setujui rencana ini, dan saya akan langsung mengeksekusi penambahan semua endpoint yang kurang!
