# Orbitani Backend — Frontend Integration Guide
**Versi:** 1.0 | **Dibuat untuk:** Tim Frontend (Khulika & Mikhael)

---

## Quick Start

```
Base URL : http://localhost:8000 (dev) | https://api.orbitani.xyz (prod)
Auth     : Semua endpoint (kecuali /register & /login) butuh header:
           Authorization: Bearer <ACCESS_TOKEN>
```

---

## Autentikasi (RBAC)

### Register
```
POST /api/auth/register
Body: { "username": "string (3-50 char, alphanumeric)", "password": "string (min 8 char)" }
Resp: { "access_token": "...", "token_type": "bearer", "role": "user" }
```

### Login
```
POST /api/auth/login
Body: { "username": "...", "password": "..." }
Resp: { "access_token": "...", "token_type": "bearer", "role": "user|admin|superadmin" }
```

### Profil User (untuk render sidebar & menu RBAC)
```
GET /api/auth/me
Resp: { "id": 1, "username": "budi", "role": "user" }
```

### Statistik User (untuk Dashboard)
```
GET /api/users/stats
Resp: { "username": "budi", "role": "user", "total_lahan": 2, "total_satellite_extractions": 11 }
```

---

## WebGIS — Manajemen Lahan

### Buat Lahan Baru (dari polygon yang digambar di Leaflet)
```
POST /api/lahan/
Body:
{
  "nama": "Lahan Blok A",
  "deskripsi": "Lahan percobaan Kenaf",
  "koordinat": {
    "type": "Polygon",
    "coordinates": [[[106.975, -6.701], [106.979, -6.701], 
                      [106.979, -6.705], [106.975, -6.701]]]
  }
}
Resp: { "status": "success", "data": { "id": 1, ... } }
```

### Daftar Lahan User
```
GET /api/lahan/
Resp: { "status": "success", "data": [ { "id": 1, "nama": "...", "koordinat": {...} }, ... ] }
```

### Map Data untuk Leaflet Markers
```
GET /api/lahan/{lahan_id}/data
Resp: {
  "status": "success",
  "lahan": { "id": 1, "nama": "...", "koordinat": {...} },
  "satellite_data": [
    { "longitude": 106.977, "latitude": -6.703, "n_value": -6.3,
      "p_value": 7.3, "k_value": 110.5, "ph": 8.16,
      "temperature": 27.2, "humidity": -0.015, "rainfall": 4190.0,
      "recommendation": "Pending Analysis", "extracted_at": "2026-03-23T..." },
    ...
  ]
}
```

---

## Analisis Satelit GEE

### Sinkronisasi Data Satelit (trigger GEE extraction)
```
POST /api/analyze-location
# ⚠️  Proses ini memakan waktu 2-5 menit karena GEE API.
# Tampilkan loading spinner di UI!
Body: { "lahan_id": 1, "latitude": -6.703, "longitude": 106.977 }
```

---

## Pakar Agronomi AI

### Chat Cepat — Q&A (gemini-3.1-flash, ~2-3 detik)
```
POST /api/chat/ask
Body: { "message": "Berapa dosis Urea untuk pH 5.8?" }
Resp: { "status": "success", "model": "gemini-3.1-flash", "answer": "## Orbitani Smart Analysis\n..." }
```

### Analisis Mendalam — Berdasarkan data GEE terbaru (gemini-2.5-flash, ~10-20 detik)
```
POST /api/chat/analyze-lahan
# ⚠️  Tampilkan loading skeleton, bukan spinner kecil!
Body: { "lahan_id": 1 }
Resp: {
  "status": "success",
  "model": "gemini-2.5-flash",
  "satellite_data": { "n_value": -6.3, "ph": 8.16, ... },
  "ai_analysis": "## Orbitani Smart Analysis\n### Poin 1: ...\n..."
}
```

---

## Rate Limiting (HTTP 429)

Endpoint AI dibatasi untuk role `user`: **5 permintaan per menit**.
Admin & Superadmin tidak dibatasi.

### Cara Menangani 429 di Frontend

```javascript
// Contoh fetch dengan penanganan 429
async function askAI(message, token) {
  const res = await fetch('/api/chat/ask', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${token}`
    },
    body: JSON.stringify({ message })
  });

  if (res.status === 429) {
    const err = await res.json();
    const retryAfter = err.detail.retry_after_seconds;
    // Tampilkan toast/notifikasi ke user
    showToast(`⏳ Batas permintaan tercapai. Coba lagi dalam ${retryAfter} detik.`, 'warning');
    return null;
  }

  if (!res.ok) {
    showToast('Terjadi kesalahan. Silakan coba lagi.', 'error');
    return null;
  }

  return await res.json();
}
```

> **Tip UX**: Tampilkan countdown timer (`Coba lagi dalam Xs`) menggunakan `Retry-After` header yang disertakan dalam respons 429. Gunakan `res.headers.get('Retry-After')`.

---

## Manajemen User (Superadmin Only)

| Method | Endpoint | Fungsi |
|--------|----------|--------|
| `GET` | `/api/users/` | Daftar semua user |
| `PUT` | `/api/users/{id}/role?new_role=admin` | Ubah role user |
| `DELETE` | `/api/users/{id}` | Hapus user |

---

## Kode Status HTTP Umum

| Status | Arti | Tindakan Frontend |
|--------|------|-------------------|
| `200` | Sukses | Tampilkan data |
| `201` | Data dibuat | Tampilkan toast sukses |
| `400` | Input tidak valid / duplikat | Tampilkan pesan error dari `detail` |
| `401` | Token tidak valid / expired | Redirect ke halaman login |
| `403` | Akses ditolak (role tidak cukup) | Sembunyikan menu, tampilkan pesan |
| `404` | Data tidak ditemukan | Tampilkan state kosong |
| `429` | Rate limit tercapai | Tampilkan countdown, coba lagi |
| `500` | Error server internal | Tampilkan toast "Sistem sedang sibuk" |
