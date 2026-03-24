# Syarat Wajib API Backend (API Contract Requirements)

Untuk memastikan kelancaran integrasi antara Frontend baru dan Backend, tim Backend **wajib** memenuhi kontrak API berikut. Jika ada perbedaan, Frontend akan mengalami error (400/404/500).

---

## 1. Laporan Analitik (Trend)
- **Endpoint**: `GET /api/lahan/analytics`
- **Tujuan**: Mengambil data rata-rata historis (berbentuk array) untuk ditampilkan sebagai 7 buah Line Chart.
- **Header**: `Authorization: Bearer <token>`
- **Response Format (200 OK)**:
```json
{
  "data": [
    {
      "date": "2026-03-01",
      "nitrogen": 45.2,
      "fosfor": 20.1,
      "kalium": 30.5,
      "ph": 6.5,
      "tci": 28.5,
      "ndti": 0.45,
      "rainfall": 120.5
    },
    ...
  ]
}
```

---

## 2. Histori Laporan (Tabel)
- **Endpoint**: `GET /api/history`
- **Tujuan**: Mengambil semua riwayat pengecekan (bisa dipaginasi jika perlu, tapi FE default membaca list array).
- **Header**: `Authorization: Bearer <token>`
- **Response Format (200 OK)**:
```json
{
  "data": [
    {
      "id": 1,
      "created_at": "2026-03-10T10:00:00Z",
      "longitude": 106.975437,
      "latitude": -6.701583,
      "nitrogen": 45,
      "fosfor": 20,
      "kalium": 30,
      "ph": 6.5,
      "tci": 28.5,
      "ndti": 0.45,
      "rainfall": 120,
      "label": "Padi"
    },
    ...
  ]
}
```

---

## 3. Manajemen Pengguna (RBAC)

Tim Frontend membaca peran (role) dari token JWT di `localStorage` (kolom `role`). Pastikan payload *login/register* menyertakan struktur role yang jelas (`superadmin`, `admin`, `user`).

**A. Ambil Semua User**
- **Endpoint**: `GET /api/users`
- **Role yang diizinkan**: `superadmin`, `admin`
- **Response**: Array of object user `{ "id": 1, "username": "...", "role": "..." }`

**B. Ubah Peran (Promote/Demote)**
- **Endpoint**: `PUT /api/users/{id}/role`
- **Role yang diizinkan**: `superadmin` saja
- **Payload**: `{ "role": "admin" }` atau `{ "role": "user" }`
- **Response**: HTTP 200 OK

**C. Hapus Pengguna**
- **Endpoint**: `DELETE /api/users/{id}`
- **Role yang diizinkan**: `superadmin`, `admin`
- **Response**: HTTP 200 OK atau 204 No Content

---

## 4. Retrain Model ML
Frontend **tidak lagi menembak API Retrain**. Jika BE membuat mekanisme Retrain otomatis berdasar cron/worker, tidak perlu mengirim pemberitahuan UI ke Frontend.

---

## 5. Eksplorasi Lahan Real-Time
- **Endpoint Utama**: `GET /api/lahan/{id}/data`
- **Alur**: API ini masih sama, namun Backend **DIMOHON** untuk mengeksekusi Google Earth Engine + RandomForest `.pkl` di *endpoint ini* saat dipanggil, lalu mengembalikan nilai aslinya.
- **Respons Waktu**: Frontend telah menset timeout menjadi 60 detik (60000ms) dan membuat UI Loading "*Memuat... 15-20 detik*". Silakan ambil waktu sebanyak yang dibutuhkan oleh *Engine* GEE di backend.
- **Response Format**: Sama seperti format payload `/api/lahan/{id}/data` yang lama, pastikan terdapat nilai `nitrogen`, `fosfor`, `kalium`, `ph`, `suhu`, `kelembapan`, `curah_hujan`, dan `jenis_tanaman`.
