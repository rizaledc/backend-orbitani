# Panduan Integrasi Frontend Tahap 2: Profil & Live Chat API

Dokumen ini berisi pembaruan kontrak API dari tim Backend untuk fitur **Profil Pengguna Dinamis**, **Live Chat**, dan **Analitik GEE 10-Titik**. Mohon patuhi kontrak ini agar tidak terjadi *error* atau *CORS crash* saat integrasi.

---

## 1. Profil Pengguna Dinamis

Backend telah memodifikasi struktur Database `users` dengan tambahan kolom `name`, `email`, dan `description`.

### A. Register (POST `/api/auth/register`)
Sekarang **wajib** atau **sunnah** mengirimkan `name` dan `email` untuk role User biasa.
**Request Body (JSON):**
```json
{
  "username": "petani01",
  "password": "PasswordKuat123!",
  "name": "Budi Santoso",
  "email": "budi@mail.com"
}
```

### B. Ambil Profil (GET `/api/auth/me`)
Endpoint ini akan mengembalikan data profil utuh. Lakukan *Conditional Rendering* di Frontend berdasarkan nilai `role`.
**Response (200 OK):**
```json
{
  "id": 1,
  "username": "petani01",
  "role": "user",
  "name": "Budi Santoso",
  "email": "budi@mail.com",
  "description": null
}
```
*Catatan:* Jika yang login adalah Admin, maka `name` dan `email` mungkin `null`, namun `description` akan terisi jabatannya (jika sudah diset).

---

## 2. Live Chat Antar-Pengguna (REST API)

Gunakan *Base URL*: `https://backend-orbitani-...azurewebsites.net/api/chat-live`

### A. Tampilkan Daftar Kontak Sah (GET `/contacts`)
Gunakan endpoint ini untuk me-render daftar kontak (Sidebar Kiri di menu Pusat Bantuan).
- Jika penelepon adalah **User**: Hanya akan menerima list Admin.
- Jika penelepon adalah **Admin/Superadmin**: Akan menerima list semua User & Admin.
**Response:**
```json
[
  {
    "id": 2,
    "username": "admin_cs",
    "role": "admin",
    "name": "Staff Customer Service",
    "description": "Siap membantu Anda"
  }
]
```

### B. Tampilkan Riwayat Pesan (GET `/messages/{contact_id}`)
Gunakan endpoint ini ketika Frontend mengklik salah satu nama kontak di sidebar. Ini akan memuat seluruh riwayat *bubble chat* terdahulu.
**Response:**
```json
[
  {
    "id": 15,
    "sender_id": 1,
    "receiver_id": 2,
    "message_text": "Halo pak Admin, tanaman saya layu.",
    "is_read": false,
    "timestamp": "2026-03-25T14:30:00Z"
  }
]
```

### C. Kirim Pesan Teks (POST `/messages`)
Gunakan ini untuk melempar pesan teks. Jika mengirim ke role yang salah (User mengirim ke User lain), Backend akan memblokir dan mengirim **HTTP 403 Forbidden**.
**Request Body (JSON):**
```json
{
  "receiver_id": 2,
  "message_text": "Halo pak Admin, tanaman saya layu."
}
```
*Tips Frontend:* Anda bisa memanggil endpoint ini sebagai alat utama untuk "*Send Message*", namun untuk **menerima pesan baru tanpa refresh**, gunakan WebSocket di bawah ini.

---

## 3. Live Chat Real-Time (WebSockets)

Untuk membuat pengalaman chat layaknya WhatsApp (pesan masuk seketika), Frontend **WAJIB** menyambungkan socket di latar belakang.

- **URL WS:** `wss://backend-orbitani-...azurewebsites.net/api/chat-live/ws/{my_user_id}`
*(Perhatikan penggunaan `wss://` bukan `https://`, dan `{my_user_id}` adalah ID milik Anda sendiri yang didapat dari `/api/auth/me`).*

**Cara Kerja di Frontend (Contoh React/JS):**
```javascript
const myId = currentUser.id;
const ws = new WebSocket(`wss://{YOUR_AZURE_DOMAIN}/api/chat-live/ws/${myId}`);

ws.onmessage = (event) => {
  const incoming = JSON.parse(event.data);
  if (incoming.event === "new_message") {
    // Tambahkan incoming.data ke state array chat bubble!
    console.log("Pesan baru dari:", incoming.data.sender_id);
    console.log("Teks pesan:", incoming.data.message_text);
  }
};
```
*Saran:* Cukup gunakan `POST /messages` untuk mengirim pesan, lalu biarkan *event listener* `ws.onmessage` yang bertugas menangkap pantulan pesan masuk dari orang lain.

---

## 4. Ekstraksi Google Earth Engine (10-Titik Rata-rata)

Penting untuk tim Frontend ketahui bahwa endpoint real-time lahan:
`GET /api/lahan/{lahan_id}/data?lat=...&lon=...`

Sekarang **TIDAK LAGI** hanya mengekstrak 1 titik berdasar lat/lon yang diklik.
Backend kini secara cerdas mengabaikan titik klik tersebut dan **secara otomatis menyebarkan 10 Titik Grid Acak di dalam batas Poligon Lahan Hibisc** `[106.975437, -6.701583] ...`.
Data dari ke-10 titik tersebut kemudian ditarik secara paralel dari *Landsat 8 GEE*, lalu dirata-rata ke dalam 1 angka konklusi untuk N, P, K, pH.

**Dampak bagi Frontend:**
- Tidak perlu mengubah payload request apa-apa. Tetap panggil parameter `?lat=x&lon=y` seperti biasa.
- Nilai yang kembali 100% merupakan Data Satelit Asli *(Bukan Dummy)*, dan representasinya sudah mencakup luas keseluruhan petak Lahan (karena nilai dari 10 titik telah dirata-ratakan).
- Hasil agregasi 10 titik ini yang nantinya akan otomatis digambar oleh grafik di menu **Laporan Analitik**.
