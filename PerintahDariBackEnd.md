# 🛑 PERINTAH DARI BACKEND UNTUK FRONTEND 🛑

Dokumen ini ditulis khusus oleh tim Backend sebagai acuan mutlak bagi tim Frontend. Harap dibaca dan dipahami sebelum mengintegrasikan API. Pelanggaran terhadap poin-poin di bawah ini akan menyebabkan 401, 404, 500, atau CORS Error.

---

## 1. ATURAN TRAILING SLASHES (SANGAT KRITIS)
FastAPI **sangat ketat** soal garis miring `/` di akhir URL. Jika salah, browser akan kena *307 Temporary Redirect* lalu mati diblokir CORS.

| Nama Rute | Endpoint yang BENAR | Keterangan |
|:---|:---|:---|
| **Root (Check)** | `GET /` | Harus slash `/` di akhir |
| **Login** | `POST /api/auth/login` | ❌ **HARAM** slash di akhir |
| **Register** | `POST /api/auth/register`| ❌ **HARAM** slash di akhir |
| **Cek Profil**| `GET /api/auth/me` | ❌ **HARAM** slash di akhir |
| **Chat Cepat**| `POST /api/chat/ask` | ❌ **HARAM** slash di akhir |
| **Chat Lahan**| `POST /api/chat/analyze-lahan` | ❌ **HARAM** slash di akhir |
| **Ambil Lahan** | `GET /api/lahan/` | ✅ **WAJIB** slash di akhir |
| **Buat Lahan** | `POST /api/lahan/` | ✅ **WAJIB** slash di akhir |
| **Data Lahan** | `GET /api/lahan/{lahan_id}/data` | ❌ **HARAM** slash di akhir |
| **Prediksi ML** | `POST /api/predict` | ❌ **HARAM** slash di akhir |

*Note: Frontend Axios sering merapikan path secara otomatis. Pastikan instance Axios kalian tidak menghapus/menambah slash yang salah.*

---

## 2. ATURAN LOGIN & PAYLOAD
Backend **TIDAK** menggunakan standar `OAuth2` Form-Data terenkripsi.
- Payload login harus berupa raw **JSON**.
- Header wajib: `Content-Type: application/json`.

**Contoh Payload Login yang Benar:**
```json
{
  "username": "contoh_user",
  "password": "password123"
}
```

---

## 3. ATURAN PENGIRIMAN TOKEN JARINGAN
Ini adalah jebakan paling sering terjadi.

1. **Rute Publik (Login & Register)**:
   - Kalian **DILARANG KERAS** mengirim header `Authorization`.
   - Mengirim `Authorization: Bearer null` atau `Bearer undefined` akan membuat middleware keamanan meledak dan mengembalikan HTTP 401 Unauthorized.
2. **Rute Protected (Selain Login/Register)**:
   - Kalian **Wajib** mengirim token valid.
   - Format: `Authorization: Bearer eyJhbGciOiJIUzI1...`

**Saran Implementasi Axios Interceptors:**
```javascript
// Kalau mau nembak login/register, HAPUS header auth!
if (config.url.includes('/auth/login') || config.url.includes('/auth/register')) {
    delete config.headers.Authorization;
}
```

---

## 4. PENAGKAPAN ERROR (EXCEPTION HANDLING)
Backend melemparkan format error secara seragam dengan kunci `"detail"`.

- **Error Umum (400, 401, 403, 404, 429)**: Response `"detail"` berupa **String**.
  *Contoh: `{"detail": "Username atau password salah"}`*

- **Error Pydantic Validasi (422 Unprocessable Entity)**: Response `"detail"` berupa **Array**.
  *Contoh kalau kalian lupa mengirim field password:*
  `{"detail": [{"loc": ["body", "password"], "msg": "Field required"}]}`

Tolong tulis penanganan error di FE yang mengecek apakah error `.detail` itu betipe Array atau String sebelum menampilkan *toast message*.

---

## 5. UPDATE INFRASTRUKTUR TEKNOLOGI
Untuk pengetahuan tim FE (jangan sampai *timeout* atau salah mengira backend mati):

1. **AI Chat Cepat (`/api/chat/ask`)**
   - Menggunakan model `gemini-flash-lite-latest`.
   - Latensi: ~2 detik. Respons sangat cepat.
2. **AI Analisis Lahan (`/api/chat/analyze-lahan`)**
   - Menggunakan model `gemini-2.5-flash`.
   - Prosesnya mengekstrak data satelit, lalu melakukan analisis mendalam.
   - **Latensi: Bisa 15-20 detik.**
   - *Tolong FE beri Loading State/Spinner panjang (mungkin dengan text animasi "Satelit sedang memindai...") untuk rute ini.*
3. **Model Machine Learning (`/api/predict` & Lahan)**
   - Semua `.pkl` sekarang **Lazy-Loaded**. Backend tidak memakan RAM, aman dari *crash*.

---

## 6. INFO KREDENSIAL DEVELOPMENT
Kalian bisa memakai akun ini untuk testing API (Auth Level: Superadmin):
- **Username:** `OrbitaniCorp`
- **Password:** `Corp32426!`

Selamat membangun Frontend!
*— Tertanda, Tim Backend*
