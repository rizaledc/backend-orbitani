# Pembagian Tugas Tahap 2: Profile, Chat & Penghapusan Dark Mode

Permintaan penambahan fitur Interaksi Chat Antar-Pihak dan Profil Pengguna **MUTLAK MENGHARUSKAN** campur tangan tim Backend. Fitur ini tidak bisa hanya dikerjakan di Frontend karena membutuhkan media penyimpanan (*database*) dan jalur komunikasi jaringan (*WebSockets / REST*).

Berikut adalah rancangan tugas dan spesifikasi API (*API Contract*) agar tidak terjadi bentrok.

---

## 1. Menghapus Mode Malam (Dark Mode)
**Status:** 🎨 Murni Tugas Frontend

**Tugas Frontend:**
- Mematikan atau menghapus tombol *Toggle* Dark Mode di Navbar/Sidebar.
- Menonaktifkan/menghapus modul [themeStore.js](file:///c:/frontend-orbitani/src/store/themeStore.js).
- Membuang kelas Tailwind terkait aktivasi *Dark Mode* di file [index.css](file:///c:/frontend-orbitani/src/index.css) atau mengunci root HTML secara permanen di atribut `data-theme="light"`.

**Tugas Backend:** *Tidak ada.*

---

## 2 & 3. Profil Pengguna Dinamis Tersegmentasi (User vs Admin/Superadmin)
**Status:** 💻 Backend & 🎨 Frontend

**Kebutuhan Bisnis:** 
- Saat *User* biasa melihat profilnya, muncul data hasil register (Nama Lengkap, Email). 
- Saat *Admin / Superadmin* melihat profilnya, hanya muncul (Username, Role, dan Deskripsi). 

**Tugas Backend (API Contract):**
- Menambahkan kolom `description` (opsional/nullable text) pada tabel skema [User](file:///c:/frontend-orbitani/src/services/authService.js#3-7) di Database.
- Meng-update Endpoint `GET /api/auth/me` agar kembalian JSON-nya selengkap ini:
  ```json
  {
    "id": 1,
    "username": "petani01",
    "name": "Budi Santoso",    // (Diperlukan oleh Role User)
    "email": "budi@mail.com", // (Diperlukan oleh Role User)
    "role": "user",
    "description": null       
  }
  ```
- (Opsional) Endpoint `PUT /api/auth/profile` untuk memungkinkan Admin/Superadmin mengupdate teks `description` mereka.

**Tugas Frontend:**
- Membuat tombol "Profil" (bisa di pojok kanan Navbar) yang jika diklik akan membuka Modal/Pop-up Menu.
- Menerapkan *Conditional Rendering*:
  - `if (role === 'user') { return <CardUser name={data.name} email={data.email} /> }`
  - `else { return <CardAdmin username={data.username} role={data.role} desc={data.description} /> }`

---

## 4. Fitur Live Chat Antar Manusia (Human-to-Human)
**Status:** 💻 Backend (Sangat Berat) & 🎨 Frontend

Ini adalah komponen aplikasi skala menengah karena chat membutuhkan komunikasi interaktif dan pembatasan arus komunikasi yang sangat sempit.

**Aturan Arus (*Flow Rules*):**
- **User** ➔ Hanya boleh mengirim pesan ke **Admin**. (Dilarang mengirim ke sesama User atau Superadmin).
- **Admin** ➔ Boleh mengirim / membalas pesan ke **User** dan berdiskusi ke atas dengan **Superadmin**.

**Tugas Backend (API Contract & Infrastructure):**
1. **Schema Tabel `Message`**:
   - [id](file:///c:/frontend-orbitani/src/components/layout/Sidebar.jsx#12-129), `sender_id`, `receiver_id`, `message_text`, `timestamp`, `is_read`
2. **Logic Blocker (Validasi Hak Akses Arahan Chat)**:
   - Jika `sender.role == 'user'`, periksa apakah `receiver.role == 'admin'`. Jika bukan, tolak dengan `HTTP 403 (You can only message Admins)`.
3. **Endpoint yang Harus Disediakan**:
   - `GET /api/chat-live/contacts` ➔ Mengembalikan daftar akun yang sah diajak bicara oleh pengirim. 
     *(Misal: Kalau saya user, list kontak berisi nama-nama admin. Kalau saya admin, list kontak berisi semua user dan superadmin).*
   - `GET /api/chat-live/messages/{contact_id}` ➔ Menarik riwayat obrolan masa lalu (urut berdasar waktu) dengan lawan bicara spesifik tersebut.
   - `POST /api/chat-live/messages` ➔ Endpoint jika Frontend mengirim chat pasif.
   - **[OPSIONAL TAPI KUAT DIANJURKAN] `WebSocket /ws/chat/{user_id}`** ➔ Untuk pengalaman chat secara *real-time* dan *Live* tanpa harus Frontend melakukan *refresh* (polling) setiap detiknya yang bisa memberatkan server.

**Tugas Frontend:**
- Membuat Menu dan Halaman Baru: `HumanChat.jsx` (Diberi nama *"Pusat Bantuan"* atau semacamnya agar tidak ambigu dengan *Pakar AI*).
- Membuat UI **Daftar Kontak** sebelah kiri, dan **Room Obrolan** gelembung chat (Bubble Chat layaknya WhatsApp) di sebelah kanan.
- Menarik daftar kontak valid dari Backend.
- Jika Backend menggunakan *REST API biasa*: Frontend harus mengeksekusi *Polling* `setInterval` menembak `GET /messages` setiap 3 detik untuk mengecek apakah ada chat baru masuk.
- Jika Backend menggunakan *WebSockets*: Frontend bertugas menyambungkan protokol ws:// agar pesan yang diketik pengirim langsung berdenting seketika.
