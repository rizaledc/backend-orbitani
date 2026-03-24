# Pembagian Tugas: Backend & Frontend

Berikut adalah rancangan pembagian tugas (*API Contract & UI requirements*) untuk pengembangan tahap selanjutnya di platform Orbitani.

---

## 1. Laporan Analitik (Line Chart Trend & Sampling 200 Titik)
**Filosofi Bisnis:** Mengambil *Polygon* tertentu, me-random ~200 titik di dalamnya untuk mencari nilai tengah NPK & Iklim, lalu menyajikannya dalam wujud grafik tren dari waktu ke waktu (pembacaan satelit berkala).

**💻 Tugas Backend:**
- Mendesain skrip Google Earth Engine (GEE) menggunakan spesifikasi Poligon tersebut (`ee.Geometry.Polygon`).
- Membuat logika komputasi untuk men-generate 200 titik acak di dalam area, kemudian mengekstrak nilai rata-rata untuk indikator: N, P, K, pH, NDTI, TCI, dan Curah Hujan.
- Menjalankan fungsi ini secara periodik atau *on-demand*, dan menyimpan agregat datanya (sebagai nilai tren *time-series*) ke dalam *database*.
- Mengekspos REST API endpoint (contoh: `GET /api/lahan/analytics_trend`) yang mengembalikan array nilai tiap tanggal panen data satelit.

**🎨 Tugas Frontend:**
- Membuat halaman antarmuka "Laporan Analitik".
- Memanggil `GET /api/lahan/analytics_trend`.
- Menerapkan library visualisasi grafik (seperti `Recharts`, `Chart.js`, atau `ApexCharts`) untuk membangun 7 *Line Chart* terpisah (atau dengan sistem tab) agar fluktuasi elemen tanah mudah dibaca secara visual.

---

## 2. Tabel "Report" Historis & Ekspor ke Excel (.xlsx)
**Filosofi Bisnis:** Setiap prediksi dan pengecekan lahan yang pernah dilakukan tidak hanya lenyap, melainkan masuk ke *Ledger* Historis yang bisa ditinjau dan diekspor ke format pelaporan (Excel).

**💻 Tugas Backend:**
- Memastikan bahwa setiap *scanning lahan/predict* tersimpan dengan aman di tabel database rekaman pengecekan secara permanen.
- Membuat endpoint spesifik (contoh: `GET /api/history`) yang membalikkan *Array of Objects* berisi variabel: `Longitude`, `Latitude`, `N`, `P`, `K`, `pH`, `TCI (Temperatur)`, `NDTI (Kelembapan)`, `Rainfall`, dan `Label (Tanaman)`.

**🎨 Tugas Frontend:**
- Menghapus tombol primitif "Export CSV" yang lama.
- Membuat komponen tabel *DataGrid* yang lebih modern (*Searchable & Paginated*) di *dashboard* untuk menampilkan 10 kolom data tersebut.
- Menyediakan tombol "Export to .xlsx". Frontend bisa menggunakan utilitas NPM (seperti `xlsx` atau `file-saver`) untuk mengubah *Array of JSON* langsung menjelas file Excel standar untuk diunduh tanpa harus membebani komputasi server backend.

---

## 3. Manajemen Pengguna (Superadmin vs Admin)
**Filosofi Bisnis:** Pengaturan lapis otorisasi *Role-Based Access Control* (RBAC). Mode kekuasaan penuh (Promosi, Penurunan Pangkat, Hapus) untuk Superadmin, dan kekuasaan terbatas (hanya Hapus User) untuk Admin biasa.

**💻 Tugas Backend:**
- Menambahkan skema kolom peran/pangkat (`Role Enum`) ke tabel User: `SUPERADMIN`, `ADMIN`, `USER`.
- Membuat *Middleware / Dependency Injection* pengamanan rute.
- Mendevelop 3 API untuk Manajemen Registri:
  1. `GET /api/users` (Melihat daftar semua anggota)
  2. `PUT /api/users/{id}/role` (Menaikkan/menurunkan pangkat pengguna). **Di-lock HANYA untuk SUPERADMIN**. Mengakses ini nganggi token Admin biasa akan menghasilkan HTTP 403 Forbidden.
  3. `DELETE /api/users/{id}` (Menghapus anggota). **Di-lock untuk SUPERADMIN & ADMIN**.

**🎨 Tugas Frontend:**
- Membangun menu tambahan "Manajemen Pengguna" di Sidebar (hanya _render_ jika `role` di *State Manager* adalah Admin/Superadmin).
- Menampilkan daftar semua profil *user*.
- Melakukan evaluasi komponen secara dinamis:
  - Jika yang _login_ adalah `ADMIN`, maka tombol aksi "Jadikan Admin" / "Jadikan User" **DISALUT (disabled) / DIHILANGKAN**. Hanya ada tombol tong sampah (Hapus).
  - Jika yang _login_ adalah `SUPERADMIN`, seluruh panel kendali interaktif bisa digunakan.

---

## 4. Pembaruan Model ML Berjalan *Di Belakang Layar* (Background Retrain)
**Filosofi Bisnis:** Proses *Machine Learning Retraining* .pkl tidak perlu ditunggui oleh manusia karena memakan waktu dan berpotensi gagal render di antarmuka jika *timeout*. Semua harus beroperasi secara *Daemon/Asynchronous*.

**💻 Tugas Backend (Sangat Penting):**
- Menanamkan integrasi asinkron (menggunakan modul *Celery*, *RabbitMQ* atau sekadar *APScheduler / FastAPI BackgroundTasks*) yang aktif mandiri.
- Mengatur kondisi pemicu otomatis (*Auto-Trigger*), semisalnya jika "ada 50 parameter data lahan baru masuk ke database", maka *script retrain* .pkl terpicu sendiri tanpa perlu klik dari admin.

**🎨 Tugas Frontend:**
- Menghancurkan sepenuhnya halaman antarmuka UI "Retrain Model ML".
- Menarik menu routing di navigasi, sehingga pemicu retrain secara manual musnah dari genggaman pengguna sepenuhnya di sisi frontend.

---

## 5. Eksplorasi Lahan Real-Time
**Filosofi Bisnis:** Nilai koordinat yang diklik secara *live* memancarkan nilai asli dari pindaian satelit di orbit, yang lalu seketika diklasifikasi bibitnya oleh AI (model konvensional .pkl), bukan cuma *dummy data*.

**💻 Tugas Backend:**
- Menyatukan *pipeline* sinkronisasi: Ketika endpoint sinkronisasi wilayah (atau prediksi) ditembak, server mengeksekusi GEE untuk mendapatkan *Value NPK & Iklim* saat ini juga.
- Menyodorkan array parameter tersebut ke instansi Model Regresi `RandomForest.pkl` (yang sudah *Lazy-Loaded*) untuk menemukan label `jenis_tanaman` yang paling cocok.
- Mengembalikan hasil ekstraksi Satelit beserta hasil Analisis Prediksi secara serentak (*JSON Bundle*) ke penunggu API frontend.

**🎨 Tugas Frontend:**
- Fokus ke UX pemuatan: Mempertahankan fungsionalitas memanggil `/api/lahan/{id}/data` ketika marker diklik atau disinkronisasi.
- Membuat indikator Spinner Loading khusus berdurasi panjang (estimasi 15-20 detik selama Google Earth Engine memutar dunia) supaya user paham bahwa kalkulasi ML memang sedang berlangsung, alih-alih mengira aplikasi nge-*lag*/beku.
- Begitu data asli dari backend berlabuh, mengisi langsung *Value NPK, pH, Cuaca, dan Tanaman Identifikasi* ke *Bottom Sheet Panel*.

---

## 6. Pakar AI
**Status:** **SELESAI ✅**
Baik Frontend dan Backend (Gemini-Flash-Lite & Gemini-2.5-Flash integration) tidak butuh intervensi lagi terkait *Chatbot* argonomi. Sudah beroperasi dengan sempurna.
