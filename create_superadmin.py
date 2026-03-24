"""
create_superadmin.py
Script untuk mendaftarkan akun Superadmin utama OrbitaniCorp ke database Supabase.

Cara menjalankan:
  python create_superadmin.py
"""
import sys
import os

# Pastikan root folder ada di sys.path agar import app.* bisa berjalan
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

from app.db.database import supabase as db
from app.core.security import hash_password

# ---- Konfigurasi Superadmin ----
USERNAME = "OrbitaniCorp"
PASSWORD = "Corp32426!"
ROLE = "superadmin"


def main():
    print("=" * 50)
    print("  Orbitani — Superadmin Creator")
    print("=" * 50)

    # 1. Cek apakah user sudah ada
    existing = db.table("users").select("id, username, role").eq("username", USERNAME).execute()

    if existing.data:
        user = existing.data[0]
        print(f"\n⚠️  User '{USERNAME}' sudah ada di database!")
        print(f"   ID   : {user['id']}")
        print(f"   Role : {user['role']}")

        if user["role"] != "superadmin":
            # Upgrade role ke superadmin jika belum
            db.table("users").update({"role": "superadmin"}).eq("id", user["id"]).execute()
            print(f"\n✅ Role berhasil di-upgrade dari '{user['role']}' → 'superadmin'")
        else:
            print("\n✅ Sudah menjadi superadmin. Tidak perlu perubahan.")
        return

    # 2. Buat user baru
    hashed = hash_password(PASSWORD)
    result = db.table("users").insert({
        "username": USERNAME,
        "password_hash": hashed,
        "role": ROLE,
    }).execute()

    if result.data:
        user = result.data[0]
        print(f"\n🎉 Superadmin berhasil dibuat!")
        print(f"   Username : {user['username']}")
        print(f"   Role     : {user['role']}")
        print(f"   ID       : {user['id']}")
        print(f"\n   Login di frontend dengan:")
        print(f"   Username : {USERNAME}")
        print(f"   Password : {PASSWORD}")
    else:
        print("\n❌ Gagal membuat superadmin. Cek koneksi Supabase.")


if __name__ == "__main__":
    main()
