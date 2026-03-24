import gc
import logging
import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from app.db.database import supabase

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DIR = os.path.join(BASE_DIR, "models_ml")

RF_MODEL_PATH = os.path.join(MODEL_DIR, "random_forest_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "minmax_scaler.pkl")
ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")

def retrain_model():
    """
    Fungsi untuk melatih ulang model RandomForestClassifier menggunakan semua
    data dari tabel satellite_results di database.
    Hanya dijalankan lewat BackgroundTask untuk menghindari timeout HTTP.
    """
    try:
        logger.info("Memulai proses retraining ML di *Background*...")
        
        # 1. Tarik semua data dari database
        # Supabase default limit is 1000, we should fetch all if using pagination, but for now 1000 is enough for a demo
        result = supabase.table("satellite_results").select(
            "n_value, p_value, k_value, temperature, humidity, ph, rainfall, recommendation"
        ).execute()

        data = result.data
        if not data or len(data) < 10:
            logger.warning("Data satelit tidak cukup (<10 baris) untuk retraining.")
            return

        df = pd.DataFrame(data)

        # Hapus baris tanpa rekomendasi final
        df = df[df["recommendation"].notna() & (df["recommendation"] != "Pending Analysis")]
        
        if len(df) < 10:
            logger.warning("Data training bersih tidak cukup untuk retraining.")
            return

        # 2. Persiapkan Feature (X) dan Label (y)
        features = ["n_value", "p_value", "k_value", "temperature", "humidity", "ph", "rainfall"]
        X = df[features]
        y = df["recommendation"]

        # 3. Fit Encoder & Scaler
        encoder = LabelEncoder()
        y_encoded = encoder.fit_transform(y)

        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)

        # 4. Train Model
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X_scaled, y_encoded)

        # 5. Simpan (Overwrite) ke path .pkl
        os.makedirs(MODEL_DIR, exist_ok=True)
        joblib.dump(model, RF_MODEL_PATH)
        joblib.dump(scaler, SCALER_PATH)
        joblib.dump(encoder, ENCODER_PATH)

        logger.info(f"Retraining sukses! Model diperbarui dengan {len(df)} baris data.")

    except Exception as e:
        logger.error(f"Gagal retraining model: {e}")
    finally:
        # Bersihkan RAM secara agresif
        gc.collect()

def check_and_trigger_retrain():
    """
    Mengecek jumlah baris. Jika kelipatan 50, trigger retrain secara asinkron.
    """
    try:
        res = supabase.table("satellite_results").select("id", count="exact").limit(1).execute()
        count = res.count
        if count and count > 0 and count % 50 == 0:
            logger.info(f"Jumlah baris mencapai {count}. Memicu auto-retrain...")
            retrain_model()
    except Exception as e:
        logger.error(f"Gagal check auto-retrain: {e}")
