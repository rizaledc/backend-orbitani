"""
retrain_service.py
MLOps: Retraining Random Forest with dataset CSV + ml_feedback Ground Truth.

Alur retrain_model_full():
  1. Load Crop_recommendation.csv (dataset awal)
  2. Pull data ml_feedback (ground truth) dari Supabase
  3. Rename kolom feedback agar selaras dengan CSV (case-sensitive)
  4. pandas.concat() kedua sumber
  5. Fit LabelEncoder, MinMaxScaler, RandomForestClassifier
  6. Simpan model baru dengan timestamp (archive)
  7. Timpa file utama .pkl (Hot-Swap tanpa restart Uvicorn)
"""
import gc
import logging
import os
from datetime import datetime

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

from app.db.database import supabase

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DIR = os.path.join(BASE_DIR, "models_ml")
DATASET_CSV = os.path.join(MODEL_DIR, "Crop_recommendation.csv")

RF_MODEL_PATH  = os.path.join(MODEL_DIR, "random_forest_model.pkl")
SCALER_PATH    = os.path.join(MODEL_DIR, "minmax_scaler.pkl")
ENCODER_PATH   = os.path.join(MODEL_DIR, "label_encoder.pkl")

# Mapping kolom satellite_results/ml_feedback → nama kolom CSV (case-sensitive)
FEATURE_RENAME_MAP = {
    # ml_feedback columns
    "n": "N", "p": "P", "k": "K",
    # n_value / p_value / k_value dari satellite_results jika diperlukan
    "n_value": "N", "p_value": "P", "k_value": "K",
    # env columns — sudah lowercase, konfirm saja
    "temperature": "temperature",
    "humidity":    "humidity",
    "ph":          "ph",
    "rainfall":    "rainfall",
}

FEATURE_COLS = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
TARGET_COL   = "label"


def _load_csv_dataset() -> pd.DataFrame:
    """Load dataset CSV awal dan pastikan kolom sesuai standar."""
    if not os.path.exists(DATASET_CSV):
        logger.warning("Dataset CSV tidak ditemukan di: %s", DATASET_CSV)
        return pd.DataFrame()

    df = pd.read_csv(DATASET_CSV)
    # CSV sudah punya kolom sesuai standar (N, P, K, temperature, humidity, ph, rainfall, label)
    df = df[FEATURE_COLS + [TARGET_COL]].dropna()
    logger.info("Dataset CSV dimuat: %d baris", len(df))
    return df


def _load_feedback_dataset() -> pd.DataFrame:
    """Pull data ground truth dari tabel ml_feedback dan normalise kolom."""
    try:
        result = supabase.table("ml_feedback").select(
            "n, p, k, temperature, humidity, ph, rainfall, actual_crop"
        ).not_.is_("actual_crop", "null").execute()
    except Exception as e:
        logger.warning("Gagal tarik ml_feedback: %s", e)
        return pd.DataFrame()

    if not result.data:
        logger.info("ml_feedback kosong — tidak ada ground truth baru.")
        return pd.DataFrame()

    df = pd.DataFrame(result.data)
    # Rename kolom agar case-sensitive sesuai CSV
    df = df.rename(columns={
        "n": "N", "p": "P", "k": "K",
        "actual_crop": TARGET_COL,
    })
    df = df[FEATURE_COLS + [TARGET_COL]].dropna()
    logger.info("ml_feedback dimuat: %d baris valid", len(df))
    return df


def retrain_model_full():
    """
    Retrain penuh: CSV + ml_feedback → fit → simpan timestamp → hot-swap .pkl.
    Aman dijalankan di BackgroundTask — tidak memblokir event loop.
    """
    try:
        logger.info("=== [RETRAIN] Memulai proses retraining full ===")

        # 1. Load sumber data
        df_csv      = _load_csv_dataset()
        df_feedback = _load_feedback_dataset()

        # 2. Gabungkan
        frames = [f for f in [df_csv, df_feedback] if not f.empty]
        if not frames:
            logger.error("[RETRAIN] Tidak ada data sama sekali. Abort.")
            return

        df = pd.concat(frames, ignore_index=True)
        df = df.dropna(subset=FEATURE_COLS + [TARGET_COL])
        logger.info("[RETRAIN] Total baris gabungan: %d", len(df))

        if len(df) < 20:
            logger.warning("[RETRAIN] Data terlalu sedikit (<20 baris). Abort.")
            return

        # 3. Feature & Label
        X = df[FEATURE_COLS]
        y = df[TARGET_COL]

        # 4. Fit Encoder & Scaler
        encoder = LabelEncoder()
        y_encoded = encoder.fit_transform(y)

        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)

        # 5. Train Random Forest
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_scaled, y_encoded)
        logger.info("[RETRAIN] Model selesai dilatih dengan %d kelas.", len(encoder.classes_))

        # 6. Simpan dengan timestamp (arsip)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_model   = os.path.join(MODEL_DIR, f"rf_model_{ts}.joblib")
        archive_scaler  = os.path.join(MODEL_DIR, f"scaler_{ts}.joblib")
        archive_encoder = os.path.join(MODEL_DIR, f"encoder_{ts}.joblib")

        os.makedirs(MODEL_DIR, exist_ok=True)
        joblib.dump(model,   archive_model)
        joblib.dump(scaler,  archive_scaler)
        joblib.dump(encoder, archive_encoder)
        logger.info("[RETRAIN] Arsip tersimpan: %s", archive_model)

        # 7. HOT-SWAP: Timpa file utama .pkl
        # ml_service.predict() menggunakan lazy-load → setiap request berikutnya
        # otomatis membaca file baru tanpa restart Uvicorn
        joblib.dump(model,   RF_MODEL_PATH)
        joblib.dump(scaler,  SCALER_PATH)
        joblib.dump(encoder, ENCODER_PATH)
        logger.info("[RETRAIN] Hot-Swap selesai! Model aktif diperbarui.")
        logger.info("[RETRAIN] Kelas yang dikenal: %s", list(encoder.classes_))

    except Exception as e:
        logger.error("[RETRAIN] Gagal: %s", e, exc_info=True)
    finally:
        gc.collect()
        logger.info("[RETRAIN] RAM dibersihkan.")


# ---------------------------------------------------------------------------
# Legacy — auto-retrain berdasarkan jumlah baris satellite_results
# ---------------------------------------------------------------------------
def retrain_model():
    """
    Legacy retrain (hanya dari satellite_results). Dipertahankan untuk
    backward compat dengan check_and_trigger_retrain().
    """
    retrain_model_full()


def check_and_trigger_retrain():
    """Trigger auto-retrain setiap kelipatan 50 baris satellite_results."""
    try:
        res = supabase.table("satellite_results").select("id", count="exact").limit(1).execute()
        count = res.count
        if count and count > 0 and count % 50 == 0:
            logger.info("[AUTO-RETRAIN] Jumlah baris mencapai %d. Memicu retrain...", count)
            retrain_model_full()
    except Exception as e:
        logger.error("[AUTO-RETRAIN] Gagal check: %s", e)
