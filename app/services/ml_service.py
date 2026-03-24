"""
ml_service.py
Lazy-loading ML prediction service for Orbitani.

ARSITEKTUR LAZY LOADING:
  Model .pkl TIDAK dimuat di global scope.
  Setiap kali prediksi dipanggil, model dimuat → prediksi → model dihapus dari RAM.
  Ini mencegah OOM (Out of Memory) pada Azure App Service dengan RAM terbatas.
"""
import gc
import logging
import os

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Path ke folder model (hanya string, BUKAN objek berat)
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DIR = os.path.join(BASE_DIR, "models_ml")

RF_MODEL_PATH = os.path.join(MODEL_DIR, "random_forest_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "minmax_scaler.pkl")
ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")


# ---------------------------------------------------------------------------
# Kalibrasi: mapping nilai satelit mentah → skala yang dikenali model
# ---------------------------------------------------------------------------
def calibrate_input(data: dict) -> dict:
    """
    Mengonversi nilai mentah satelit ke rentang fitur yang dikenali model.
    Fungsi ini RINGAN (hanya numpy interp), tidak memakan RAM signifikan.
    """
    return {
        "N": float(np.interp(data["N"], [-15.0, -5.0], [0.0, 120.0])),
        "P": float(np.interp(data["P"], [5.0, 6.0], [10.0, 80.0])),
        "K": float(np.interp(data["K"], [60.0, 150.0], [20.0, 150.0])),
        "temperature": float(np.interp(data["temperature"], [0.0, 100.0], [18.0, 28.0])),
        "humidity": float(np.interp(data["humidity"], [0.002, 0.005], [65.0, 85.0])),
        "ph": float(data["ph"]),
        "rainfall": float(np.clip(data["rainfall"], a_min=0, a_max=290.0)),
    }


# ---------------------------------------------------------------------------
# Prediksi dengan Lazy Load + Explicit Cleanup
# ---------------------------------------------------------------------------
def predict(input_data: dict) -> dict:
    """
    Memuat model .pkl ke RAM → prediksi → hapus dari RAM → gc.collect().
    Setiap pemanggilan bersifat self-contained dan tidak meninggalkan sisa di memori.
    """
    model = None
    scaler = None
    encoder = None

    try:
        # 1. LOAD — muat model ke memori lokal fungsi
        logger.info("Lazy-loading ML models from %s ...", MODEL_DIR)
        model = joblib.load(RF_MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        encoder = joblib.load(ENCODER_PATH)
        logger.info("ML models loaded successfully.")

        # 2. PREDICT — kalibrasi + prediksi
        calibrated_data = calibrate_input(input_data)
        features_df = pd.DataFrame([calibrated_data])
        scaled_features = scaler.transform(features_df)
        prediction = model.predict(scaled_features)
        recommendation = encoder.inverse_transform(prediction)[0]

        logger.info("Prediksi selesai → %s", recommendation)
        return {
            "calibrated_data": calibrated_data,
            "recommendation": recommendation,
        }

    finally:
        # 3. CLEANUP — hapus model dari RAM secara eksplisit
        del model, scaler, encoder
        gc.collect()
        logger.info("ML models unloaded, RAM freed (gc.collect done).")
