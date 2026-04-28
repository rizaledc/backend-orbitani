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
import warnings

# Suppress sklearn InconsistentVersionWarning due to version mismatch
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

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

    Range input disesuaikan dengan output aktual GEE:
      N   : NDVI*2.5+0.5 → 0.5–3.0   → mapping ke skala training 0–120
      P   : B3/B8*30+10  → 10–40      → mapping ke skala training 10–80
      K   : B11/B12*150+50 → 50–200   → mapping ke skala training 20–150
      humidity : NDTI    → 0.1–0.5     → mapping ke persentase kelembapan 70–99%
      temperature : sudah Celsius dari GEE, langsung pakai
      rainfall : CHIRPS 30 hari → sudah bulanan langsung (tidak perlu /12)
    """
    # --- 1. KALIBRASI SATUAN (UNIT CONVERSION) ---
    raw_n = data.get("N", data.get("n", 0))
    raw_p = data.get("P", data.get("p", 0))
    raw_k = data.get("K", data.get("k", 0))
    raw_temp = data.get("temperature", 0)
    raw_hum = data.get("humidity", 0)
    raw_ph = data.get("ph", 0)
    raw_rain = data.get("rainfall", data.get("rain", 0))

    calibrated_n = raw_n * 20                         # GEE (1-3) -> menjadi skala 20-60
    calibrated_p = raw_p                              # P sudah dalam batas aman
    calibrated_k = raw_k / 6                          # GEE (~240-375) -> ditekan ke skala ~40-62 (avg training: 48)
    calibrated_temp = raw_temp - 3.0                   # LST satelit → suhu udara (offset empiris)
    # ERA5 RH sudah dalam % (nilai > 1.0); NDTI dalam skala 0.0-0.5 (nilai <= 1.0)
    if raw_hum > 1.0:
        calibrated_hum = min(max(raw_hum, 0.0), 99.0)  # ERA5 RH -> langsung pakai (sudah %)
    else:
        calibrated_hum = min(70 + (raw_hum * 60), 99.0)  # NDTI fallback -> konversi ke %
    calibrated_ph = raw_ph                            # Aman
    calibrated_rain = raw_rain                     # CHIRPS 30 hari → sudah dalam satuan bulanan

    return {
        "N":           float(calibrated_n),
        "P":           float(calibrated_p),
        "K":           float(calibrated_k),
        "temperature": float(calibrated_temp),
        "humidity":    float(calibrated_hum),
        "ph":          float(calibrated_ph),
        "rainfall":    float(calibrated_rain),
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
        
        # Pastikan urutannya persis: ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        features_array = [[
            calibrated_data["N"],
            calibrated_data["P"],
            calibrated_data["K"],
            calibrated_data["temperature"],
            calibrated_data["humidity"],
            calibrated_data["ph"],
            calibrated_data["rainfall"]
        ]]

        # Terapkan log1p pada rainfall sesuai preprocessing model baru
        features_array[0][6] = np.log1p(features_array[0][6])


        scaled_features = scaler.transform(features_array)
        prediction = model.predict(scaled_features)
        recommendation = encoder.inverse_transform(prediction)[0]

        logger.info("Prediksi selesai → %s", recommendation)
        return {
            "calibrated_data":   calibrated_data,
            "ai_recommendation": recommendation,  # Sesuai kolom DB: ai_recommendation
        }

    finally:
        # 3. CLEANUP — hapus model dari RAM secara eksplisit
        del model, scaler, encoder
        gc.collect()
        logger.info("ML models unloaded, RAM freed (gc.collect done).")
