import logging
import joblib
import pandas as pd
import numpy as np
import os

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Path ke folder model
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DIR = os.path.join(BASE_DIR, "models_ml")


class MLService:
    """Service untuk memuat model ML dan menjalankan prediksi rekomendasi tanaman."""

    def __init__(self) -> None:
        # Fail-fast: jika model tidak bisa di-load, aplikasi tidak boleh start
        self.model = joblib.load(os.path.join(MODEL_DIR, "random_forest_model.pkl"))
        self.scaler = joblib.load(os.path.join(MODEL_DIR, "minmax_scaler.pkl"))
        self.encoder = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))
        logger.info("Semua model ML berhasil dimuat dari %s", MODEL_DIR)

    # ------------------------------------------------------------------
    # Kalibrasi: mapping nilai satelit mentah → skala yang dikenali model
    # ------------------------------------------------------------------
    def calibrate_input(self, data: dict) -> dict:
        """
        Mengonversi nilai mentah satelit ke rentang fitur yang dikenali model
        menggunakan numpy.interp() untuk interpolasi linier.

        Mapping:
          N          : satelit [-15, -5]    → model [0, 120]
          P          : satelit [5, 6]       → model [10, 80]
          K          : satelit [60, 150]    → model [20, 150]
          temperature: satelit (TCI) [0, 100]   → model [18, 28] °C
          humidity   : satelit (NDTI) [0.002, 0.005] → model [65, 85] %
          ph         : pass-through (sudah skala normal)
          rainfall   : clip max 290
        """
        calibrated = {
            "N": float(np.interp(data["N"], [-15.0, -5.0], [0.0, 120.0])),
            "P": float(np.interp(data["P"], [5.0, 6.0], [10.0, 80.0])),
            "K": float(np.interp(data["K"], [60.0, 150.0], [20.0, 150.0])),
            "temperature": float(np.interp(data["temperature"], [0.0, 100.0], [18.0, 28.0])),
            "humidity": float(np.interp(data["humidity"], [0.002, 0.005], [65.0, 85.0])),
            "ph": float(data["ph"]),
            "rainfall": float(np.clip(data["rainfall"], a_min=0, a_max=290.0)),
        }
        return calibrated

    # ------------------------------------------------------------------
    # Prediksi
    # ------------------------------------------------------------------
    def predict(self, input_data: dict) -> dict:
        """Mengembalikan dict berisi data terkalibrasi dan rekomendasi tanaman."""
        calibrated_data = self.calibrate_input(input_data)

        features_df = pd.DataFrame([calibrated_data])
        scaled_features = self.scaler.transform(features_df)
        prediction = self.model.predict(scaled_features)
        recommendation = self.encoder.inverse_transform(prediction)[0]

        logger.info("Prediksi selesai → %s", recommendation)
        return {
            "calibrated_data": calibrated_data,
            "recommendation": recommendation,
        }


# Singleton — dimuat sekali saat aplikasi start
ml_service = MLService()
