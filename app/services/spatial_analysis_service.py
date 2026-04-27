"""
spatial_analysis_service.py
Layanan analisis spasial untuk Orbitani.

Alur Kerja:
  1. Terima objek Shapely Polygon dari koordinat GeoJSON lahan.
  2. Buat bounding box → generate titik acak di dalamnya.
  3. Filter titik yang benar-benar .within(polygon) hingga tepat N_SAMPLES titik valid.
  4. Panggil ml_service.predict() untuk setiap titik (batch loop).
  5. Hitung frekuensi kemunculan tiap tanaman (majority voting).
  6. Konversi ke persentase → kembalikan Top-K hasil.

Catatan performa:
  - Fungsi ini bersifat CPU-bound (shapely + sklearn).
  - Untuk skalabilitas tinggi, pertimbangkan menjalankan via run_in_executor.
  - Saat ini dipanggil langsung dari endpoint (cocok untuk beban normal).
"""
import logging
import random
from collections import Counter
from typing import Any

from shapely.geometry import Point, Polygon, shape

from app.services.ml_service import predict

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Konstanta
# ---------------------------------------------------------------------------
N_SAMPLES: int = 10          # Jumlah titik sampling acak yang diambil per lahan
TOP_K: int = 3               # Jumlah rekomendasi teratas yang dikembalikan
MAX_ATTEMPTS: int = 10_000   # Batas iterasi untuk menghindari infinite loop pada poligon kecil/aneh


# ---------------------------------------------------------------------------
# Helper: Konversi GeoJSON koordinat → Shapely Polygon
# ---------------------------------------------------------------------------
def _geojson_to_polygon(koordinat: Any) -> Polygon:
    """
    Konversi field koordinat lahan (GeoJSON Polygon dict atau array nested)
    menjadi objek Shapely Polygon.

    Format yang didukung:
      - GeoJSON dict: {"type": "Polygon", "coordinates": [[[lon, lat], ...]]}
      - Array langsung: [[[lon, lat], ...]]   (GeoJSON coordinates value)
    """
    if isinstance(koordinat, dict):
        # GeoJSON Polygon object lengkap
        try:
            return shape(koordinat)  # shapely.geometry.shape handles GeoJSON natively
        except Exception as e:
            raise ValueError(f"Gagal parse GeoJSON Polygon: {e}") from e

    if isinstance(koordinat, list):
        # Array koordinat langsung (tanpa wrapper GeoJSON)
        try:
            exterior = koordinat[0]   # ring pertama = exterior
            return Polygon([(pt[0], pt[1]) for pt in exterior])
        except Exception as e:
            raise ValueError(f"Gagal parse array koordinat: {e}") from e

    raise ValueError(f"Format koordinat tidak dikenal: {type(koordinat)}")


# ---------------------------------------------------------------------------
# Core: Sample N titik acak yang valid di dalam poligon
# ---------------------------------------------------------------------------
def _sample_points_within(polygon: Polygon, n: int = N_SAMPLES) -> list[tuple[float, float]]:
    """
    Generate tepat `n` titik acak (lon, lat) yang berada di dalam `polygon`.
    Menggunakan metode rejection sampling berbasis bounding box.

    Raises:
        RuntimeError: Jika MAX_ATTEMPTS terlampaui (poligon sangat kecil / tidak valid).
    """
    min_lon, min_lat, max_lon, max_lat = polygon.bounds
    sampled: list[tuple[float, float]] = []
    attempts = 0

    while len(sampled) < n:
        if attempts >= MAX_ATTEMPTS:
            raise RuntimeError(
                f"Gagal mengumpulkan {n} titik dalam {MAX_ATTEMPTS} percobaan. "
                "Pastikan poligon lahan valid dan memiliki area yang cukup."
            )
        lon = random.uniform(min_lon, max_lon)
        lat = random.uniform(min_lat, max_lat)
        pt = Point(lon, lat)
        if pt.within(polygon):
            sampled.append((lon, lat))
        attempts += 1

    logger.info("Sampling selesai: %d titik valid dari %d percobaan.", len(sampled), attempts)
    return sampled


# ---------------------------------------------------------------------------
# Core: Prediksi batch + Majority Voting → Top-K hasil
# ---------------------------------------------------------------------------
def _predict_and_aggregate(
    points: list[tuple[float, float]],
    satellite_data_template: dict | None = None,
) -> list[dict]:
    """
    Untuk setiap titik koordinat:
      1. Ambil data referensi dari template satelit (jika ada), atau gunakan nilai default.
      2. Panggil ml_service.predict().
      3. Kumpulkan hasil, hitung frekuensi, dan kembalikan Top-K dalam persentase.

    Args:
        points: List (lon, lat) dari _sample_points_within.
        satellite_data_template: Dict data satelit yang bisa dijadikan basis prediksi
                                 (N, P, K, temperature, humidity, ph, rainfall).
                                 Jika None, gunakan nilai rata-rata default Indonesia.

    Returns:
        List dict: [{"tanaman": str, "persentase": float}, ...] — Top-K, diurutkan descending.
    """
    # Nilai default realistis untuk konteks pertanian Indonesia
    # (digunakan jika tidak ada data satelit aktual)
    DEFAULT_TEMPLATE = {
        "n": 50.0,
        "p": 30.0,
        "k": 50.0,
        "temperature": 25.0,
        "humidity": 75.0,
        "ph": 6.5,
        "rainfall": 150.0,
    }

    base_data = satellite_data_template or DEFAULT_TEMPLATE
    predictions: list[str] = []

    for i, (lon, lat) in enumerate(points):
        # Terapkan deviasi acak +/- 5% hingga 10% untuk mensimulasikan
        # kondisi tanah/cuaca yang bervariasi di area poligon
        input_data = {
            "n":           base_data.get("n", base_data.get("N", DEFAULT_TEMPLATE["n"])) * random.uniform(0.9, 1.1),
            "p":           base_data.get("p", base_data.get("P", DEFAULT_TEMPLATE["p"])) * random.uniform(0.9, 1.1),
            "k":           base_data.get("k", base_data.get("K", DEFAULT_TEMPLATE["k"])) * random.uniform(0.9, 1.1),
            "temperature": base_data.get("temperature", DEFAULT_TEMPLATE["temperature"]) * random.uniform(0.95, 1.05),
            "humidity":    base_data.get("humidity", DEFAULT_TEMPLATE["humidity"]) * random.uniform(0.95, 1.05),
            "ph":          base_data.get("ph", DEFAULT_TEMPLATE["ph"]) * random.uniform(0.95, 1.05),
            "rainfall":    base_data.get("rainfall", DEFAULT_TEMPLATE["rainfall"]) * random.uniform(0.9, 1.1),
        }
        try:
            result = predict(input_data)
            tanaman = result.get("ai_recommendation", "Unknown")
            predictions.append(tanaman)
            logger.debug("Titik %d/%d (lon=%.4f, lat=%.4f) → %s", i + 1, len(points), lon, lat, tanaman)
        except Exception as e:
            logger.warning("Prediksi gagal pada titik %d (lon=%.4f, lat=%.4f): %s", i + 1, lon, lat, e)
            # Titik yang gagal diprediksi dilewati (tidak crash seluruh proses)

    if not predictions:
        raise RuntimeError("Semua prediksi gagal. Periksa kondisi model ML.")

    # Majority Voting
    total = len(predictions)
    counter = Counter(predictions)
    top_k = counter.most_common(TOP_K)

    hasil = [
        {
            "tanaman":     tanaman,
            "persentase": round((count / total) * 100, 1),
        }
        for tanaman, count in top_k
    ]

    logger.info(
        "Agregasi selesai: %d prediksi, Top-%d → %s",
        total, TOP_K, [(h["tanaman"], h["persentase"]) for h in hasil],
    )
    return hasil


# ---------------------------------------------------------------------------
# Public API: Fungsi utama yang dipanggil endpoint
# ---------------------------------------------------------------------------
def run_spatial_analysis(
    koordinat: Any,
    satellite_data_template: dict | None = None,
) -> list[dict]:
    """
    Entry point analisis spasial lahan.

    Args:
        koordinat: Field koordinat dari tabel lahan (GeoJSON dict atau array nested).
        satellite_data_template: Data satelit aktual lahan untuk digunakan sebagai basis
                                 fitur ML. Ambil dari satellite_results terbaru, atau None
                                 untuk menggunakan nilai default.

    Returns:
        List Top-K rekomendasi tanaman dalam format:
        [{"tanaman": "Pisang", "persentase": 70.0}, ...]

    Raises:
        ValueError: Jika koordinat tidak valid / tidak bisa di-parse.
        RuntimeError: Jika sampling gagal atau semua prediksi ML gagal.
    """
    logger.info("Memulai analisis spasial...")

    # Step 1: Parse koordinat → Shapely Polygon
    polygon = _geojson_to_polygon(koordinat)
    logger.info("Polygon valid: bounds=%s, area=%.8f", polygon.bounds, polygon.area)

    # Step 2: Sampling titik acak di dalam polygon
    points = _sample_points_within(polygon, n=N_SAMPLES)

    # Step 3: Prediksi + Majority Voting
    hasil = _predict_and_aggregate(points, satellite_data_template=satellite_data_template)

    return hasil
