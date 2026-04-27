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
    points_data: list[dict]
) -> tuple[list[dict], list[dict]]:
    """
    Untuk setiap data satelit di titik koordinat:
      1. Panggil ml_service.predict().
      2. Kumpulkan hasil, hitung frekuensi, dan kembalikan Top-K dalam persentase.

    Args:
        points_data: List dict berisi data satelit per titik dari GEE.

    Returns:
        Tuple: (List Top-K rekomendasi, List data mentah titik + rekomendasi)
    """
    predictions: list[str] = []

    for i, data in enumerate(points_data):
        input_data = {
            "n":           data.get("n", 0),
            "p":           data.get("p", 0),
            "k":           data.get("k", 0),
            "temperature": data.get("temperature", 0),
            "humidity":    data.get("humidity", 0),
            "ph":          data.get("ph", 0),
            "rainfall":    data.get("rainfall", 0),
        }
        try:
            result = predict(input_data)
            tanaman = result.get("ai_recommendation", "Unknown")
            data["ai_recommendation"] = tanaman
            predictions.append(tanaman)
            logger.debug("Titik %d/%d (lon=%.4f, lat=%.4f) → %s", i + 1, len(points_data), data.get("lon"), data.get("lat"), tanaman)
        except Exception as e:
            logger.warning("Prediksi gagal pada titik %d (lon=%.4f, lat=%.4f): %s", i + 1, len(points_data), data.get("lon"), data.get("lat"), e)
            data["ai_recommendation"] = "Failed"

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
    return hasil, points_data


# ---------------------------------------------------------------------------
# Public API: Fungsi utama yang dipanggil endpoint
# ---------------------------------------------------------------------------
def run_spatial_analysis(
    koordinat: Any,
) -> tuple[list[dict], list[dict]]:
    """
    Entry point analisis spasial lahan.

    Args:
        koordinat: Field koordinat dari tabel lahan (GeoJSON dict atau array nested).

    Returns:
        Tuple: (List Top-K rekomendasi tanaman, List data mentah 10 titik)

    Raises:
        ValueError: Jika koordinat tidak valid / tidak bisa di-parse.
        RuntimeError: Jika sampling gagal atau semua prediksi ML gagal.
    """
    from app.services.gee_service import extract_multi_point_data
    logger.info("Memulai analisis spasial...")

    # Step 1: Parse koordinat → Shapely Polygon
    polygon = _geojson_to_polygon(koordinat)
    logger.info("Polygon valid: bounds=%s, area=%.8f", polygon.bounds, polygon.area)

    # Step 2: Sampling titik acak di dalam polygon
    points = _sample_points_within(polygon, n=N_SAMPLES)
    
    # Step 3: Ekstraksi data satelit asli via GEE (Batch)
    polygon_coords = list(polygon.exterior.coords)
    points_data = extract_multi_point_data(polygon_coords, points)

    # Step 4: Prediksi + Majority Voting
    hasil_rekomendasi, data_lengkap_titik = _predict_and_aggregate(points_data)

    return hasil_rekomendasi, data_lengkap_titik
