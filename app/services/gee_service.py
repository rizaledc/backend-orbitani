import logging
import os
from datetime import datetime, timedelta
import ee
from shapely.geometry import Point, Polygon
from app.db.database import supabase as db  # Client Supabase

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Konfigurasi GEE & Pagar Gaib (Geofencing)
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GEE_KEY_PATH = os.path.join(BASE_DIR, "gee-key.json")
GEE_SERVICE_ACCOUNT = "orbitani-gee@studycilacap.iam.gserviceaccount.com"
GEE_PROJECT = "studycilacap"

# Poligon Lahan Hibisc (Cilacap, Jawa Tengah) - Format: [Longitude, Latitude]
LAHAN_HIBISC_COORDS = [
    [106.975437, -6.701583],
    [106.979299, -6.701583],
    [106.979299, -6.705214],
    [106.975437, -6.705214],
    [106.975437, -6.701583],
]

_gee_ready = False

def _init_gee() -> None:
    """Inisialisasi koneksi ke GEE menggunakan Service Account."""
    global _gee_ready
    try:
        if not os.path.exists(GEE_KEY_PATH):
            logger.warning("gee-key.json tidak ditemukan di %s — GEE dinonaktifkan", GEE_KEY_PATH)
            return
        credentials = ee.ServiceAccountCredentials(
            GEE_SERVICE_ACCOUNT, GEE_KEY_PATH
        )
        ee.Initialize(credentials=credentials, project=GEE_PROJECT)
        _gee_ready = True
        logger.info("GEE Berhasil diinisialisasi (project=%s)", GEE_PROJECT)
    except Exception as e:
        logger.error("Gagal inisialisasi GEE: %s", e)

_init_gee()

# ---------------------------------------------------------------------------
# Fungsi Validasi Area (Geofencing)
# ---------------------------------------------------------------------------
def is_inside_hibisc(lat: float, lon: float) -> bool:
    """Mengecek apakah titik berada di dalam area Lahan Hibisc."""
    poly = Polygon(LAHAN_HIBISC_COORDS)
    point = Point(lon, lat)
    return poly.contains(point)

# ---------------------------------------------------------------------------
# Fungsi Utama: Ekstraksi Data Satelit per Titik
# ---------------------------------------------------------------------------
def process_point_satellite_data(lahan_id: int, lat: float, lon: float) -> dict:
    """
    Menarik data GEE untuk satu titik koordinat dan menyimpan ke Supabase.
    """
    try:
        # 1. Validasi Geofencing lokal
        if not is_inside_hibisc(lat, lon):
            return {"error": "Luar area", "message": "Koordinat berada di luar Lahan Hibisc."}

        # 2. Setup GEE Point dengan Buffer 30m
        point_geom = ee.Geometry.Point([lon, lat]).buffer(30)
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=365)

        # 3. Koleksi Landsat 8 TOA
        landsat = (
            ee.ImageCollection("LANDSAT/LC08/C02/T1_TOA")
            .filterDate(start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
            .filterBounds(point_geom)
            .filter(ee.Filter.lt("CLOUD_COVER", 30))
            .median()
        )

        # 4. Kalkulasi Indeks (Sesuai rumus riset Hibisc)
        n_index = landsat.select("B3").divide(landsat.select("B4")).log10().multiply(10).rename("N")
        p_index = landsat.select("B4").divide(landsat.select("B5")).multiply(8).rename("P")
        k_index = landsat.select("B6").multiply(500).rename("K")
        ph_index = landsat.select("B4").divide(landsat.select("B2")).multiply(2).add(6).rename("ph")
        tci = landsat.select("B10").subtract(270).divide(0.5).clamp(0, 100).rename("temp")
        ndti = landsat.normalizedDifference(["B6", "B7"]).rename("humidity")

        chirps = (
            ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")
            .filterDate(start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
            .filterBounds(point_geom)
            .select("precipitation").sum().rename("rainfall")
        )

        composite = n_index.addBands([p_index, k_index, ph_index, tci, ndti, chirps])

        # 5. Reduksi nilai rata-rata pada area buffer 30m
        stats = composite.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=point_geom,
            scale=30
        ).getInfo()

        # 6. Simpan ke Supabase (satellite_results)
        payload = {
            "lahan_id": lahan_id,
            "longitude": lon,
            "latitude": lat,
            "n_value": float(stats.get("N", 0)),
            "p_value": float(stats.get("P", 0)),
            "k_value": float(stats.get("K", 0)),
            "ph": float(stats.get("ph", 0)),
            "temperature": float(stats.get("temp", 0)),
            "humidity": float(stats.get("humidity", 0)),
            "rainfall": float(stats.get("rainfall", 0)),
            "recommendation": "Pending Analysis",
            "extracted_at": datetime.utcnow().isoformat()
        }

        res = db.table("satellite_results").insert(payload).execute()
        return {"status": "success", "data": res.data}

    except Exception as e:
        logger.error("Error GEE Point processing: %s", e)
        return {"error": "Internal Error", "message": str(e)}