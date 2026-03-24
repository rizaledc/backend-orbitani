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
    [106.975437, -6.701583], [106.975469, -6.701945], [106.975167, -6.702229],
    [106.975197, -6.702509], [106.976689, -6.703712], [106.977246, -6.704102],
    [106.977883, -6.704273], [106.978967, -6.704490], [106.978760, -6.704051],
    [106.979103, -6.703701], [106.978578, -6.702982], [106.978264, -6.702511],
    [106.977603, -6.701899], [106.977425, -6.702003], [106.977209, -6.701600],
    [106.977086, -6.701630], [106.976603, -6.701757], [106.976261, -6.701084],
    [106.975901, -6.701049], [106.975375, -6.700860], [106.974965, -6.700642],
    [106.974768, -6.700749], [106.974693, -6.700639], [106.974569, -6.700784],
    [106.975437, -6.701583]
]

import json

_gee_ready = False

def _init_gee() -> None:
    """
    Inisialisasi koneksi ke GEE menggunakan Service Account.

    Urutan prioritas kredensial:
      1. File  : gee-key.json (untuk dev/local)
      2. EnvVar: GEE_JSON_KEY (string JSON, untuk Vercel / cloud deployment)
    """
    global _gee_ready
    try:
        if os.path.exists(GEE_KEY_PATH):
            # --- Prioritas 1: File lokal ---
            credentials = ee.ServiceAccountCredentials(
                GEE_SERVICE_ACCOUNT, GEE_KEY_PATH
            )
            logger.info("GEE: menggunakan kredensial dari file gee-key.json")
        else:
            # --- Prioritas 2: Environment variable (Vercel / cloud) ---
            gee_json_str = os.getenv("GEE_JSON_KEY")
            if not gee_json_str:
                logger.warning(
                    "gee-key.json tidak ditemukan dan GEE_JSON_KEY tidak di-set — GEE dinonaktifkan"
                )
                return

            key_data = json.loads(gee_json_str)

            # Buat objek Credentials langsung dari dict (tidak menulis ke disk)
            credentials = ee.ServiceAccountCredentials(
                GEE_SERVICE_ACCOUNT, key_data=key_data
            )
            logger.info("GEE: menggunakan kredensial dari env var GEE_JSON_KEY")

        ee.Initialize(credentials=credentials, project=GEE_PROJECT)
        _gee_ready = True
        logger.info("GEE berhasil diinisialisasi (project=%s)", GEE_PROJECT)

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
        import numpy as np
        # 1. Validasi Geofencing lokal
        poly = Polygon(LAHAN_HIBISC_COORDS)
        if not is_inside_hibisc(lat, lon):
            return {"error": "Luar area", "message": "Koordinat berada di luar Lahan Hibisc."}

        # 2. Setup 10 Grid Points di dalam area Hibisc
        roi_geom = ee.Geometry.Polygon([LAHAN_HIBISC_COORDS])
        min_lon, min_lat, max_lon, max_lat = poly.bounds
        
        # Buat grid resolusi untuk mencari titik valid
        lons = np.linspace(min_lon, max_lon, 6)
        lats = np.linspace(min_lat, max_lat, 6)
        
        valid_points = []
        for x in lons:
            for y in lats:
                pt = Point(x, y)
                if poly.contains(pt):
                    valid_points.append([x, y])
                    
        # Filter tepat 10 titik yang menyebar
        if len(valid_points) > 10:
            step = len(valid_points) / 10.0
            selected_coords = [valid_points[int(i * step)] for i in range(10)]
        else:
            selected_coords = valid_points

        if not selected_coords:
            return {"error": "Geometry Error", "message": "Gagal menemukan 10 titik dalam poligon."}

        # Buat FeatureCollection dari ke-10 titik tersebut (ditambah buffer kecil agar nilainya robust)
        features = [ee.Feature(ee.Geometry.Point(coord).buffer(10)) for coord in selected_coords]
        points_fc = ee.FeatureCollection(features)
        
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=365)

        # 3. Koleksi Landsat 8 TOA (difilter keseluruhan ROI)
        landsat = (
            ee.ImageCollection("LANDSAT/LC08/C02/T1_TOA")
            .filterDate(start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
            .filterBounds(roi_geom)
            .filter(ee.Filter.lt("CLOUD_COVER", 30))
            .median()
        )
        
        # 3.5. Koleksi Landsat 8 L2 (Surface Temperature) untuk Suhu Akurat
        landsat_l2 = (
            ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
            .filterDate(start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
            .filterBounds(roi_geom)
            .filter(ee.Filter.lt("CLOUD_COVER", 30))
            .median()
        )

        # 4. Kalkulasi Indeks (Sesuai rumus riset Hibisc)
        n_index = landsat.select("B3").divide(landsat.select("B4")).log10().multiply(10).rename("N")
        p_index = landsat.select("B4").divide(landsat.select("B5")).multiply(8).rename("P")
        k_index = landsat.select("B6").multiply(500).rename("K")
        ph_index = landsat.select("B4").divide(landsat.select("B2")).multiply(2).add(6).rename("ph")
        
        # Kalibrasi Suhu C2 L2 (ST_B10 scale: 0.00341802, offset: 149.0). Konversi Kelvin -> Celcius.
        tci = landsat_l2.select("ST_B10").multiply(0.00341802).add(149.0).subtract(273.15).rename("temp")
        
        ndti = landsat.normalizedDifference(["B6", "B7"]).rename("humidity")

        chirps = (
            ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")
            .filterDate(start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
            .filterBounds(roi_geom)
            .select("precipitation").sum().rename("rainfall")
        )

        composite = n_index.addBands([p_index, k_index, ph_index, tci, ndti, chirps])

        # 5. Reduksi rata-rata ditiap titik (ReduceRegions)
        stats_collection = composite.reduceRegions(
            collection=points_fc,
            reducer=ee.Reducer.mean(),
            scale=30
        ).getInfo()

        features_data = stats_collection.get("features", [])
        avg_stats = {"N": 0.0, "P": 0.0, "K": 0.0, "ph": 0.0, "temp": 0.0, "humidity": 0.0, "rainfall": 0.0}
        valid_count = 0
        
        # Iterasi 10 titik dan rata-ratakan hasilnya untuk 1 konklusi Lahan 
        for feat in features_data:
            props = feat.get("properties", {})
            if props.get("N") is not None:
                avg_stats["N"] += float(props.get("N", 0))
                avg_stats["P"] += float(props.get("P", 0))
                avg_stats["K"] += float(props.get("K", 0))
                avg_stats["ph"] += float(props.get("ph", 0))
                avg_stats["temp"] += float(props.get("temp", 0))
                avg_stats["humidity"] += float(props.get("humidity", 0))
                avg_stats["rainfall"] += float(props.get("rainfall", 0))
                valid_count += 1
                
        if valid_count > 0:
            for key in avg_stats:
                avg_stats[key] /= valid_count
        else:
            return {"error": "Zero Result", "message": "Tidak ada sinyal Landsat divalidasi pada ke-10 titik tersebut."}

        # 6. Simpan ke Supabase (satellite_results)
        payload = {
            "lahan_id": lahan_id,
            "longitude": lon,  # Pusat klik pertama 
            "latitude": lat,   # Pusat klik pertama
            "n_value": float(avg_stats["N"]),
            "p_value": float(avg_stats["P"]),
            "k_value": float(avg_stats["K"]),
            "ph": float(avg_stats["ph"]),
            "temperature": float(avg_stats["temp"]),
            "humidity": float(avg_stats["humidity"]),
            "rainfall": float(avg_stats["rainfall"]),
            "recommendation": "Pending Analysis",
            "extracted_at": datetime.utcnow().isoformat()
        }

        res = db.table("satellite_results").insert(payload).execute()
        return {"status": "success", "data": res.data}

    except Exception as e:
        logger.error("Error GEE Point processing: %s", e)
        return {"error": "Internal Error", "message": str(e)}