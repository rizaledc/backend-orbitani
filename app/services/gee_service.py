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
    Inisialisasi koneksi ke GEE menggunakan Service Account via Environment Variables (Azure).
    """
    global _gee_ready
    try:
        sa_email = os.environ.get("GEE_SERVICE_ACCOUNT")
        json_key_str = os.environ.get("GEE_JSON_KEY")
        project = os.environ.get("GEE_PROJECT")

        if sa_email and json_key_str:
            key_dict = json.loads(json_key_str)
            credentials = ee.ServiceAccountCredentials(sa_email, key_data=key_dict)
            ee.Initialize(credentials=credentials, project=project)
            logger.info("GEE berhasil diinisialisasi menggunakan credentials dari environment variables.")
        else:
            # Fallback untuk lokal
            ee.Initialize(project=project)
            logger.info(f"GEE berhasil diinisialisasi (fallback lokal project={project}).")
            
        _gee_ready = True

    except Exception as e:
        logger.error("Gagal inisialisasi GEE: %s", e)
        print(f"Gagal inisialisasi GEE: {e}")

_init_gee()


# ---------------------------------------------------------------------------
# Sentinel-2: Cloud Masking & Scaling
# ---------------------------------------------------------------------------
def _mask_s2_clouds(image):
    """
    Masking piksel awan & bayangan awan menggunakan band SCL (Scene Classification Layer).
    SCL values yang di-mask:
      3 = Cloud Shadow
      8 = Cloud Medium Probability
      9 = Cloud High Probability
     10 = Thin Cirrus
    """
    scl = image.select("SCL")
    mask = (
        scl.neq(3)   # Cloud Shadow
        .And(scl.neq(8))   # Cloud Medium Probability
        .And(scl.neq(9))   # Cloud High Probability
        .And(scl.neq(10))  # Thin Cirrus
    )
    return image.updateMask(mask)


def _scale_s2(image):
    """
    Mengalikan band optik Sentinel-2 dengan faktor skala 0.0001
    sesuai spesifikasi COPERNICUS/S2_SR_HARMONIZED.
    Band yang di-scale: B1-B9, B11, B12, B8A
    """
    optical_bands = image.select(
        ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12"]
    ).multiply(0.0001)
    # Gabungkan kembali band non-optik (SCL, QA60, dll.)
    return image.addBands(optical_bands, overwrite=True)


def _preprocess_s2(image):
    """Pipeline preprocessing: cloud mask → scale."""
    return _scale_s2(_mask_s2_clouds(image))


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
    Hybrid Approach:
      - Sentinel-2 (COPERNICUS/S2_SR_HARMONIZED) → parameter optik (N, P, K, pH, humidity)
      - Landsat-9 L2 (LANDSAT/LC09/C02/T1_L2)   → suhu permukaan (ST_B10)
      - CHIRPS                                     → curah hujan
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
        date_start_str = start_date.strftime("%Y-%m-%d")
        date_end_str = end_date.strftime("%Y-%m-%d")

        # =====================================================================
        # 3. Sentinel-2 SR Harmonized (Parameter Optik) — Resolusi 10m
        # =====================================================================
        sentinel2 = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterDate(date_start_str, date_end_str)
            .filterBounds(roi_geom)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
            .map(_preprocess_s2)
            .median()
        )

        # Band Mapping Sentinel-2:
        #   B2=Blue, B3=Green, B4=Red, B8=NIR, B11=SWIR1, B12=SWIR2
        n_index = sentinel2.select("B3").divide(sentinel2.select("B4")).log10().multiply(10).rename("N")
        p_index = sentinel2.select("B4").divide(sentinel2.select("B8")).multiply(8).rename("P")
        k_index = sentinel2.select("B11").multiply(500).rename("K")
        ph_index = sentinel2.select("B4").divide(sentinel2.select("B2")).multiply(2).add(6).rename("ph")
        ndti = sentinel2.normalizedDifference(["B11", "B12"]).rename("humidity")

        # Komposit optik Sentinel-2
        optical_composite = n_index.addBands([p_index, k_index, ph_index, ndti])

        # =====================================================================
        # 3.5. Landsat-9 L2 (Surface Temperature ONLY) — Resolusi 30m
        # =====================================================================
        landsat_l2 = (
            ee.ImageCollection("LANDSAT/LC09/C02/T1_L2")
            .filterDate(date_start_str, date_end_str)
            .filterBounds(roi_geom)
            .filter(ee.Filter.lt("CLOUD_COVER", 30))
            .median()
        )
        # Kalibrasi Suhu C2 L2 (ST_B10 scale: 0.00341802, offset: 149.0). Konversi Kelvin -> Celcius.
        tci = landsat_l2.select("ST_B10").multiply(0.00341802).add(149.0).subtract(273.15).rename("temp")

        # =====================================================================
        # 4. CHIRPS Rainfall
        # =====================================================================
        chirps = (
            ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")
            .filterDate(date_start_str, date_end_str)
            .filterBounds(roi_geom)
            .select("precipitation").sum().rename("rainfall")
        )

        # =====================================================================
        # 5. Reduce Regions — Sentinel-2 @10m, Landsat thermal @30m
        # =====================================================================
        # Reduce optik (Sentinel-2) pada resolusi 10m
        optical_stats = optical_composite.reduceRegions(
            collection=points_fc,
            reducer=ee.Reducer.mean(),
            scale=10
        ).getInfo()

        # Reduce suhu (Landsat L2) pada resolusi 30m
        temp_stats = tci.reduceRegions(
            collection=points_fc,
            reducer=ee.Reducer.mean(),
            scale=30
        ).getInfo()

        # Reduce curah hujan (CHIRPS) pada resolusi 5000m (native ~5.5km)
        rain_stats = chirps.reduceRegions(
            collection=points_fc,
            reducer=ee.Reducer.mean(),
            scale=5000
        ).getInfo()

        # =====================================================================
        # 6. Rata-ratakan 10 titik → 1 konklusi
        # =====================================================================
        avg_stats = {"N": 0.0, "P": 0.0, "K": 0.0, "ph": 0.0, "temp": 0.0, "humidity": 0.0, "rainfall": 0.0}

        # -- Optik dari Sentinel-2 --
        optical_features = optical_stats.get("features", [])
        valid_optical = 0
        for feat in optical_features:
            props = feat.get("properties", {})
            if props.get("N") is not None:
                avg_stats["N"] += float(props.get("N", 0))
                avg_stats["P"] += float(props.get("P", 0))
                avg_stats["K"] += float(props.get("K", 0))
                avg_stats["ph"] += float(props.get("ph", 0))
                avg_stats["humidity"] += float(props.get("humidity", 0))
                valid_optical += 1

        if valid_optical > 0:
            for key in ["N", "P", "K", "ph", "humidity"]:
                avg_stats[key] /= valid_optical
        else:
            return {"error": "Zero Result", "message": "Tidak ada sinyal Sentinel-2 yang valid pada ke-10 titik tersebut."}

        # -- Suhu dari Landsat-9 L2 --
        temp_features = temp_stats.get("features", [])
        valid_temp = 0
        for feat in temp_features:
            props = feat.get("properties", {})
            if props.get("temp") is not None:
                avg_stats["temp"] += float(props.get("temp", 0))
                valid_temp += 1
        if valid_temp > 0:
            avg_stats["temp"] /= valid_temp
        else:
            logger.warning("Tidak ada data suhu Landsat L2 — temp di-set ke 0.")

        # -- Curah hujan dari CHIRPS --
        rain_features = rain_stats.get("features", [])
        valid_rain = 0
        for feat in rain_features:
            props = feat.get("properties", {})
            if props.get("rainfall") is not None:
                avg_stats["rainfall"] += float(props.get("rainfall", 0))
                valid_rain += 1
        if valid_rain > 0:
            avg_stats["rainfall"] /= valid_rain

        # 7. Simpan ke Supabase (satellite_results)
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