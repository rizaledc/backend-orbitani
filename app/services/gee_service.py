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

def _init_gee():
    global _gee_ready
    try:
        project = os.environ.get("GEE_PROJECT")
        json_input = os.environ.get("GEE_JSON_KEY")
        
        if json_input and project:
            # 1. Konversi input ke DICTIONARY (Handle string vs dict)
            if isinstance(json_input, str):
                # Bersihkan kotoran tanda kutip Azure jika ada
                json_input = json_input.strip("'").strip('"')
                info = json.loads(json_input)
            else:
                info = json_input
            
            # 2. Autentikasi Eksplisit (Tanpa File Sementara)
            credentials = ee.ServiceAccountCredentials(
                info['client_email'], 
                key_data=json.dumps(info)
            )
            
            ee.Initialize(credentials=credentials, project=project)
            logger.info("GEE Authenticated successfully via Service Account Credentials!")
        else:
            # Fallback untuk testing lokal
            ee.Initialize(project=project)
            logger.info("GEE Initialized via Local Fallback.")
            
        _gee_ready = True
    except Exception as e:
        logger.error(f"GAGAL LOGIN GEE: {str(e)}")

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
            # TANPA filter CLOUD_COVER — .median() yang handle noise awan
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

        # Reduce curah hujan (CHIRPS) — buffer 5km dari centroid agar pasti menangkap piksel
        rain_dict = chirps.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=roi_geom.centroid().buffer(5000),
            scale=5000,
            maxPixels=1e9
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
            avg_stats["temp"] = 26.5  # Fallback Suhu Rata-rata Bogor
            logger.warning("Landsat gagal, pakai suhu fallback 26.5 C")

        # -- Curah hujan dari CHIRPS (reduceRegion pada polygon) --
        rainfall_val = rain_dict.get("rainfall")
        avg_stats["rainfall"] = float(rainfall_val) if rainfall_val is not None else 0.0

        # Fallback: jika curah hujan masih 0 atau terlalu rendah, gunakan rata-rata Bogor
        if avg_stats["rainfall"] == 0.0 or avg_stats["rainfall"] < 100:
            avg_stats["rainfall"] = 3500.0  # Fallback Curah Hujan Bogor ~3500 mm/tahun
            logger.warning("CHIRPS gagal/terlalu rendah, pakai curah hujan fallback 3500 mm")

        # 7. Prediksi ML (Random Forest) sebelum menyimpan ke database
        ml_recommendation = "Pending Analysis"
        calibrated_data = None
        try:
            from app.services.ml_service import predict
            ml_input = {
                "N": avg_stats["N"],
                "P": avg_stats["P"],
                "K": avg_stats["K"],
                "temperature": avg_stats["temp"],
                "humidity": avg_stats["humidity"],
                "ph": avg_stats["ph"],
                "rainfall": avg_stats["rainfall"],
            }
            ml_result = predict(ml_input)
            ml_recommendation = ml_result["recommendation"]
            calibrated_data = ml_result["calibrated_data"]
            logger.info("ML Prediction berhasil: %s", ml_recommendation)
        except Exception as ml_err:
            logger.warning("ML Prediction gagal (fallback Pending): %s", ml_err)

        # 8. Simpan ke Supabase (satellite_results) dengan rekomendasi ML
        payload = {
            "lahan_id": lahan_id,
            "longitude": lon,
            "latitude": lat,
            "n_value": float(avg_stats["N"]),
            "p_value": float(avg_stats["P"]),
            "k_value": float(avg_stats["K"]),
            "ph": float(avg_stats["ph"]),
            "temperature": float(avg_stats["temp"]),
            "humidity": float(avg_stats["humidity"]),
            "rainfall": float(avg_stats["rainfall"]),
            "recommendation": ml_recommendation,
            "extracted_at": datetime.utcnow().isoformat()
        }

        res = db.table("satellite_results").insert(payload).execute()
        return {
            "status": "success",
            "data": res.data,
            "ml_recommendation": ml_recommendation,
            "calibrated_data": calibrated_data,
        }

    except Exception as e:
        logger.error("Error GEE Point processing: %s", e)
        return {"error": "Internal Error", "message": str(e)}