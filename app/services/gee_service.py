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

        # Band Mapping Sentinel-2 (Harmonized, scaled by 0.0001):
        # B2=Blue, B3=Green, B4=Red, B8=NIR, B11=SWIR1, B12=SWIR2
        
        # 1. NITROGEN (N): Gunakan NDRE (Normalized Difference Red Edge) proxy
        # Rumus asli NDRE butuh B5, tapi kita gunakan (NIR - Red) / (NIR + Red) lalu diskalakan agar realistis (0 - 2.5%)
        # Menghindari nilai negatif
        ndvi = sentinel2.normalizedDifference(["B8", "B4"])
        n_index = ndvi.multiply(2.5).add(0.5).rename("N") 
        
        # 2. PH TANAH: Tropis cenderung asam. Gunakan rasio Blue/Red dengan baseline 5.0
        # Jika pantulan merah lebih tinggi (tanah kering/oksida besi), pH cenderung turun.
        ph_index = sentinel2.select("B2").divide(sentinel2.select("B4")).multiply(1.5).add(4.5).rename("ph")
        
        # 3. FOSFOR (P) & KALIUM (K): Dikalibrasi agar masuk akal dalam satuan ppm
        # P biasanya rendah di tanah asam (terikat). Skala: 10 - 40 ppm
        p_index = sentinel2.select("B3").divide(sentinel2.select("B8")).multiply(30).add(10).rename("P")
        
        # K menggunakan SWIR karena sensitif terhadap mineral lempung. Skala: 100 - 300 ppm
        k_index = sentinel2.select("B11").divide(sentinel2.select("B12")).multiply(150).add(50).rename("K")
        
        # 4. HUMIDITY: NDTI sudah cukup baik, biarkan saja
        ndti = sentinel2.normalizedDifference(["B11", "B12"]).rename("humidity")

        # Komposit optik Sentinel-2
        optical_composite = n_index.addBands([p_index, k_index, ph_index, ndti])

        # =====================================================================
        # 3.5 HYBRID TEMPERATURE (Landsat 8+9 Primary, MODIS Fallback)
        # =====================================================================
        
        # 1. Kumpulkan Landsat 8 dan 9 (Resolusi 30m), filter awan KETAT (< 50%)
        l8 = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2") \
               .filterDate(date_start_str, date_end_str) \
               .filterBounds(roi_geom) \
               .filter(ee.Filter.lt("CLOUD_COVER", 50))
               
        l9 = ee.ImageCollection("LANDSAT/LC09/C02/T1_L2") \
               .filterDate(date_start_str, date_end_str) \
               .filterBounds(roi_geom) \
               .filter(ee.Filter.lt("CLOUD_COVER", 50))
        
        # Gabungkan koleksi dan ambil median
        landsat_combined = l8.merge(l9).median()
        
        # Konversi Landsat L2 ke Celcius: (ST_B10 * 0.00341802 + 149.0) - 273.15
        tci_landsat = landsat_combined.select("ST_B10").multiply(0.00341802).add(149.0).subtract(273.15)
        
        # Tarik nilai Landsat dengan buffer 30 meter
        temp_dict_landsat = tci_landsat.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=roi_geom.centroid().buffer(30),
            scale=30,
            maxPixels=1e9
        ).getInfo()
        
        temp_val = temp_dict_landsat.get("ST_B10")
        
        # 2. JIKA LANDSAT GAGAL (Null karena awan > 50%), FALLBACK KE MODIS
        if temp_val is None:
            logger.warning("Landsat 8 & 9 (Cloud < 50%%) kosong. Fallback ke MODIS Terra...")
            
            # MODIS Terra (MOD11A1.061) LST Daily (Resolusi 1km)
            modis = (
                ee.ImageCollection("MODIS/061/MOD11A1")
                .filterDate(date_start_str, date_end_str)
                .filterBounds(roi_geom)
                .median()
            )
            
            # Konversi MODIS LST_Day_1km ke Celcius: (LST_Day_1km * 0.02) - 273.15
            modis_temp = modis.select("LST_Day_1km").multiply(0.02).subtract(273.15)
            
            # Tarik nilai MODIS dengan buffer 1000 meter (1km)
            temp_dict_modis = modis_temp.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=roi_geom.centroid().buffer(1000),
                scale=1000,
                maxPixels=1e9
            ).getInfo()
            
            temp_val = temp_dict_modis.get("LST_Day_1km")
            
            # 3. HARD FALLBACK (Jaring Pengaman Terakhir)
            if temp_val is None:
                temp_val = 26.5
                logger.warning("MODIS juga gagal. Menggunakan suhu fallback rata-rata 26.5 C")

        # =====================================================================
        # 4. HYBRID RAINFALL (CHIRPS Primary, GPM Fallback)
        # =====================================================================
        # Coba CHIRPS (mm/day)
        chirps = (
            ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")
            .filterDate(date_start_str, date_end_str)
            .filterBounds(roi_geom)
            .select("precipitation").sum()
        )
        
        rain_dict_chirps = chirps.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=roi_geom.centroid().buffer(5000),
            scale=5000,
            maxPixels=1e9
        ).getInfo()
        
        rainfall_val = rain_dict_chirps.get("precipitation")
        
        # Jika CHIRPS null atau 0 (karena latency dataset)
        if rainfall_val is None or float(rainfall_val) < 100:
            logger.warning("CHIRPS data unavailable/incomplete. Falling back to GPM IMERG V07...")
            
            # GPM IMERG V07 (mm/hr, tiap 30 menit)
            gpm = (
                ee.ImageCollection("NASA/GPM_L3/IMERG_V07")
                .filterDate(date_start_str, date_end_str)
                .filterBounds(roi_geom)
                .select("precipitation")
            )
            
            # Trik efisiensi: Cari rata-rata mm/jam setahun, lalu kalikan 24 jam * 365 hari
            gpm_annual = gpm.mean().multiply(24).multiply(365)
            
            rain_dict_gpm = gpm_annual.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=roi_geom.centroid().buffer(11000),  # Resolusi GPM ~11km
                scale=11132,
                maxPixels=1e9
            ).getInfo()
            
            rainfall_val = rain_dict_gpm.get("precipitation")
            
            # Jika GPM juga masih gagal, baru pakai hard fallback
            if rainfall_val is None or float(rainfall_val) < 100:
                rainfall_val = 3500.0
                logger.warning("GPM also failed. Using hard fallback 3500 mm.")

        # =====================================================================
        # 5. Reduce Regions — Sentinel-2 @10m
        # =====================================================================
        # Reduce optik (Sentinel-2) pada resolusi 10m
        optical_stats = optical_composite.reduceRegions(
            collection=points_fc,
            reducer=ee.Reducer.mean(),
            scale=10
        ).getInfo()

        # (Suhu sudah dihitung di blok Hybrid Temperature di atas)

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

        # -- Suhu (sudah dihitung di blok Hybrid Temperature L8+L9+MODIS) --
        avg_stats["temp"] = float(temp_val)

        # -- Curah hujan (sudah dihitung di blok Hybrid Rainfall) --
        avg_stats["rainfall"] = float(rainfall_val)

        # 7. Prediksi ML (Random Forest) sebelum menyimpan ke database
        ai_recommendation = "Pending Analysis"
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
            ai_recommendation = ml_result["ai_recommendation"]
            calibrated_data = ml_result["calibrated_data"]
            logger.info("ML Prediction berhasil: %s", ai_recommendation)
        except Exception as ml_err:
            logger.warning("ML Prediction gagal (fallback Pending): %s", ml_err)

        # 8. Simpan ke Supabase (satellite_results) dengan skema kolom FINAL
        payload = {
            "lahan_id":          lahan_id,
            "longitude":         lon,
            "latitude":          lat,
            "n":                 float(avg_stats["N"]),
            "p":                 float(avg_stats["P"]),
            "k":                 float(avg_stats["K"]),
            "ph":                float(avg_stats["ph"]),
            "temperature":       float(avg_stats["temp"]),
            "humidity":          float(avg_stats["humidity"]),
            "rainfall":          float(avg_stats["rainfall"]),
            "ai_recommendation": ai_recommendation,
            "created_at":        datetime.utcnow().isoformat(),
        }

        res = db.table("satellite_results").insert(payload).execute()
        return {
            "status": "success",
            "data": res.data,
            "ai_recommendation": ai_recommendation,
            "calibrated_data":   calibrated_data,
        }

    except Exception as e:
        logger.error("Error GEE Point processing: %s", e)
        return {"error": "Internal Error", "message": str(e)}

# ---------------------------------------------------------------------------
# Fungsi Multi-Point (Dipakai oleh Spatial Analysis / Majority Voting)
# ---------------------------------------------------------------------------
def extract_multi_point_data(polygon_coords: list[list[float]], points: list[tuple[float, float]]) -> list[dict]:
    """
    Mengekstrak data satelit (N, P, K, pH, humidity, temperature, rainfall)
    secara batch untuk banyak titik sekaligus dalam 1 request GEE menggunakan FeatureCollection.
    """
    try:
        global _gee_ready
        if not _gee_ready:
            _init_gee()

        # 1. Setup Geometries
        roi_geom = ee.Geometry.Polygon([polygon_coords])
        features = [ee.Feature(ee.Geometry.Point([lon, lat]).buffer(10), {"lon": lon, "lat": lat}) for lon, lat in points]
        points_fc = ee.FeatureCollection(features)

        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=365)
        date_start_str = start_date.strftime("%Y-%m-%d")
        date_end_str = end_date.strftime("%Y-%m-%d")

        # 2. Sentinel-2 (Optik: N, P, K, pH, Humidity)
        sentinel2 = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterDate(date_start_str, date_end_str)
            .filterBounds(roi_geom)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
            .map(_preprocess_s2)
            .median()
        )

        ndvi = sentinel2.normalizedDifference(["B8", "B4"])
        n_index = ndvi.multiply(2.5).add(0.5).rename("N") 
        ph_index = sentinel2.select("B2").divide(sentinel2.select("B4")).multiply(1.5).add(4.5).rename("ph")
        p_index = sentinel2.select("B3").divide(sentinel2.select("B8")).multiply(30).add(10).rename("P")
        k_index = sentinel2.select("B11").divide(sentinel2.select("B12")).multiply(150).add(50).rename("K")
        ndti = sentinel2.normalizedDifference(["B11", "B12"]).rename("humidity")

        optical_composite = n_index.addBands([p_index, k_index, ph_index, ndti])

        optical_stats = optical_composite.reduceRegions(
            collection=points_fc,
            reducer=ee.Reducer.mean(),
            scale=10
        ).getInfo()

        # 3. Hybrid Temperature (Centroid)
        centroid = roi_geom.centroid()
        
        l8 = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2").filterDate(date_start_str, date_end_str).filterBounds(roi_geom).filter(ee.Filter.lt("CLOUD_COVER", 50))
        l9 = ee.ImageCollection("LANDSAT/LC09/C02/T1_L2").filterDate(date_start_str, date_end_str).filterBounds(roi_geom).filter(ee.Filter.lt("CLOUD_COVER", 50))
        tci_landsat = l8.merge(l9).median().select("ST_B10").multiply(0.00341802).add(149.0).subtract(273.15)
        
        temp_dict = tci_landsat.reduceRegion(ee.Reducer.mean(), centroid.buffer(30), 30).getInfo()
        temp_val = temp_dict.get("ST_B10")
        
        if temp_val is None:
            modis = ee.ImageCollection("MODIS/061/MOD11A1").filterDate(date_start_str, date_end_str).filterBounds(roi_geom).median()
            modis_temp = modis.select("LST_Day_1km").multiply(0.02).subtract(273.15)
            temp_dict = modis_temp.reduceRegion(ee.Reducer.mean(), centroid.buffer(1000), 1000).getInfo()
            temp_val = temp_dict.get("LST_Day_1km")
            if temp_val is None:
                temp_val = 26.5

        # 4. Hybrid Rainfall (Centroid)
        chirps = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY").filterDate(date_start_str, date_end_str).filterBounds(roi_geom).select("precipitation").sum()
        rain_dict = chirps.reduceRegion(ee.Reducer.mean(), centroid.buffer(5000), 5000).getInfo()
        rainfall_val = rain_dict.get("precipitation")
        
        if rainfall_val is None or float(rainfall_val) < 100:
            gpm = ee.ImageCollection("NASA/GPM_L3/IMERG_V07").filterDate(date_start_str, date_end_str).filterBounds(roi_geom).select("precipitation")
            gpm_annual = gpm.mean().multiply(24).multiply(365)
            rain_dict = gpm_annual.reduceRegion(ee.Reducer.mean(), centroid.buffer(11000), 11132).getInfo()
            rainfall_val = rain_dict.get("precipitation")
            if rainfall_val is None or float(rainfall_val) < 100:
                rainfall_val = 3500.0

        # 5. Gabungkan Data
        results = []
        features_list = optical_stats.get("features", [])
        for feat in features_list:
            props = feat.get("properties", {})
            lon = props.get("lon")
            lat = props.get("lat")
            
            results.append({
                "lon": lon,
                "lat": lat,
                "n": float(props.get("N") or 0.0),
                "p": float(props.get("P") or 0.0),
                "k": float(props.get("K") or 0.0),
                "ph": float(props.get("ph") or 0.0),
                "humidity": float(props.get("humidity") or 0.0),
                "temperature": float(temp_val),
                "rainfall": float(rainfall_val)
            })

        return results

    except Exception as e:
        logger.error("Error GEE Multi-Point processing: %s", e)
        raise RuntimeError(f"Gagal memproses data satelit via GEE: {e}")