import os
import sys
import logging
from datetime import datetime
import calendar
import ee

# Agar bisa import app modules dari root folder
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.db.database import supabase as db
from app.services.gee_service import LAHAN_HIBISC_COORDS, _init_gee, _gee_ready
from shapely.geometry import Polygon, Point
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def seed_historical_monthly():
    if not _gee_ready:
        logger.error("GEE tidak siap. Pastikan gee-key.json tersedia.")
        return

    # Bikin 10 points grid
    poly = Polygon(LAHAN_HIBISC_COORDS)
    min_lon, min_lat, max_lon, max_lat = poly.bounds
    roi_geom = ee.Geometry.Polygon([LAHAN_HIBISC_COORDS])
    
    lons = np.linspace(min_lon, max_lon, 6)
    lats = np.linspace(min_lat, max_lat, 6)
    valid_points = [[x, y] for x in lons for y in lats if poly.contains(Point(x, y))]
    
    if len(valid_points) > 10:
        step = len(valid_points) / 10.0
        selected_coords = [valid_points[int(i * step)] for i in range(10)]
    else:
        selected_coords = valid_points

    features = [ee.Feature(ee.Geometry.Point(coord).buffer(10)) for coord in selected_coords]
    points_fc = ee.FeatureCollection(features)
    
    # Ambil lahan_id pertama di database, atau buat dummy = 1
    # Asumsikan lahan Hibisc punya ID 1
    lahan_res = db.table("lahan").select("id").limit(1).execute()
    if not lahan_res.data:
        logger.error("Tidak ada data lahan di tabel 'lahan'. Silakan buat lahan dulu dari UI.")
        return
    lahan_id = lahan_res.data[0]["id"]
    
    logger.info(f"Targeting lahan_id = {lahan_id} for historical seeding.")

    # Loop dari Jan 2024 sampai Jan 2026
    months_to_fetch = []
    for y in [2024, 2025]:
        for m in range(1, 13):
            months_to_fetch.append((y, m))
    months_to_fetch.append((2026, 1))

    for year, month in months_to_fetch:
        start_date = f"{year}-{month:02d}-01"
        last_day = calendar.monthrange(year, month)[1]
        end_date = f"{year}-{month:02d}-{last_day}"
        
        logger.info(f"Mengambil data {start_date} hingga {end_date}...")
        
        try:
            landsat = (
                ee.ImageCollection("LANDSAT/LC08/C02/T1_TOA")
                .filterDate(start_date, end_date)
                .filterBounds(roi_geom)
                # Jangan filter cloud terlalu ketat buat history bulanan, kalau kosong nanti error
                .median()
            )
            
            landsat_l2 = (
                ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
                .filterDate(start_date, end_date)
                .filterBounds(roi_geom)
                .median()
            )
            
            # Jika koleksi kosong, fungsi di bawah akan mereturn null / error.
            n_index = landsat.select("B3").divide(landsat.select("B4")).log10().multiply(10).rename("N")
            p_index = landsat.select("B4").divide(landsat.select("B5")).multiply(8).rename("P")
            k_index = landsat.select("B6").multiply(500).rename("K")
            ph_index = landsat.select("B4").divide(landsat.select("B2")).multiply(2).add(6).rename("ph")
            
            # Suhu L2 -> Celsius
            tci = landsat_l2.select("ST_B10").multiply(0.00341802).add(149.0).subtract(273.15).rename("temp")
            
            ndti = landsat.normalizedDifference(["B6", "B7"]).rename("humidity")

            chirps = (
                ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")
                .filterDate(start_date, end_date)
                .filterBounds(roi_geom)
                .select("precipitation").sum().rename("rainfall")
            )

            composite = n_index.addBands([p_index, k_index, ph_index, tci, ndti, chirps])

            stats_collection = composite.reduceRegions(
                collection=points_fc,
                reducer=ee.Reducer.mean(),
                scale=30
            ).getInfo()

            features_data = stats_collection.get("features", [])
            avg_stats = {"N": 0.0, "P": 0.0, "K": 0.0, "ph": 0.0, "temp": 0.0, "humidity": 0.0, "rainfall": 0.0}
            valid_count = 0
            
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
                    
                # Format extracted_at supaya terbaca sebagai tanggal tersebut di frontend
                # Pastikan timestamp valid format ISO
                extracted_at = f"{year}-{month:02d}-15T12:00:00Z"
                
                payload = {
                    "lahan_id": lahan_id,
                    "longitude": selected_coords[0][0], 
                    "latitude": selected_coords[0][1],
                    "n_value": float(avg_stats["N"]),
                    "p_value": float(avg_stats["P"]),
                    "k_value": float(avg_stats["K"]),
                    "ph": float(avg_stats["ph"]),
                    "temperature": float(avg_stats["temp"]),
                    "humidity": float(avg_stats["humidity"]),
                    "rainfall": float(avg_stats["rainfall"]),
                    "recommendation": "Historical Data",
                    "extracted_at": extracted_at
                }
                
                db.table("satellite_results").insert(payload).execute()
                logger.info(f"✅ Inserted history for {year}-{month:02d} (Valid points: {valid_count}/10)")
            else:
                logger.warning(f"⚠️ No valid satellite imagery for {year}-{month:02d} (Tertutup awan/Missing)")

        except Exception as e:
            logger.error(f"❌ Error at {year}-{month:02d}: {e}")

if __name__ == "__main__":
    seed_historical_monthly()
