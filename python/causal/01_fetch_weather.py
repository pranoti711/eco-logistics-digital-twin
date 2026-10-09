import time
from pathlib import Path
import pandas as pd
import requests
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL

USER, PASSWORD, PORT = "postgres", "IASTOBE7", 5432
HOST, DB = "127.0.0.1", "eco_logistics_twin"
engine = create_engine(URL.create("postgresql+psycopg2", username=USER, password=PASSWORD,
host=HOST, port=PORT, database=DB))

pts = pd.read_sql("""
    SELECT c.state,
           percentile_cont(0.5) WITHIN GROUP (ORDER BY g.lat) AS lat,
           
           percentile_cont(0.5) WITHIN GROUP (ORDER BY g.lng) AS lng,
           count(*) AS customers
    FROM clean.customers c JOIN clean.geo_zip g ON g.zip_prefix = c.zip_prefix
    GROUP BY c.state ORDER BY 1""", engine)
print(pts.round(2).to_string(index=False))

API = "https://archive-api.open-meteo.com/v1/archive"
frames = []
for r in pts.itertuples():
    params = dict(latitude=round(r.lat, 3), longitude=round(r.lng, 3),
                  start_date="2017-01-01", end_date="2018-10-31",
                  daily="precipitation_sum,temperature_2m_max,wind_speed_10m_max",
                  timezone="America/Sao_Paulo")
    for attempt in range(4):
        resp = requests.get(API, params=params, timeout=60)
        if resp.status_code == 200:
            break
        print(f"  {r.state}: HTTP {resp.status_code}, retrying...")
        time.sleep(10 * (attempt + 1))
    resp.raise_for_status()
    d = resp.json()["daily"]
    frames.append(pd.DataFrame({
        "state": r.state, "date": pd.to_datetime(d["time"]),
        "precip_mm": d["precipitation_sum"], "temp_max_c": d["temperature_2m_max"],
        "wind_max_kmh": d["wind_speed_10m_max"], "lat": r.lat, "lng": r.lng}))
    print(f"fetched {r.state}")
    time.sleep(2)

wx = pd.concat(frames, ignore_index=True)

out = Path(__file__).resolve().parents[2] / "data" / "external"
out.mkdir(parents=True, exist_ok=True)
wx.to_csv(out / "weather_daily.csv", index=False)

with engine.begin() as conn:
    conn.execute(text("CREATE SCHEMA IF NOT EXISTS external"))
wx.to_sql("weather_daily", engine, schema="external", if_exists="replace", index=False)
print(f"\nSaved {len(wx)} rows to external.weather_daily and data/external/weather_daily.csv")
print("Attribution: weather data by Open-Meteo.com (CC BY 4.0), based on ERA5 reanalysis.")
