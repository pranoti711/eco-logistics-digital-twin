from pathlib import Path
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.engine import URL

USER, PASSWORD, PORT = "postgres", "IASTOBE7", 5432
HOST, DB = "127.0.0.1", "eco_logistics_twin"
engine = create_engine(URL.create("postgresql+psycopg2", username=USER, password=PASSWORD,
                                  host=HOST, port=PORT, database=DB))

BASE = """
WITH dem AS (
  SELECT f.category, f.region, avg(f.units) AS d_wk
  FROM warehouse.fact_demand_weekly f
  JOIN warehouse.demand_segments s ON s.category=f.category AND s.region=f.region
  WHERE s.forecast_ready AND f.week_start >= DATE '2018-07-02' GROUP BY 1,2),
err AS (SELECT category, region, sqrt(avg(power(actual-forecast,2))) AS sigma_wk
        FROM forecast.backtest_results WHERE model='ma8' GROUP BY 1,2),
pair AS (
  SELECT seller_region AS hub, customer_region AS region,
         avg(transit_days)/7.0 AS l_wk, stddev_samp(transit_days)/7.0 AS sd_l_wk,
         sum(freight_value)/sum(n_items) AS freight_unit
  FROM warehouse.fact_delivery
  WHERE in_window AND NOT ts_anomaly AND NOT is_multi_seller AND NOT is_outlier_60d
    AND handling_days>=0 AND transit_days>=0
  GROUP BY 1,2 HAVING count(*)>=30),
px AS (SELECT COALESCE(p.category,'unknown') AS category, sum(i.price)/count(*) AS unit_price
       FROM clean.order_items i LEFT JOIN clean.products p ON p.product_id=i.product_id GROUP BY 1)
"""
series = pd.read_sql(BASE + """
SELECT d.category, d.region, p.hub, d.d_wk, e.sigma_wk, p.l_wk, p.sd_l_wk,
       p.freight_unit, px.unit_price
FROM dem d JOIN err e USING(category,region)
JOIN pair p ON p.region=d.region JOIN px USING(category)""", engine)
fin = pd.read_sql(BASE + "SELECT region AS hub, freight_unit AS f_in FROM pair WHERE hub='Southeast'", engine)

out = Path(__file__).resolve().parents[2] / "data" / "processed"
out.mkdir(parents=True, exist_ok=True)
series.to_csv(out / "sim_series.csv", index=False)
fin.to_csv(out / "sim_fin.csv", index=False)
print("Saved", len(series), "series rows and", len(fin), "inbound-freight rows to", out)
