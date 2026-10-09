from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL

USER, PASSWORD, PORT = "postgres", "IASTOBE7", 5432
HOST, DB = "127.0.0.1", "eco_logistics_twin"
engine = create_engine(URL.create("postgresql+psycopg2", username=USER, password=PASSWORD,
host=HOST, port=PORT, database=DB))
RUN_M3 = True

orders = pd.read_sql("""
    SELECT f.order_id, f.customer_state AS state, f.seller_region, f.distance_km, f.transit_days,
o.order_carrier_ts::date AS carrier_date, COALESCE(p.category,'unknown') AS category
    FROM warehouse.fact_delivery f
    JOIN clean.orders o ON o.order_id = f.order_id
    LEFT JOIN clean.products p ON p.product_id = f.product_id
    WHERE f.in_window AND NOT f.ts_anomaly AND NOT f.is_multi_seller AND NOT f.is_outlier_60d
AND f.handling_days >= 0 AND f.transit_days >= 0 AND f.distance_km IS NOT NULL""",
    engine, parse_dates=["carrier_date"])
wx = pd.read_sql("SELECT state, date, precip_mm FROM external.weather_daily",
engine, parse_dates=["date"])
roll = wx.pivot(index="date", columns="state", values="precip_mm").sort_index().rolling(7).sum()

# window starting k days after carrier pickup covers days k .. k+6
WINDOWS = {"w_m14":-14, "w_m7":-7, "w0": 0, "w_p7": 7, "w_p14": 14, "w_p21": 21}
LABEL = {"w_m14": "days -14 to -8 (before pickup)", "w_m7": "days -7 to -1 (before pickup)",
"w0": "days 0 to 6 (TREATMENT)", "w_p7": "days 7 to 13", "w_p14": "days 14 to 20",
"w_p21": "days 21 to 27 (placebo)"}
for name, k in WINDOWS.items():
    fr = roll.shift(-(k + 6))
    s = fr.stack(future_stack=True).rename(name + "_rain").reset_index()
    s.columns = ["date", "state", name + "_rain"]
    orders = orders.merge(s, left_on=["state", "carrier_date"], right_on=["state", "date"],
how="left").drop(columns="date")
    thr = fr.loc["2017-01-01":"2018-09-30"].quantile(0.9)
    orders[name] = (orders[name + "_rain"] >= orders.state.map(thr)).astype(int)
orders = orders.dropna(subset=[n + "_rain" for n in WINDOWS])

orders["dist100"] = orders.distance_km / 100.0
orders["week"] = orders.carrier_date.dt.to_period("W").astype(str)
orders["state_month"] = orders.state + "_" + orders.carrier_date.dt.month.astype(str)
top = orders.category.value_counts().index[:25]
orders["cat_"] = np.where(orders.category.isin(top), orders.category, "other")
print(f"Orders in analysis: {len(orders):,}")


def fit(formula):
    g = pd.factorize(orders["state"])[0]
    return smf.ols(formula, data=orders).fit(cov_type="cluster", cov_kwds={"groups": g}, use_t=True)


terms = " + ".join(WINDOWS)
base = f"transit_days ~ {terms} + dist100 + C(cat_) + C(seller_region) + C(state) + C(week)"
specs = {"M2 state + week FE": base}
if RUN_M3:
    specs["M3 + state x month FE"] = base + " + C(state_month)"

rows = []
for spec, f in specs.items():
    m = fit(f)
    ci = m.conf_int()
    for w in WINDOWS:
        rows.append(dict(spec=spec, window=LABEL[w], estimate=m.params[w], se=m.bse[w],
ci_low=ci.loc[w, 0], ci_high=ci.loc[w, 1], p=m.pvalues[w]))
res = pd.DataFrame(rows)
pd.set_option("display.width", 220)
print("\nEXTRA TRANSIT DAYS FROM HEAVY RAIN, BY WINDOW RELATIVE TO CARRIER PICKUP")
print(res.round(3).to_string(index=False))

with engine.begin() as conn:
    conn.execute(text("CREATE SCHEMA IF NOT EXISTS causal"))
res.to_sql("rain_lead_lag", engine, schema="causal", if_exists="replace", index=False)
dest = Path(__file__).resolve().parents[2] / "research" / "results"
dest.mkdir(parents=True, exist_ok=True)
res.to_csv(dest / "rain_lead_lag.csv", index=False)
print("\nSaved causal.rain_lead_lag and research/results/rain_lead_lag.csv")
