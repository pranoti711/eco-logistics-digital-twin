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
RUN_M3 = True  # state x calendar-month effects: slower, uses more memory

orders = pd.read_sql("""
    SELECT f.order_id, f.customer_state AS state, f.customer_region AS region, f.seller_region,
f.distance_km, f.transit_days, f.delay_days, f.purchase_date,
o.order_carrier_ts::date AS carrier_date, COALESCE(p.category,'unknown') AS category
    FROM warehouse.fact_delivery f
    JOIN clean.orders o ON o.order_id = f.order_id
    LEFT JOIN clean.products p ON p.product_id = f.product_id
    WHERE f.in_window AND NOT f.ts_anomaly AND NOT f.is_multi_seller AND NOT f.is_outlier_60d
AND f.handling_days >= 0 AND f.transit_days >= 0 AND f.distance_km IS NOT NULL""",
    engine, parse_dates=["purchase_date", "carrier_date"])
wx = pd.read_sql("SELECT state, date, precip_mm FROM external.weather_daily",
engine, parse_dates=["date"])

# ---- rainfall windows per state ----
wxp = wx.pivot(index="date", columns="state", values="precip_mm").sort_index()
roll = wxp.rolling(7).sum()
fwd7 = roll.shift(-6)  # days t .. t+6  after carrier pickup  (treatment)
fut7 = roll.shift(-27)  # days t+21 .. t+27 (placebo: after nearly all deliveries)
pre7 = roll.shift(1)  # days t-7 .. t-1 before purchase (placebo)


def long(frame, name):
    s = frame.stack(dropna=False).rename(name).reset_index()
    s.columns = ["date", "state", name]
    return s


orders = orders.merge(long(fwd7, "rain7"), left_on=["state", "carrier_date"],
right_on=["state", "date"], how="left").drop(columns="date")
orders = orders.merge(long(fut7, "rain_future"), left_on=["state", "carrier_date"],
right_on=["state", "date"], how="left").drop(columns="date")
orders = orders.merge(long(pre7, "rain_pre"), left_on=["state", "purchase_date"],
right_on=["state", "date"], how="left").drop(columns="date")

thr = pd.DataFrame({k: fr.loc["2017-01-01":"2018-09-30"].quantile(0.9)
                    for k, fr in {"t7": fwd7, "tf": fut7, "tp": pre7}.items()})
orders = orders.join(thr, on="state").dropna(subset=["rain7", "rain_future", "rain_pre"])
orders["heavy"] = (orders.rain7 >= orders.t7).astype(int)
orders["heavy_future"] = (orders.rain_future >= orders.tf).astype(int)
orders["heavy_pre"] = (orders.rain_pre >= orders.tp).astype(int)
orders["rain7_50"] = orders.rain7 / 50.0
orders["dist100"] = orders.distance_km / 100.0
orders["week"] = orders.carrier_date.dt.to_period("W").astype(str)
orders["state_month"] = orders.state + "_" + orders.carrier_date.dt.month.astype(str)
top = orders.category.value_counts().index[:25]
orders["cat_"] = np.where(orders.category.isin(top), orders.category, "other")
print(f"Orders in analysis: {len(orders):,}   treated: {orders.heavy.mean():.1%}")


def fit(formula, data, cluster="state"):
    groups = pd.factorize(data[cluster])[0]
    return smf.ols(formula, data=data).fit(cov_type="cluster", cov_kwds={"groups": groups}, use_t=True)


def rows(spec, m, terms):
    ci = m.conf_int()
    return [dict(spec=spec, term=t, estimate=m.params[t], se=m.bse[t], ci_low=ci.loc[t, 0],
ci_high=ci.loc[t, 1], p=m.pvalues[t], n=int(m.nobs)) for t in terms]


CTRL = "dist100 + C(cat_) + C(seller_region)"
FE = "C(state) + C(week)"
out = []
out += rows("M0 naive", fit("transit_days ~ heavy", orders), ["heavy"])
out += rows("M1 + distance, category, seller region", fit(f"transit_days ~ heavy + {CTRL}", orders), ["heavy"])
m2 = fit(f"transit_days ~ heavy + {CTRL} + {FE}", orders)
out += rows("M2 + state and week FE (MAIN)", m2, ["heavy"])
out += rows("M2, cluster state x week", fit(f"transit_days ~ heavy + {CTRL} + {FE}", orders.assign(
    sw=orders.state + "_" + orders.week), cluster="sw"), ["heavy"])
if RUN_M3:
    out += rows("M3 + state x month FE", fit(f"transit_days ~ heavy + {CTRL} + {FE} + C(state_month)", orders), ["heavy"])
out += rows("Placebo: future / pre-purchase rain",
            fit(f"transit_days ~ heavy + heavy_future + heavy_pre + {CTRL} + {FE}", orders),
            ["heavy", "heavy_future", "heavy_pre"])
out += rows("Dose-response (per 50 mm)", fit(f"transit_days ~ rain7_50 + {CTRL} + {FE}", orders), ["rain7_50"])
out += rows("Alt outcome: delay vs estimate", fit(f"delay_days ~ heavy + {CTRL} + {FE}", orders), ["heavy"])
for reg, sub in orders.groupby("region"):
    if sub.heavy.sum() >= 50:
        out += rows(f"M2 region = {reg}", fit(f"transit_days ~ heavy + {CTRL} + {FE}", sub), ["heavy"])

res = pd.DataFrame(out)
pd.set_option("display.width", 220)
print("\nEFFECT OF HEAVY RAIN ON DAYS (coefficient = extra days vs. same state/week without heavy rain)")
print(res.round(3).to_string(index=False))

with engine.begin() as conn:
    conn.execute(text("CREATE SCHEMA IF NOT EXISTS causal"))
res.to_sql("rain_effect_results", engine, schema="causal", if_exists="replace", index=False)
dest = Path(__file__).resolve().parents[2] / "research" / "results"
dest.mkdir(parents=True, exist_ok=True)
res.to_csv(dest / "rain_effect_results.csv", index=False)
print("\nSaved causal.rain_effect_results and research/results/rain_effect_results.csv")
