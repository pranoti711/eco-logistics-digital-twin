import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from sqlalchemy import create_engine
from sqlalchemy.engine import URL

warnings.filterwarnings("ignore")

USER, PASSWORD, PORT = "postgres", "IASTOBE7", 5432  # same as 01_backtest.py
HOST, DB = "127.0.0.1", "eco_logistics_twin"
engine = create_engine(URL.create("postgresql+psycopg2", username=USER, password=PASSWORD, host=HOST, port=PORT, database=DB))

H, MIN_TRAIN, MIN_T = 4, 52, 13

df = pd.read_sql("""
    SELECT d.category, d.region, d.week_start, d.units
    FROM warehouse.fact_demand_weekly d
    JOIN warehouse.demand_segments s ON s.category = d.category AND s.region = d.region
    WHERE s.forecast_ready ORDER BY 1,2,3""", engine, parse_dates=["week_start"])

cal = pd.read_sql("""
    SELECT week_start, sum(is_holiday::int) AS holidays, bool_or(is_black_friday)::int AS bf
    FROM warehouse.dim_date GROUP BY week_start""", engine, parse_dates=["week_start"])
hol = dict(zip(cal.week_start, cal.holidays))
bf = dict(zip(cal.week_start, cal.bf))


def features(y, t):
    """features using only y[:t]"""
    f = {f"lag{k}": y[t - k] for k in range(1, 9)}
    f["m4"], f["m8"], f["m13"] = y[t - 4:t].mean(), y[t - 8:t].mean(), y[t - 13:t].mean()
    f["sd8"] = y[t - 8:t].std()
    return f


rows = []
for (cat, reg), g in df.groupby(["category", "region"]):
    y = g["units"].to_numpy(dtype=float)
    wk = g["week_start"].tolist()
    for t in range(MIN_T, len(y)):
        base = features(y, t)
        for h in range(1, H + 1):
            if t + h - 1 >= len(y):
                continue
            r = dict(base)
            r.update(category=cat, region=reg, t=t, h=h,
holidays=hol.get(wk[t + h - 1], 0), bf=bf.get(wk[t + h - 1], 0),
target=y[t + h - 1], origin_week=wk[t])
            rows.append(r)

data = pd.DataFrame(rows)
data["category"] = data["category"].astype("category")
data["region"] = data["region"].astype("category")
FEATS = [c for c in data.columns if c not in ("t", "target", "origin_week")]

params = dict(objective="poisson", n_estimators=300, learning_rate=0.03, num_leaves=15,
min_child_samples=20, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
verbose=-1, random_state=42)

n_weeks = df.groupby(["category", "region"]).size().iloc[0]
out = []
for origin in range(MIN_TRAIN, n_weeks - H + 1, H):
    train = data[data["t"] + data["h"] <= origin]  # targets strictly before origin
    test = data[data["t"] == origin]
    model = lgb.LGBMRegressor(**params).fit(train[FEATS], train["target"])
    test = test.assign(forecast=np.maximum(model.predict(test[FEATS]), 0))
    out.append(test[["category", "region", "origin_week", "h", "target", "forecast"]])

res = pd.concat(out).rename(columns={"target": "actual"})
res["category"] = res["category"].astype(str)
res["region"] = res["region"].astype(str)
res["model"] = "lgbm"
res["abs_err"] = (res.actual - res.forecast).abs()

# compare with the models already in the database, on identical rows
prev = pd.read_sql("SELECT category, region, model, origin_week, h, actual, forecast, abs_err, scale "
"FROM forecast.backtest_results WHERE model IN ('naive','ma8')",
engine, parse_dates=["origin_week"])
scale = prev.drop_duplicates(["category", "region", "origin_week"])[["category", "region", "origin_week", "scale"]]
res = res.merge(scale, on=["category", "region", "origin_week"], how="inner")
res["ase"] = res.abs_err / res.scale
allm = pd.concat([prev.assign(ase=prev.abs_err / prev.scale), res], ignore_index=True)


def summarize(d):
    return pd.Series({"WAPE": d.abs_err.sum() / d.actual.sum(), "MASE": d.ase.mean(),
"bias": (d.forecast.sum() - d.actual.sum()) / d.actual.sum()})


print("\nOVERALL")
print(allm.groupby("model").apply(summarize).round(3).sort_values("WAPE"))

folds = allm.groupby(["origin_week", "model"]).apply(lambda d: d.abs_err.sum() / d.actual.sum()).unstack()
print("\nWAPE BY FOLD (does lgbm beat ma8 in most folds?)")
print(folds.round(3))
print("\nFolds where lgbm < ma8:", int((folds["lgbm"] < folds["ma8"]).sum()), "of", len(folds))

res.to_sql("backtest_lgbm", engine, schema="forecast", if_exists="replace", index=False)
print("\nSaved forecast.backtest_lgbm")
