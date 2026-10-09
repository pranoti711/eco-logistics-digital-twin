import warnings
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing

warnings.filterwarnings("ignore")

USER, PASSWORD, PORT = "postgres", "IASTOBE7", 5432  # CHANGE password
HOST, DB = "localhost", "eco_logistics_twin"
from sqlalchemy.engine import URL
engine = create_engine(URL.create("postgresql+psycopg2", username=USER, password=PASSWORD, host=HOST, port=PORT, database=DB))
H = 4  # forecast horizon, weeks
MIN_TRAIN = 52  # first training window, weeks
Z90 = 1.645  # 90% interval

df = pd.read_sql("""
    SELECT d.category, d.region, d.week_start, d.units
    FROM warehouse.fact_demand_weekly d
    JOIN warehouse.demand_segments s ON s.category = d.category AND s.region = d.region
    WHERE s.forecast_ready
    ORDER BY 1, 2, 3""", engine, parse_dates=["week_start"])


# ---- models: each returns (forecast for H weeks, in-sample residuals) ----
def naive(y, h):
    return np.repeat(y[-1], h), np.diff(y)


def ma(k):

    def f(y, h):
        resid = np.array([y[t] - y[t - k:t].mean() for t in range(k, len(y))])
        return np.repeat(y[-k:].mean(), h), resid

    return f


def ses(y, h):
    fit = SimpleExpSmoothing(y, initialization_method="estimated").fit()
    return fit.forecast(h), np.asarray(fit.resid)


def holt_damped(y, h):
    fit = ExponentialSmoothing(y, trend="add", damped_trend=True,
initialization_method="estimated").fit()
    return fit.forecast(h), np.asarray(fit.resid)


MODELS = {"naive": naive, "ma4": ma(4), "ma8": ma(8), "ses": ses, "holt_damped": holt_damped}


def run(model, y, h):
    fc, resid = model(y, h)
    fc = np.maximum(np.asarray(fc, dtype=float), 0)
    sigma = np.std(resid[-26:]) if len(resid) > 5 else np.std(y)
    steps = np.arange(1, h + 1)
    half = Z90 * sigma * np.sqrt(steps)
    return fc, np.maximum(fc - half, 0), fc + half


# ---- rolling-origin backtest ----
rows = []
for (cat, reg), g in df.groupby(["category", "region"]):
    y = g["units"].to_numpy(dtype=float)
    weeks = g["week_start"].to_numpy()
    for origin in range(MIN_TRAIN, len(y) - H + 1, H):
        train, test = y[:origin], y[origin:origin + H]
        scale = np.mean(np.abs(np.diff(train)))
        if scale == 0:
            continue
        for name, fn in MODELS.items():
            fc, lo, hi = run(fn, train, H)
            for i in range(H):
                rows.append((cat, reg, name, pd.Timestamp(weeks[origin]), i + 1,
test[i], fc[i], lo[i], hi[i], scale))

bt = pd.DataFrame(rows, columns=["category", "region", "model", "origin_week", "h",
"actual", "forecast", "lo90", "hi90", "scale"])
bt["abs_err"] = (bt.actual - bt.forecast).abs()
bt["ase"] = bt.abs_err / bt.scale
bt["covered"] = (bt.actual >= bt.lo90) & (bt.actual <= bt.hi90)

summary = bt.groupby("model").agg(
    WAPE=("abs_err", "sum"), actual=("actual", "sum"),
    MASE=("ase", "mean"), coverage90=("covered", "mean")).reset_index()
summary["WAPE"] = summary["WAPE"] / summary["actual"]
bias = bt.groupby("model").apply(lambda d: (d.forecast.sum() - d.actual.sum()) / d.actual.sum())
summary["bias"] = summary["model"].map(bias)
summary = summary.drop(columns="actual").sort_values("WAPE").round(3)
print("\nBACKTEST SUMMARY (lower WAPE and MASE is better; coverage90 should be near 0.90)")
print(summary.to_string(index=False))

best = summary.iloc[0]["model"]
print(f"\nBest model by WAPE: {best}")

# ---- final forecast: refit on all data, next H weeks, global best model ----
last_week = df["week_start"].max()
future = pd.date_range(last_week + pd.Timedelta(days=7), periods=H, freq="7D")
out = []
for (cat, reg), g in df.groupby(["category", "region"]):
    y = g["units"].to_numpy(dtype=float)
    fc, lo, hi = run(MODELS[best], y, H)
    for i in range(H):
        out.append((cat, reg, best, future[i], i + 1, fc[i], lo[i], hi[i]))
fut = pd.DataFrame(out, columns=["category", "region", "model", "week_start", "h",
"forecast", "lo90", "hi90"])

with engine.begin() as conn:
    conn.execute(text("CREATE SCHEMA IF NOT EXISTS forecast"))
bt.to_sql("backtest_results", engine, schema="forecast", if_exists="replace", index=False)
summary.to_sql("model_summary", engine, schema="forecast", if_exists="replace", index=False)
fut.to_sql("next_4_weeks", engine, schema="forecast", if_exists="replace", index=False)
print("\nSaved to schema forecast: backtest_results, model_summary, next_4_weeks")
