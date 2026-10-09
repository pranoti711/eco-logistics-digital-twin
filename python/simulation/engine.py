import numpy as np
import pandas as pd
import pulp
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL

USER, PASSWORD, PORT = "postgres", "IASTOBE7", 5432
HOST, DB = "127.0.0.1", "eco_logistics_twin"
engine = create_engine(URL.create("postgresql+psycopg2", username=USER, password=PASSWORD,
host=HOST, port=PORT, database=DB))

Z = {0.95: 1.6449, 0.97: 1.8808, 0.98: 2.0537, 0.99: 2.3263}
SUPPLY, HOLD, REVIEW = "Southeast", 0.25, 1.0
BULK, FIXED = 0.1, 10_000  # planning-case assumptions: change these two

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
fin = pd.read_sql(BASE + "SELECT region AS hub, freight_unit AS f_in FROM pair WHERE hub='Southeast'",
engine).set_index("hub")["f_in"].to_dict()
hubs, regions = sorted(series.hub.unique()), sorted(series.region.unique())


def build_matrix(sl=0.95, demand_mult=1.0, lead_add_days=0.0, lead_mean_mult=1.0, lead_sd_mult=1.0):
    t = series.copy()
    d = t.d_wk * demand_mult
    sig = t.sigma_wk * demand_mult
    L = t.l_wk * lead_mean_mult + lead_add_days / 7.0
    sdL = t.sd_l_wk * lead_sd_mult
    ss = Z[sl] * np.sqrt((L + REVIEW) * sig ** 2 + d ** 2 * sdL ** 2)
    cycle = d * (L + REVIEW)
    t["d"], t["ss"], t["units"] = d, ss, cycle + ss
    t["value"] = t.unit_price * t.units
    t["holding"] = HOLD * t.unit_price * (ss + cycle / 2)
    t["transport"] = d * 52 * t.freight_unit
    t["inbound"] = [0.0 if h == SUPPLY else BULK * dd * 52 * fin[h] for h, dd in zip(t.hub, d)]
    t["w_transit"] = d * L * 7
    g = t.groupby(["region", "hub"]).agg(ss=("ss", "sum"), units=("units", "sum"), value=("value", "sum"),
holding=("holding", "sum"), transport=("transport", "sum"),
inbound=("inbound", "sum"), d=("d", "sum"),
w_transit=("w_transit", "sum")).reset_index()
    g["cost"] = g.holding + g.transport + g.inbound
    return g


def choose_hubs(g, fixed):
    c = g.set_index(["region", "hub"])["cost"].to_dict()
    safe = lambda s: s.replace("-", "_")
    prob = pulp.LpProblem("hubs", pulp.LpMinimize)
    op = {h: pulp.LpVariable("open_" + safe(h), cat="Binary") for h in hubs}
    asg = {k: pulp.LpVariable("a_%s_%s" % (safe(k[0]), safe(k[1])), cat="Binary") for k in c}
    prob += pulp.lpSum(c[k] * v for k, v in asg.items()) + fixed * pulp.lpSum(op[h] for h in hubs if h != SUPPLY)
    prob += op[SUPPLY] == 1
    for r in regions:
        prob += pulp.lpSum(v for (rr, h), v in asg.items() if rr == r) == 1
    for (r, h), v in asg.items():
        prob += v <= op[h]
    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    return {r: h for (r, h), v in asg.items() if v.value() > 0.5}


def run(name, sl=0.95, **kw):
    g = build_matrix(sl, **kw)
    assign = choose_hubs(g, FIXED)
    pick = g[[assign[r] == h for r, h in zip(g.region, g.hub)]]
    extra = sorted(set(assign.values()) - {SUPPLY})
    fixed_cost = FIXED * len(extra)
    sq = g[g.hub == SUPPLY]
    total = pick.cost.sum() + fixed_cost
    return dict(scenario=name, service_level=sl, hubs_open="+".join(extra) or "none",
                safety_stock_units=round(pick.ss.sum()), inventory_units=round(pick.units.sum()),
                inventory_value_brl=round(pick.value.sum()),
                holding=round(pick.holding.sum()), transport=round(pick.transport.sum()),
                inbound=round(pick.inbound.sum()), hub_fixed=round(fixed_cost),
                total_cost=round(total), status_quo_cost=round(sq.cost.sum()),
                saving=round(sq.cost.sum() - total),
                avg_transit_days=round(pick.w_transit.sum() / pick.d.sum(), 1),
                status_quo_transit_days=round(sq.w_transit.sum() / sq.d.sum(), 1))


SHOCK = dict(lead_mean_mult=1.45, lead_sd_mult=1.44)  # calibrated on Feb-Mar 2018
results = pd.DataFrame([
    run("Baseline"),
    run("A: demand +10%", demand_mult=1.10),
    run("B: demand +25%", demand_mult=1.25),
    run("C: lead time +2 days", lead_add_days=2),
    run("D: service level 99%", sl=0.99),
    run("E: disruption (Feb-Mar 2018 level)", **SHOCK),
    run("F: demand +25% and disruption", demand_mult=1.25, **SHOCK),
])
pd.set_option("display.width", 250)
print(results[["scenario", "hubs_open", "inventory_units", "inventory_value_brl", "total_cost",
               "saving", "avg_transit_days", "status_quo_transit_days"]].to_string(index=False))

with engine.begin() as conn:
    conn.execute(text("CREATE SCHEMA IF NOT EXISTS simulation"))
results.to_sql("scenario_results", engine, schema="simulation", if_exists="replace", index=False)
print("\nSaved simulation.scenario_results")
