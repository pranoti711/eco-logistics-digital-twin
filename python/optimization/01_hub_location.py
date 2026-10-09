import pandas as pd
import pulp
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL

USER, PASSWORD, PORT = "postgres", "IASTOBE7", 5432
HOST, DB = "127.0.0.1", "eco_logistics_twin"
engine = create_engine(URL.create("postgresql+psycopg2", username=USER, password=PASSWORD,
host=HOST, port=PORT, database=DB))

Z = {0.95: 1.6449, 0.97: 1.8808, 0.98: 2.0537, 0.99: 2.3263}
SUPPLY = "Southeast"
HOLD = 0.25  # annual holding rate (assumption)

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
MATRIX = BASE + """
SELECT p.hub, d.region, sum(d.d_wk) AS d_wk,
       sum(d.d_wk*p.l_wk*7)/sum(d.d_wk) AS transit_days,
       sum(%f*px.unit_price*sqrt((p.l_wk+1.0)*power(e.sigma_wk,2)+power(d.d_wk,2)*power(p.sd_l_wk,2))) AS a,
       sum(%f*px.unit_price*d.d_wk*(p.l_wk+1.0)/2) AS b_cycle,
       sum(d.d_wk*52*p.freight_unit) AS transport
FROM dem d JOIN err e USING(category,region) JOIN pair p ON p.region=d.region JOIN px USING(category)
GROUP BY 1,2""" % (HOLD, HOLD)
FIN = BASE + "SELECT region AS hub, freight_unit AS f_in FROM pair WHERE hub='Southeast'"

mat = pd.read_sql(MATRIX, engine)
fin = pd.read_sql(FIN, engine).set_index("hub")["f_in"].to_dict()
hubs, regions = sorted(mat.hub.unique()), sorted(mat.region.unique())


def costs(sl, bulk, mult):
    t = mat.copy()
    t["inbound"] = [0 if h == SUPPLY else bulk * d * 52 * fin[h] for h, d in zip(t.hub, t.d_wk)]
    t["cost"] = mult * (Z[sl] * t.a + t.b_cycle + t.transport + t.inbound)
    return t.set_index(["region", "hub"])["cost"].to_dict()


def solve(sl, bulk, fixed, mult=1.0):
    c = costs(sl, bulk, mult)
    prob = pulp.LpProblem("hubs", pulp.LpMinimize)
    safe = lambda s: s.replace("-", "_")
    op = {h: pulp.LpVariable("open_" + safe(h), cat="Binary") for h in hubs}
    asg = {k: pulp.LpVariable("a_%s_%s" % (safe(k[0]), safe(k[1])), cat="Binary") for k in c}
    prob += pulp.lpSum(c[k] * v for k, v in asg.items()) + \
            fixed * pulp.lpSum(op[h] for h in hubs if h != SUPPLY)
    prob += op[SUPPLY] == 1
    for r in regions:
        prob += pulp.lpSum(v for (rr, h), v in asg.items() if rr == r) == 1
    for (r, h), v in asg.items():
        prob += v <= op[h]
    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    chosen = [h for h in hubs if op[h].value() > 0.5]
    base = sum(c[(r, SUPPLY)] for r in regions)  # status quo: all from Southeast
    return chosen, pulp.value(prob.objective), base


rows = []
for sl in (0.95, 0.99):
    for mult in (1.0, 1.25):
        for bulk in (0.0, 0.1, 0.2, 0.3, 0.5):
            for fixed in (0, 10_000, 25_000, 50_000):
                chosen, total, base = solve(sl, bulk, fixed, mult)
                rows.append(dict(service_level=sl, demand_mult=mult, bulk_inbound=bulk,
                                 hub_fixed_cost=fixed,
                                 hubs_open="+".join(h for h in chosen if h != SUPPLY) or "none",
                                 total_cost=round(total), status_quo_cost=round(base),
                                 saving=round(base - total)))
res = pd.DataFrame(rows)

print("\nHUBS TO OPEN (95% service, current demand): rows = bulk inbound factor, cols = hub fixed cost")
sub = res[(res.service_level == 0.95) & (res.demand_mult == 1.0)]
print(sub.pivot(index="bulk_inbound", columns="hub_fixed_cost", values="hubs_open").to_string())

print("\nBREAK-EVEN FIXED COST PER HUB (annual BRL, 95% service)")
for bulk in (0.0, 0.1, 0.2, 0.3):
    c = costs(0.95, bulk, 1.0)
    be = {h: round(c[(h, SUPPLY)] - c[(h, h)]) for h in hubs if h != SUPPLY}
    print("bulk inbound %.1f:" % bulk, be)

with engine.begin() as conn:
    conn.execute(text("CREATE SCHEMA IF NOT EXISTS optimization"))
res.to_sql("hub_scenarios", engine, schema="optimization", if_exists="replace", index=False)
print("\nSaved optimization.hub_scenarios")
