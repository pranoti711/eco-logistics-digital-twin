from dataclasses import dataclass
import numpy as np
import pandas as pd
import pulp

Z = {0.95: 1.6449, 0.97: 1.8808, 0.98: 2.0537, 0.99: 2.3263}
SUPPLY, REVIEW = "Southeast", 1.0

@dataclass
class Params:
    service_level: float = 0.95
    demand_mult: float = 1.0
    lead_add_days: float = 0.0
    lead_mean_mult: float = 1.0
    lead_sd_mult: float = 1.0
    hold_rate: float = 0.25
    bulk_inbound: float = 0.10
    hub_fixed: float = 10_000

def load_inputs(series_path, fin_path):
    series = pd.read_csv(series_path)
    fin = pd.read_csv(fin_path).set_index("hub")["f_in"].to_dict()
    return series, fin

def build_matrix(series, fin, p):
    t = series.copy()
    d = t.d_wk * p.demand_mult
    sig = t.sigma_wk * p.demand_mult
    L = t.l_wk * p.lead_mean_mult + p.lead_add_days / 7.0
    sdL = t.sd_l_wk * p.lead_sd_mult
    ss = Z[p.service_level] * np.sqrt((L + REVIEW) * sig**2 + d**2 * sdL**2)
    cycle = d * (L + REVIEW)
    t["d"], t["ss"], t["units"] = d, ss, cycle + ss
    t["value"] = t.unit_price * t.units
    t["holding"] = p.hold_rate * t.unit_price * (ss + cycle / 2)
    t["transport"] = d * 52 * t.freight_unit
    t["inbound"] = [0.0 if h == SUPPLY else p.bulk_inbound * dd * 52 * fin[h]
                    for h, dd in zip(t.hub, d)]
    t["w_transit"] = d * L * 7
    g = t.groupby(["region", "hub"]).agg(
        d=("d", "sum"), ss=("ss", "sum"), units=("units", "sum"), value=("value", "sum"),
        holding=("holding", "sum"), transport=("transport", "sum"),
        inbound=("inbound", "sum"), w_transit=("w_transit", "sum")).reset_index()
    g["cost"] = g.holding + g.transport + g.inbound
    return g

def choose_hubs(g, fixed):
    hubs, regions = sorted(g.hub.unique()), sorted(g.region.unique())
    c = g.set_index(["region", "hub"])["cost"].to_dict()
    safe = lambda s: s.replace("-", "_")
    prob = pulp.LpProblem("hubs", pulp.LpMinimize)
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
    return {r: h for (r, h), v in asg.items() if v.value() > 0.5}

def run(series, fin, p):
    g = build_matrix(series, fin, p)
    assign = choose_hubs(g, p.hub_fixed)
    pick = g[[assign[r] == h for r, h in zip(g.region, g.hub)]].copy()
    extra = sorted(set(assign.values()) - {SUPPLY})
    fixed_cost = p.hub_fixed * len(extra)
    sq = g[g.hub == SUPPLY]
    total = pick.cost.sum() + fixed_cost
    pick["transit_days"] = pick.w_transit / pick.d
    return dict(
        hubs=extra, total_cost=total, status_quo_cost=sq.cost.sum(), saving=sq.cost.sum() - total,
        safety_stock=pick.ss.sum(), units=pick.units.sum(), value=pick.value.sum(),
        holding=pick.holding.sum(), transport=pick.transport.sum(),
        inbound=pick.inbound.sum(), hub_fixed=fixed_cost,
        sq_holding=sq.holding.sum(), sq_transport=sq.transport.sum(),
        transit=pick.w_transit.sum() / pick.d.sum(), sq_transit=sq.w_transit.sum() / sq.d.sum(),
        detail=pick[["region", "hub", "d", "ss", "units", "value", "holding",
"transport", "inbound", "cost", "transit_days"]])

if __name__ == "__main__":
    from pathlib import Path
    root = Path(__file__).resolve().parents[2] / "data" / "processed"
    s, f = load_inputs(root / "sim_series.csv", root / "sim_fin.csv")
    r = run(s, f, Params())
    print("Baseline check: hubs =", r["hubs"], "| total =", round(r["total_cost"]),
"| saving =", round(r["saving"]))