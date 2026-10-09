import sys
from dataclasses import replace
from pathlib import Path
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "python" / "simulation"))
import core

st.set_page_config(page_title="Eco-Logistics Digital Twin", layout="wide")

@st.cache_data
def load():
    return core.load_inputs(ROOT / "data/processed/sim_series.csv", ROOT / "data/processed/sim_fin.csv")

series, fin = load()

# ---------------- sidebar ----------------
st.sidebar.header("Scenario")
sl = st.sidebar.select_slider("Target service level", options=[0.95, 0.97, 0.98, 0.99],
                              value=0.95, format_func=lambda x: f"{x:.0%}")
demand = st.sidebar.slider("Demand change (%)", -30, 50, 0, 5)
lead = st.sidebar.slider("Extra lead time (days)", 0, 7, 0)
disruption = st.sidebar.selectbox("Delivery disruption", [
    "None", "Moderate (half the Feb-Mar 2018 episode)", "Severe (Feb-Mar 2018 level)"])
LEAD = {"None": (1.0, 1.0), "Moderate (half the Feb-Mar 2018 episode)": (1.225, 1.22),
        "Severe (Feb-Mar 2018 level)": (1.45, 1.44)}[disruption]

st.sidebar.header("Assumptions (not in the data)")
hub_fixed = st.sidebar.number_input("Fixed cost per extra hub (BRL/year)", 0, 200_000, 10_000, 5_000)
bulk = st.sidebar.slider("Bulk inbound cost (% of parcel freight)", 0, 50, 10, 5) / 100
hold = st.sidebar.slider("Annual holding cost (% of stock value)", 10, 40, 25, 1) / 100

st.sidebar.header("Display")
show_inr = st.sidebar.checkbox("Show in INR")
rate = st.sidebar.number_input("INR per 1 BRL (your chosen rate)", 1.0, 50.0, 16.0, 0.5)

def money(x):
    return f"₹{x * rate:,.0f}" if show_inr else f"R${x:,.0f}"

# ---------------- run model ----------------
p = core.Params(service_level=sl, demand_mult=1 + demand / 100, lead_add_days=lead,
                lead_mean_mult=LEAD[0], lead_sd_mult=LEAD[1],
                hold_rate=hold, bulk_inbound=bulk, hub_fixed=hub_fixed)
res = core.run(series, fin, p)
base = core.run(series, fin, core.Params(hold_rate=hold, bulk_inbound=bulk, hub_fixed=hub_fixed))

# ---------------- header & KPIs ----------------
st.title("Eco-Logistics Digital Twin")
st.caption("Decision support for hub placement and inventory, built on the Olist Brazilian "
           "e-commerce dataset (top 45 category-by-region series, about 78% of units). "
           "Change the scenario on the left; the model re-solves instantly.")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Hubs to open", ", ".join(res["hubs"]) or "None (ship from Southeast)")
c2.metric("Total annual cost", money(res["total_cost"]),
          f"{(res['total_cost'] / base['total_cost'] - 1) * 100:+.1f}% vs baseline", delta_color="inverse")
c3.metric("Saving vs. shipping all from Southeast", money(res["saving"]))
c4.metric("Avg delivery transit", f"{res['transit']:.1f} days",
          f"{res['transit'] - res['sq_transit']:+.1f} vs no hub", delta_color="inverse")

d1, d2, d3, d4 = st.columns(4)
d1.metric("Recommended inventory", f"{res['units']:,.0f} units",
          f"{(res['units'] / base['units'] - 1) * 100:+.1f}% vs baseline", delta_color="inverse")
d2.metric("Inventory value", money(res["value"]))
d3.metric("Safety stock", f"{res['safety_stock']:,.0f} units",
          f"{(res['safety_stock'] / base['safety_stock'] - 1) * 100:+.1f}% vs baseline", delta_color="inverse")
d4.metric("Stockout risk per replenishment cycle", f"{(1 - sl) * 100:.0f}%")

# ---------------- tabs ----------------
t1, t2, t3, t4 = st.tabs(["By region", "Cost breakdown", "Service-level trade-off", "Assumptions & limits"])

with t1:
    d = res["detail"].copy()
    d.columns = ["Region", "Served from", "Demand / week", "Safety stock", "Inventory units",
                 "Inventory value (BRL)", "Holding", "Transport", "Inbound", "Total cost (BRL)",
                 "Transit days"]
    st.dataframe(d.round(1), use_container_width=True, hide_index=True)

with t2:
    k = rate if show_inr else 1.0
    cb = pd.DataFrame({
        "Holding": [res["sq_holding"], res["holding"]],
        "Transport": [res["sq_transport"], res["transport"]],
        "Bulk inbound": [0, res["inbound"]],
        "Hub fixed": [0, res["hub_fixed"]]},
        index=["No hub (all from Southeast)", "Recommended network"]) * k
    st.bar_chart(cb)
    st.caption("Currency: " + ("INR" if show_inr else "BRL") +
               ". Transport is the dominant cost, so demand moves total cost far more than inventory settings.")

with t3:
    rows = []
    for s in [0.95, 0.97, 0.98, 0.99]:
        r = core.run(series, fin, replace(p, service_level=s))
        rows.append({"Service level": f"{s:.0%}", "pts": s * 100,
                     "Safety stock (units)": r["safety_stock"],
                     "Inventory value (BRL)": r["value"], "Total annual cost (BRL)": r["total_cost"]})
    tr = pd.DataFrame(rows)
    tr["Extra cost per +1 pt (BRL)"] = tr["Total annual cost (BRL)"].diff() / tr["pts"].diff()
    st.line_chart(tr.set_index("Service level")[["Safety stock (units)"]])
    st.dataframe(tr.drop(columns="pts").round(0), use_container_width=True, hide_index=True)
    st.caption("Higher service needs more safety stock at rising marginal cost. "
               "Shortage costs are not modelled, so this shows affordability, not the optimal level.")

with t4:
    st.markdown("""
**What is observed (Olist data):** demand, delivery and transit times, freight charged, prices, seller and customer regions.

**What is assumed (sidebar):** hub fixed cost, bulk inbound cost, holding rate, and the INR exchange rate.

**Modelling choices and limits**
- A hub is a *region*; stock held in region H serves region R with the transit time and freight that Olist sellers in H actually had into R.
- Freight is what the customer was charged, a proxy for carrier cost.
- Demand forecast error scales with demand in the demand scenarios, so safety stock may be overstated.
- The disruption setting uses the Feb-Mar 2018 slowdown (mean transit x1.45, spread x1.44). Its cause has not been established.
- Safety stock uses a normal approximation; real lead times are right-skewed.
- No North hub: Olist has no seller activity there. No warehouse capacity limits, since the data has none.
- "Stockout risk" is the probability of a stockout during one replenishment cycle (1 minus the service level).
""")