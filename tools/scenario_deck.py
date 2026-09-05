"""Build the leadership scenario deck: reports/scenarios.html.

Twelve named situations, each answered by the REAL decision engine
(pricing.dp.solve -- the code that prices the shelf), precomputed over a
grid of shelf states and embedded in one self-contained HTML page. Sliders
snap to grid points; nothing is re-implemented in the browser. The demand
forecast is a slider, not the trained model, so the page shows the DECISION
LOGIC under the config on this machine -- never a claim about a real SKU.
On the repo's fixture every number is a fixture number (AGENTS rule 19).

    python3 -m tools.scenario_deck [--out reports/scenarios.html] [--workers 0] [--quick]
"""

import argparse
import html
import json
import math
import os

import numpy as np

from common.config import load_config, reference_discount
from common.io import read_json
from common.parallel import map_episodes
from pricing import dp as dp_mod
from pricing import explore
from pricing.demand import expected_min_demand_inventory, mu_at

P0 = 10_000.0          # list price; every currency figure scales with it
GRID = {
    "q": [1, 2, 4, 8, 12, 20],                 # units on the shelf
    "h": [2, 4, 8, 12, 24],                    # hours left in the window
    "gamma": [0.3, 0.5, 0.7],                  # cost / list price
    "mu": [0.25, 0.5, 1.0, 2.0, 4.0],          # expected units/hour at the reference
}
QUICK = {"q": [2, 8], "h": [4, 12], "gamma": [0.3, 0.7], "mu": [0.5, 2.0]}
WORLD = {"as_forecast": 1.0, "half": 0.5, "double": 2.0}


def beliefs(cfg):
    """Cold start (the prior on disk, or the design default) and a learned
    posterior: the same shelf under a wide belief and a tighter, steeper one."""
    prior = read_json(cfg["posterior"]["prior"]["path"]) or {}
    per = prior.get("per_category") or {}
    cold = float(np.mean([v["mean"] for v in per.values()])) if per else -1.0
    cold = max(min(cold, -0.3), -2.5)
    return {"cold": round(cold, 3), "learned": round(max(cold * 1.4, -3.0), 3)}


def _solve(cache, key, cfg, d_ref, r):
    if key not in cache:
        gamma, mu, eps, q, h, anchor = key
        cache[key] = dp_mod.solve(P0, P0 * gamma, q, [mu] * h, d_ref, eps, r, cfg,
                                  anchor_discount=anchor, entry=(anchor is None))
    return cache[key]


def simulate(cache, cfg, d_ref, r, gamma, mu, eps_belief, q0, hours,
             world=1.0, restock_at=None, restock_units=0, fixed=None):
    """Hour-by-hour: re-solve from the actual shelf (as production does),
    sell E[min(D, q)] under the WORLD's demand, carry the anchor. `fixed`
    replaces the solver with a schedule (a flat reference or a legacy ramp).
    Returns the path and the scorecard."""
    cost = P0 * gamma
    q, anchor = float(q0), None
    rows, disc_cost = [], 0.0
    for t in range(hours):
        q_int = int(round(q))
        if restock_at is not None and t == restock_at:
            q += restock_units
            q_int = int(round(q))
        if q_int <= 0:
            rows.append({"h": t, "q": 0.0, "d": None, "sold": 0.0})
            continue
        if fixed is not None:
            d = float(fixed[t])
        else:
            res = _solve(cache, (gamma, mu, eps_belief, q_int, hours - t, anchor),
                         cfg, d_ref, r)
            d = float(res.tiers[res.optimal_index])
            anchor = d
        mu_w = mu_at(mu * world, d, d_ref, eps_belief, cfg["pricing"]["demand_floor"])
        sold = min(expected_min_demand_inventory(mu_w, r, q_int, cfg["pricing"]["negbin_max_k"]), q)
        disc_cost += P0 * d * sold
        rows.append({"h": t, "q": round(q, 2), "d": round(d, 3), "sold": round(sold, 2)})
        q -= sold
    left = max(q, 0.0)
    return {"path": rows,
            "score": {"leftover": round(left, 2), "scrap_cost": round(cost * left, 0),
                      "discount_cost": round(disc_cost, 0),
                      "il": round(cost * left + disc_cost, 0),
                      "sold": round(q0 + (restock_units if restock_at is not None else 0) - left, 2)}}


def legacy_ramp(d_ref, hours):
    """An illustrative fixed schedule: shallow, reference, deep by thirds."""
    third = max(hours // 3, 1)
    return [max(d_ref - 0.15, 0.0)] * third + [d_ref] * third \
        + [min(d_ref + 0.15, 0.6)] * (hours - 2 * third)


def one_state(item, cfg):
    """Everything the page needs for one (gamma, mu, belief, q, h) cell."""
    gamma, mu, eps_name, eps, q, h, d_ref, r = item
    cache = {}
    res = _solve(cache, (gamma, mu, eps, q, h, None), cfg, d_ref, r)
    star = res.optimal_index
    tiers_all, d_max = dp_mod.feasible_tiers(P0, P0 * gamma, cfg["pricing"]["tier_step"])
    q_by = {int(j): round(float(v), 1) for j, v in res.q_by_tier.items()}
    out = {
        "key": [gamma, mu, eps_name, q, h],
        "tiers": [round(float(t), 3) for t in res.tiers],
        "d_max": round(float(d_max), 3),
        "q_by_tier": q_by,                 # Q = -expected IL, allowed entry arms only
        "star": int(star),
        "d_ref": d_ref,
        "paths": {
            "dp": simulate(cache, cfg, d_ref, r, gamma, mu, eps, q, h),
            "dp_world_half": simulate(cache, cfg, d_ref, r, gamma, mu, eps, q, h, world=0.5),
            "dp_world_double": simulate(cache, cfg, d_ref, r, gamma, mu, eps, q, h, world=2.0),
            "dp_restock": simulate(cache, cfg, d_ref, r, gamma, mu, eps, q, h,
                                   restock_at=max(h // 2, 1), restock_units=max(q // 2, 1)),
            "flat_reference": simulate(cache, cfg, d_ref, r, gamma, mu, eps, q, h,
                                       fixed=[d_ref] * h),
            "legacy_ramp": simulate(cache, cfg, d_ref, r, gamma, mu, eps, q, h,
                                    fixed=legacy_ramp(d_ref, h)),
        },
    }
    return out


SCENARIOS = [
    {"id": "heavy_time", "title": "Heavy inventory, plenty of time",
     "state": {"q": 20, "h": 24, "gamma": 0.5, "mu": 1.0, "belief": "cold"},
     "ask": "Twenty units, a full day, ordinary demand. Does it panic?",
     "read": "It opens shallow and holds. Every hour it re-solves from the actual shelf; "
             "the deep tiers are available but not worth their margin yet. Compare the "
             "scorecard against the flat-reference and the legacy ramp."},
    {"id": "heavy_late", "title": "Heavy inventory, hours left",
     "state": {"q": 12, "h": 4, "gamma": 0.5, "mu": 1.0, "belief": "cold"},
     "ask": "Twelve units, four hours. Unsold stock at close is scrap at full cost.",
     "read": "It deepens hard and early -- discount cost now against scrap cost later. "
             "The decision curve shows why: the shallow tiers carry the largest expected loss."},
    {"id": "light_late", "title": "Light inventory, hours left",
     "state": {"q": 2, "h": 4, "gamma": 0.5, "mu": 1.0, "belief": "cold"},
     "ask": "Two units, four hours, demand that will clear them anyway.",
     "read": "It barely moves. A discount here only gives margin away; the legacy ramp "
             "keeps deepening on a schedule and pays for it."},
    {"id": "high_cogs", "title": "High-COGS item",
     "state": {"q": 12, "h": 8, "gamma": 0.7, "mu": 1.0, "belief": "cold"},
     "ask": "Cost is 70% of list. The deep tiers would sell below cost.",
     "read": "The cost floor removes them structurally (greyed out on the curve) -- there is "
             "no check to forget. It accepts more scrap risk rather than a below-cost price."},
    {"id": "demand", "title": "Weak vs strong demand, same shelf",
     "state": {"q": 8, "h": 12, "gamma": 0.5, "mu": 0.5, "belief": "cold"},
     "ask": "Move the demand slider from 0.5 to 2 units/hour with everything else fixed.",
     "read": "The forecast, not the inventory count, drives the decision. The same eight "
             "units are a problem at 0.5/hour and a non-event at 2/hour."},
    {"id": "explore", "title": "Exploration: what a forced price costs",
     "state": {"q": 8, "h": 12, "gamma": 0.5, "mu": 1.0, "belief": "cold", "tau": 4000, "dmin": 0.1},
     "ask": "Which alternative prices could be tried today, and what does each cost?",
     "read": "Every tier's cost is the expected IL it gives up against the optimum. The "
             "budget tau admits the cheap ones; delta_min removes tiers too close to the "
             "reference to teach anything. The draw among the rest is uniform -- that is "
             "what makes the outcome evidence."},
    {"id": "legacy", "title": "Legacy ramp vs the system, same shelf",
     "state": {"q": 12, "h": 12, "gamma": 0.5, "mu": 1.0, "belief": "cold"},
     "ask": "Today's fixed schedule against the solver on the identical state.",
     "read": "Like-for-like under the same demand model -- not a claim about a real SKU; "
             "the A/B is the evidence. Read leftover, scrap cost, discount cost, and the sum."},
    {"id": "shock", "title": "Sales come in faster or slower than forecast",
     "state": {"q": 12, "h": 12, "gamma": 0.5, "mu": 1.0, "belief": "cold"},
     "ask": "The forecast is wrong by half, in either direction.",
     "read": "It follows the shelf, not a plan: every hour re-solves from the actual "
             "inventory, so a slow start deepens the path and a fast one holds it."},
    {"id": "restock", "title": "Stock arrives mid-episode after a markdown",
     "state": {"q": 8, "h": 12, "gamma": 0.5, "mu": 1.0, "belief": "cold"},
     "ask": "Half the units again arrive at the midpoint, after the price has already moved.",
     "read": "The price cannot go back up -- monotonicity is structural. It re-solves from "
             "the new shelf at or below the price in force. This is a real limitation, "
             "shown on purpose."},
    {"id": "dead", "title": "Dead stock: near-zero demand",
     "state": {"q": 8, "h": 24, "gamma": 0.5, "mu": 0.25, "belief": "cold"},
     "ask": "A quarter of a unit per hour. Does it chase a sale that will not come?",
     "read": "It deepens to what the demand response can still buy, then stops: below the "
             "cost floor nothing is offered, and the scorecard books the scrap honestly."},
    {"id": "learning", "title": "Day one vs after learning",
     "state": {"q": 12, "h": 12, "gamma": 0.5, "mu": 1.0, "belief": "learned"},
     "ask": "The same shelf under the cold-start prior and under a learned, steeper belief.",
     "read": "Toggle the belief. The decision changes because the system learned how "
             "much a discount actually moves demand -- that is what the exploration "
             "budget buys."},
    {"id": "refuses", "title": "What it refuses",
     "state": {"q": 12, "h": 8, "gamma": 0.7, "mu": 1.0, "belief": "cold"},
     "ask": "The states it will not price, the prices it will never emit, the day it stops itself.",
     "read": "Designed failure modes: a state with a missing cost or no stock is REJECTED, "
             "not priced best-effort; no tier below cost or above the anchor exists in the "
             "action set; exploration suspends when spend passes the stop multiple while "
             "exploitation pricing continues."},
]


def build(cfg, grid, workers=None):
    d_ref = reference_discount(cfg, "_default")
    r_lookup = read_json(cfg["dispersion"]["r_lookup_path"]) or {}
    r = float(r_lookup.get("global", 0.9))
    bel = beliefs(cfg)
    items = [(g, m, name, eps, q, h, d_ref, r)
             for g in grid["gamma"] for m in grid["mu"]
             for name, eps in bel.items() for q in grid["q"] for h in grid["h"]]
    states = map_episodes(one_state, items, cfg, workers)
    ec, sc = cfg["exploration"], cfg["monitoring"]["stop_conditions"]
    return {
        "grid": grid, "beliefs": bel, "d_ref": d_ref, "r": r, "P0": P0,
        "world": WORLD,
        "config": {
            "tier_step": cfg["pricing"]["tier_step"],
            "entry_offsets": cfg["pricing"]["entry_offsets"],
            "epsilon_max": cfg["posterior"]["epsilon_max"],
            "budget_share_of_il": ec["budget_share_of_il"],
            "delta_min_log_bias": ec.get("delta_min_log_bias"),
            "delta_min_bias_multiple": ec.get("delta_min_bias_multiple", 1.0),
            "stop_multiple": sc["exploration_cost_vs_budget"],
            "persistence_days": sc["persistence_days"],
            "tau_adjust_clip": ec["tau_adjust_clip"],
            "config_version": cfg["meta"]["config_version"],
        },
        "scenarios": SCENARIOS,
        "states": states,
    }


# ------------------------------------------------------------------ page

PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Perishable Markdown — how it decides</title>
<style>
:root{--bg:#f7f6f2;--ink:#1e1e1e;--mute:#6b6b6b;--line:#dcd9d0;--card:#fff;--acc:#0b6e4f;--warn:#b3261e;--soft:#e8f1ee;--hatch:#c9c4b6}
*{box-sizing:border-box}body{margin:0;font:15px/1.45 -apple-system,Segoe UI,Helvetica,Arial,sans-serif;color:var(--ink);background:var(--bg)}
header{padding:18px 28px 10px;border-bottom:1px solid var(--line);background:#fff}
header h1{margin:0;font-size:20px;font-weight:600}header p{margin:4px 0 0;color:var(--mute);font-size:13px}
.wrap{display:grid;grid-template-columns:260px 1fr;min-height:calc(100vh - 70px)}
nav{border-right:1px solid var(--line);background:#fff;padding:12px 0}
nav button{display:block;width:100%;text-align:left;border:0;background:none;padding:9px 20px;font:inherit;color:var(--ink);cursor:pointer;border-left:3px solid transparent}
nav button.on{border-left-color:var(--acc);background:var(--soft);font-weight:600}
nav small{display:block;color:var(--mute);font-size:11px;padding:10px 20px 2px;text-transform:uppercase;letter-spacing:.06em}
main{padding:20px 28px 40px;max-width:1180px}
h2{margin:0 0 4px;font-size:22px;font-weight:600}.ask{color:var(--mute);margin:0 0 14px}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:14px 16px}
.card h3{margin:0 0 8px;font-size:13px;text-transform:uppercase;letter-spacing:.06em;color:var(--mute)}
.sl{display:grid;grid-template-columns:150px 1fr 90px;gap:10px;align-items:center;margin:6px 0;font-size:14px}
.sl input{width:100%}.sl b{text-align:right;font-variant-numeric:tabular-nums}
.seg{display:inline-flex;border:1px solid var(--line);border-radius:6px;overflow:hidden}
.seg button{border:0;background:#fff;padding:5px 12px;font:inherit;cursor:pointer}.seg button.on{background:var(--acc);color:#fff}
svg{width:100%;height:auto;display:block}
.score{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums;font-size:14px}
.score th,.score td{padding:6px 8px;border-bottom:1px solid var(--line);text-align:right}.score th:first-child,.score td:first-child{text-align:left}
.score tr.dp td{font-weight:600;color:var(--acc)}
.read{background:var(--soft);border-left:3px solid var(--acc);padding:10px 14px;border-radius:0 6px 6px 0;margin-top:14px}
.never{list-style:none;padding:0;margin:0}.never li{padding:8px 0;border-bottom:1px solid var(--line)}.never li:last-child{border:0}
.never b{color:var(--warn)}.foot{margin-top:18px;color:var(--mute);font-size:12px}
.pill{display:inline-block;border:1px solid var(--line);border-radius:999px;padding:1px 8px;font-size:12px;color:var(--mute);margin-left:6px}
@media (max-width:900px){.wrap{grid-template-columns:1fr}nav{display:flex;overflow-x:auto}.grid{grid-template-columns:1fr}}
</style></head><body>
<header><h1>Perishable Markdown — how it decides</h1>
<p>Twelve situations, answered by the production solver. Move the sliders; every number is a real solve on this machine's config (version __CFGV__). The demand forecast is an input, not a prediction about any SKU.</p></header>
<div class="wrap"><nav id="nav"></nav><main id="main"></main></div>
<script type="application/json" id="data">__DATA__</script>
<script>
const D=JSON.parse(document.getElementById('data').textContent);
const idx={};for(const s of D.states)idx[s.key.join('|')]=s;
const fmt=n=>n==null?'—':Math.round(n).toLocaleString();
const pct=d=>d==null?'—':(100*d).toFixed(1)+'%';
let cur={sc:D.scenarios[0].id,q:20,h:24,gamma:0.5,mu:1,belief:'cold',world:'as_forecast',tau:0,dmin:0};
function nearest(arr,v){return arr.reduce((a,b)=>Math.abs(b-v)<Math.abs(a-v)?b:a)}
function state(){return idx[[cur.gamma,cur.mu,cur.belief,cur.q,cur.h].join('|')]}
function logMove(dref,d){return Math.abs(Math.log((1-d)/(1-dref)))}
function deltaMin(){const b=D.config.delta_min_log_bias;let bias=cur.dmin;if(bias===0&&b){bias=typeof b==='number'?b:(b._default||0)}
 const eps=D.beliefs[cur.belief];return bias?D.config.delta_min_bias_multiple*bias/Math.max(Math.abs(eps),Math.abs(D.config.epsilon_max)):0}
function nav(){const el=document.getElementById('nav');el.innerHTML='<small>how it decides</small>'+D.scenarios.slice(0,6).map(b).join('')+'<small>compared, stressed, learning</small>'+D.scenarios.slice(6,11).map(b).join('')+'<small>how it protects itself</small>'+D.scenarios.slice(11).map(b).join('');
 function b(s){return `<button class="${s.id===cur.sc?'on':''}" onclick="pick('${s.id}')">${s.title}</button>`}}
function pick(id){const s=D.scenarios.find(x=>x.id===id);cur.sc=id;cur.tau=0;cur.dmin=0;Object.assign(cur,s.state);cur.world='as_forecast';render()}
function slider(label,key,arr,f){const v=cur[key];return `<div class="sl"><span>${label}</span><input type="range" min="0" max="${arr.length-1}" value="${arr.indexOf(v)}" oninput="cur.${key}=D.grid.${key}[this.value];render()"><b>${f(v)}</b></div>`}
function curve(st){const W=520,H=230,L=46,B=34;const tiers=st.tiers;const q=st.q_by_tier;const vals=Object.values(q).map(v=>-v);
 const ils=tiers.map((d,j)=>q[j]==null?null:-q[j]);const ymax=Math.max(...vals)*1.15||1,ymin=Math.min(0,...vals);
 const x=j=>L+(W-L-10)*j/(tiers.length-1),y=v=>H-B-(H-B-14)*(v-ymin)/(ymax-ymin||1);
 const dmin=deltaMin();const eps=D.beliefs[cur.belief];const star=st.star;const costs={};for(const j in q)costs[j]=q[star]-q[j];
 let s=`<svg viewBox="0 0 ${W} ${H}"><line x1="${L}" y1="${H-B}" x2="${W-10}" y2="${H-B}" stroke="#bbb"/>`;
 [0,.5,1].forEach(t=>{const v=ymin+t*(ymax-ymin);s+=`<text x="${L-6}" y="${y(v)+4}" font-size="10" text-anchor="end" fill="#888">${fmt(v)}</text>`});
 tiers.forEach((d,j)=>{const allowed=q[j]!=null;const below=d>st.d_max+1e-9;const inad=allowed&&j!==star&&dmin>0&&logMove(st.d_ref,d)<dmin;const aff=allowed&&j!==star&&!inad&&costs[j]<=cur.tau;
  if(below)s+=`<rect x="${x(j)-5}" y="14" width="10" height="${H-B-14}" fill="#eee"/>`;
  if(allowed){s+=`<circle cx="${x(j)}" cy="${y(-q[j])}" r="${j===star?7:4.5}" fill="${j===star?'#0b6e4f':aff?'#e6a700':inad?'#c9c4b6':'#5b8def'}" stroke="#fff" stroke-width="1.5"/>`}
  else if(!below)s+=`<circle cx="${x(j)}" cy="${H-B-6}" r="2" fill="#bbb"/>`;
  if(j%2===0)s+=`<text x="${x(j)}" y="${H-B+14}" font-size="10" text-anchor="middle" fill="#666">${Math.round(d*100)}%</text>`;
  if(Math.abs(d-st.d_ref)<1e-9)s+=`<line x1="${x(j)}" y1="14" x2="${x(j)}" y2="${H-B}" stroke="#0b6e4f" stroke-dasharray="3 3" opacity=".5"/><text x="${x(j)}" y="11" font-size="10" text-anchor="${j>tiers.length*0.85?'end':j<tiers.length*0.15?'start':'middle'}" fill="#0b6e4f">reference</text>`});
 s+=`<text x="${W-10}" y="${H-2}" font-size="10" text-anchor="end" fill="#888">discount tier → expected inventory loss (won)</text></svg>`;
 return s+`<div style="font-size:12px;color:var(--mute);margin-top:6px">● chosen &nbsp;<span style="color:#5b8def">●</span> allowed at entry &nbsp;<span style="color:#e6a700">●</span> affordable at τ &nbsp;<span style="color:#c9c4b6">●</span> excluded by δ_min &nbsp; <span style="background:#eee;padding:0 6px">grey</span> below cost — not in the action set</div>`}
function pathChart(st){const W=520,H=210,L=40,B=28;
 let key='dp';if(cur.sc==='shock')key=cur.world==='half'?'dp_world_half':cur.world==='double'?'dp_world_double':'dp';if(cur.sc==='restock')key='dp_restock';
 const series=[[key,cur.sc==='restock'?'system (+restock)':'system','#0b6e4f'],['flat_reference','flat at reference','#999'],['legacy_ramp','legacy ramp','#b3261e']];
 const qmax=Math.max(...series.map(([k])=>Math.max(...st.paths[k].path.map(p=>p.q))))*1.1||1;const n=cur.h;
 const x=t=>L+(W-L-10)*t/Math.max(n-1,1),y=v=>H-B-(H-B-14)*v/qmax;
 let s=`<svg viewBox="0 0 ${W} ${H}"><line x1="${L}" y1="${H-B}" x2="${W-10}" y2="${H-B}" stroke="#bbb"/>`;
 [0,.5,1].forEach(t=>{s+=`<text x="${L-6}" y="${y(t*qmax)+4}" font-size="10" text-anchor="end" fill="#888">${(t*qmax).toFixed(0)}</text>`});
 for(const [k,label,col] of series){const p=st.paths[k].path;s+=`<polyline fill="none" stroke="${col}" stroke-width="${k===key?2.5:1.5}" points="${p.map(r=>x(r.h)+','+y(r.q)).join(' ')}"/>`;
  if(k===key)p.forEach(r=>{if(r.d!=null&&(r.h%Math.ceil(n/6)===0||r.h===n-1))s+=`<text x="${x(r.h)}" y="${y(r.q)-8}" font-size="10" text-anchor="middle" fill="${col}">${Math.round(r.d*100)}%</text>`})}
 s+=`<text x="${W-10}" y="${H-2}" font-size="10" text-anchor="end" fill="#888">hours → units on the shelf (labels: discount in force)</text></svg>`;
 return s+`<div style="font-size:12px;color:var(--mute);margin-top:6px"><span style="color:#0b6e4f">━</span> system &nbsp;<span style="color:#999">━</span> flat at reference &nbsp;<span style="color:#b3261e">━</span> legacy ramp</div>`}
function score(st){let key='dp';if(cur.sc==='shock')key=cur.world==='half'?'dp_world_half':cur.world==='double'?'dp_world_double':'dp';if(cur.sc==='restock')key='dp_restock';
 const rows=[[key,'system'],['flat_reference','flat at reference'],['legacy_ramp','legacy ramp']];
 return `<table class="score"><tr><th>policy</th><th>sold</th><th>leftover</th><th>scrap cost</th><th>discount cost</th><th>inventory loss</th></tr>`+rows.map(([k,l])=>{const s=st.paths[k].score;return `<tr class="${k===key?'dp':''}"><td>${l}</td><td>${s.sold}</td><td>${s.leftover}</td><td>${fmt(s.scrap_cost)}</td><td>${fmt(s.discount_cost)}</td><td>${fmt(s.il)}</td></tr>`}).join('')+`</table>`}
function explore(st){const q=st.q_by_tier,star=st.star,dmin=deltaMin();const rows=Object.keys(q).map(Number).filter(j=>j!==star).map(j=>{const d=st.tiers[j];const cost=q[star]-q[j];const inad=dmin>0&&logMove(st.d_ref,d)<dmin;return {d,cost,inad,aff:!inad&&cost<=cur.tau}});
 const n=rows.filter(r=>r.aff).length;return `<table class="score"><tr><th>alternative</th><th>move from reference (log)</th><th>cost if forced</th><th>status</th></tr>`+rows.map(r=>`<tr><td>${pct(r.d)}</td><td>${logMove(st.d_ref,r.d).toFixed(3)}</td><td>${fmt(r.cost)}</td><td>${r.inad?'excluded: too close to the reference to teach anything':r.aff?'<b style="color:#e6a700">affordable — drawn uniformly</b>':'over budget'}</td></tr>`).join('')+`</table><p style="font-size:13px;color:var(--mute)">${n?`${n} affordable alternative${n>1?'s':''}: one is drawn uniformly at random, and its cost is charged to the day's exploration budget (${(100*D.config.budget_share_of_il).toFixed(0)}% of trailing markdown IL).`:'Nothing affordable at this τ: the system exploits. Zero spend on a priced day raises τ by the clip the next day.'}</p>`}
function refuses(){return `<ul class="never">
<li><b>Refused, not priced best-effort.</b> A state with no cost, a cost above list, no stock, or a forecast path that disagrees with the hours left raises <code>StateRejected</code>. The caller holds the current price and alerts on the rate.</li>
<li><b>Never below cost.</b> The action set is built from tiers that keep price ≥ cost (grey on the curve is not "not chosen" — it does not exist as an option).</li>
<li><b>Never a higher price within an episode.</b> Under an anchor the action set holds only tiers at or deeper than the price in force; exploration draws from that set.</li>
<li><b>Stops itself.</b> When realised exploration spend exceeds ${D.config.stop_multiple}× the day's budget for ${D.config.persistence_days} consecutive days, or a scrap/margin guardrail breaches, exploration suspends for the cohort while exploitation pricing continues. τ moves at most ${(100*(D.config.tau_adjust_clip[1]-1)).toFixed(0)}% up or ${(100*(1-D.config.tau_adjust_clip[0])).toFixed(0)}% down per day.</li></ul>`}
function render(){nav();const sc=D.scenarios.find(s=>s.id===cur.sc);const st=state();const m=document.getElementById('main');
 const taus=[0,500,1000,2000,4000,8000];if(!taus.includes(cur.tau))cur.tau=0;
 m.innerHTML=`<h2>${sc.title}<span class="pill">${D.beliefs[cur.belief]<0?'ε = '+D.beliefs[cur.belief]:''}</span></h2><p class="ask">${sc.ask}</p>
 <div class="grid">
 <div class="card"><h3>The shelf</h3>
  ${slider('units on the shelf','q',D.grid.q,v=>v)}${slider('hours left','h',D.grid.h,v=>v+' h')}${slider('cost / list price','gamma',D.grid.gamma,v=>(100*v).toFixed(0)+'%')}${slider('expected demand at reference','mu',D.grid.mu,v=>v+' /h')}
  <div class="sl"><span>elasticity belief</span><span class="seg"><button class="${cur.belief==='cold'?'on':''}" onclick="cur.belief='cold';render()">cold start (day one)</button><button class="${cur.belief==='learned'?'on':''}" onclick="cur.belief='learned';render()">learned</button></span><b>ε ${D.beliefs[cur.belief]}</b></div>
  ${cur.sc==='shock'?`<div class="sl"><span>actual demand vs forecast</span><span class="seg">${Object.keys(D.world).map(k=>`<button class="${cur.world===k?'on':''}" onclick="cur.world='${k}';render()">${k.replace('_',' ')}</button>`).join('')}</span><b>×${D.world[cur.world]}</b></div>`:''}
  <div class="sl"><span>exploration budget τ</span><input type="range" min="0" max="${taus.length-1}" value="${taus.indexOf(cur.tau)}" oninput="cur.tau=[${taus}][this.value];render()"><b>${fmt(cur.tau)} won</b></div>
  <div class="sl"><span>δ_min bias scale</span><input type="range" min="0" max="4" value="${[0,0.05,0.1,0.15,0.3].indexOf(cur.dmin)}" oninput="cur.dmin=[0,0.05,0.1,0.15,0.3][this.value];render()"><b>${cur.dmin===0&&D.config.delta_min_log_bias?'config':cur.dmin}</b></div>
  <p style="font-size:12px;color:var(--mute);margin:8px 0 0">List price ${fmt(D.P0)} won · cost ${fmt(D.P0*cur.gamma)} · reference discount ${pct(D.d_ref)} · dispersion r ${D.r.toFixed(2)}</p></div>
 <div class="card"><h3>The decision curve — this hour</h3>${curve(st)}<p style="font-size:13px;margin:8px 0 0">Chosen: <b>${pct(st.tiers[st.star])}</b> off, expected inventory loss <b>${fmt(-st.q_by_tier[st.star])}</b> won over the remaining ${cur.h} hours.</p></div>
 <div class="card"><h3>The path — hour by hour, re-solved from the actual shelf</h3>${pathChart(st)}</div>
 <div class="card"><h3>Scorecard — expected, over the window</h3>${score(st)}</div>
 </div>
 ${cur.sc==='explore'||cur.tau>0?`<div class="card" style="margin-top:16px"><h3>Exploration — the alternatives and what each costs</h3>${explore(st)}</div>`:''}
 ${cur.sc==='refuses'?`<div class="card" style="margin-top:16px"><h3>What it refuses — by construction</h3>${refuses()}</div>`:''}
 <div class="read"><b>How to read it.</b> ${sc.read}</div>
 <p class="foot">Every point is a solve of <code>pricing.dp.solve</code> — the code that prices the shelf — on this machine's config; sliders snap to the precomputed grid. Legacy ramp is an illustrative schedule (reference −15pp, reference, reference +15pp by thirds). The demand forecast is a slider; nothing here is a prediction about a real SKU. Like-for-like comparisons share one demand model; the A/B is the evidence.</p>`}
render();
</script></body></html>
"""


def main():
    ap = argparse.ArgumentParser(prog="tools.scenario_deck", description=__doc__.split("\n")[0])
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--out", default="reports/scenarios.html")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--quick", action="store_true", help="a tiny grid, for tests")
    args = ap.parse_args()
    cfg = load_config(args.config)
    data = build(cfg, QUICK if args.quick else GRID, args.workers)
    size = write_page(data, cfg, args.out)
    print(f"wrote {args.out}  ({len(data['states'])} states, {size // 1024} KB)")


def write_page(data, cfg, out):
    page = (PAGE.replace("__DATA__", json.dumps(data, separators=(",", ":")).replace("</", "<\\/"))
                .replace("__CFGV__", html.escape(str(cfg["meta"]["config_version"]))))
    os.makedirs(os.path.dirname(str(out)) or ".", exist_ok=True)
    with open(out, "w") as f:
        f.write(page)
    return len(page)


if __name__ == "__main__":
    main()
