"""evaluate.pilot_world -- the demand world the pilot simulator prices against.

A SEPARATE model of the shop, never the agent's: its level at the reference
price is the frozen LightGBM as sealed (with the production calibration,
frozen at simulation start, so the agent's weekly re-fit is graded against
a world that does not move unless `level_drift_per_day` says so), its price
response an ASSUMED elasticity per category, its noise a negative binomial
at the agent's own `r` (scaled by `r_scale` to see the dispersion check
notice). Sales are drawn hour by hour against the shelf; the world writes
the hourly feed row exactly as the source would (tools.make_dummy_flc's
schema: percent discount, write-off sentinel at the close), which is what
the daily lane ingests.

Episodes are TEMPLATES sampled from the prepared extract's DP-eligible
population (the hold-out by default) and re-dated onto simulation days:
the SKU, FC, price, cost, opening hour, window length, opening stock and
the legacy discount path are real; the demand is the world's.

Faults are injected where the real defect would arise -- on the feed row
or on the push, never inside the agent (`FAULTS` names them).
"""

import numpy as np
import pandas as pd

from common.config import reference_discount
from common.io import read_json
from fit.fit_dispersion import lookup_r
from fit.prepare_data import add_ref_rate_features
from fit.train_baseline import BaselineModel
from tools.make_dummy_flc import SCHEMA as FEED_SCHEMA

# every fault the simulator can inject, its argument, and where it lands
FAULTS = {
    "missing": "share of feed rows that never arrive (a completeness gap)",
    "duplicate": "share of feed hours that arrive twice (two states for one hour)",
    "mismatch": "share of priced hours whose shelf price is not the returned "
                "one, with NO failure row (a silent push failure)",
    "push_fail": "share of priced hours whose push failed AND was reported "
                 "(the failures table; the shelf keeps the previous price)",
    "discount_rounding": "the feed's discount column is rounded to whole "
                         "percent (every 2.5pp tier reads as a mismatch)",
    "demand_shock": "from simulation day N on, world demand is multiplied "
                    "by F (`N:F`; a scrap/margin guardrail test)",
}


def parse_faults(specs):
    """`name[:arg]` strings -> {name: value}. Rates are floats in [0, 1];
    `demand_shock` is (day_index, factor); `discount_rounding` takes none."""
    out = {}
    for spec in specs or ():
        name, _, arg = spec.partition(":")
        if name not in FAULTS:
            raise ValueError(f"unknown fault {name!r}; one of {sorted(FAULTS)}")
        if name == "discount_rounding":
            out[name] = True
        elif name == "demand_shock":
            day, _, factor = arg.partition(":")
            out[name] = (int(day), float(factor))
        else:
            rate = float(arg)
            if not 0 <= rate <= 1:
                raise ValueError(f"fault {name}: rate {rate} not in [0, 1]")
            out[name] = rate
    return out


def episode_templates(prepared, cfg, opened_from=None):
    """One template per DP-eligible episode opening on/after `opened_from`
    (default: the hold-out start, the only span no artifact was fit on).
    Carries what a re-dated episode keeps from reality."""
    d = prepared.copy()
    d["date"] = d.date.astype(str)
    d = d[d.dp_eligible]
    start = opened_from or (cfg["data"].get("holdout") or {}).get("start") \
        or cfg["data"]["split"]["test_start"]
    opened = d.groupby("episode_id")["date"].transform("min")
    d = d[opened >= str(start)].sort_values(["episode_id", "date", "hour_of_day"])
    if d.hours_remaining.isna().any():
        # never skipped here: a prepared frame with a null counter is an
        # extract the chain let through (prepare_data's null_key_rows_dropped
        # owns it), and a template built around it would be a window of
        # unknown length priced as real
        bad = sorted(d.loc[d.hours_remaining.isna(), "episode_id"].unique())[:5]
        raise ValueError(f"prepared frame carries a null hours_remaining on "
                         f"{len(bad)}+ DP-eligible episodes ({bad}); re-run "
                         "fit.prepare_data (null_key_rows_dropped)")
    out = []
    for eid, g in d.groupby("episode_id", sort=False):
        first = g.iloc[0]
        n = int(first.hours_remaining) + 1            # hours in the window
        path = [float(x) for x in g.total_discount]
        path += [path[-1]] * (n - len(path))           # sold out early: hold
        out.append({
            "template_id": eid, "sku_id": int(first.sku_id), "fc": str(first.fc),
            "category": str(first.category), "subcategory": str(first.subcategory),
            "original_price": float(first.original_price), "cost": float(first.cost),
            "opening_hour": int(first.hour_of_day), "n_hours": n,
            "q0": int(first.starting_inventory), "legacy_path": path[:n],
        })
    if not out:
        raise ValueError(f"no DP-eligible episode opens on/after {start}")
    return out


def hour_grid(day, opening_hour, n_hours):
    """(date "YYYY-MM-DD", hour) for every hour of a window opening on
    `day` at `opening_hour` -- midnight is an ordinary hour."""
    base = pd.Timestamp(day) + pd.Timedelta(hours=opening_hour)
    return [((base + pd.Timedelta(hours=k)).strftime("%Y-%m-%d"),
             int((base + pd.Timedelta(hours=k)).hour)) for k in range(n_hours)]


def ref_rate_features(history, openings, cfg):
    """The two demand-rate features for episodes OPENING today, computed
    point-in-time by the one home (fit.prepare_data.add_ref_rate_features)
    over the trailing history -- the prepared extract plus every simulated
    hour so far. `openings` rows carry episode_id, sku_id, fc, category,
    date, hour_of_day, starting_inventory; they enter as the day's first
    hour with no sales (not anchor rows), so they read yesterday and
    before. Returns {episode_id: (sku_ref_sales_rate_30d,
    prior_episode_ref_sales_rate)} with NaN where history is empty -- the
    model's own encoding of "unknown"."""
    cols = ["episode_id", "sku_id", "fc", "category", "date", "hour_of_day",
            "starting_inventory", "units_sold", "total_discount"]
    stub = openings.assign(units_sold=0, total_discount=np.nan)[cols]
    frame = pd.concat([history[cols], stub], ignore_index=True)
    frame["d_ref"] = frame.category.map(lambda c: reference_discount(cfg, c))
    feats = add_ref_rate_features(frame, cfg)
    mine = feats[feats.episode_id.isin(set(stub.episode_id))]
    return {r.episode_id: (float(r.sku_ref_sales_rate_30d),
                           float(r.prior_episode_ref_sales_rate))
            for r in mine.itertuples()}


class World:
    """Demand truth. `cfg` is the PRODUCTION config: the level is read from
    the sealed model and calibration as they stand at construction and
    never re-fit, whatever the agent does with its own copy."""

    def __init__(self, cfg, prepared, epsilon_true, seed=0, opened_from=None,
                 r_scale=1.0, level_drift_per_day=1.0, faults=None,
                 episode_shock_sd=0.0):
        self.cfg = cfg
        self.model = BaselineModel(cfg)
        self.r_lookup = read_json(cfg["dispersion"]["r_lookup_path"])
        if self.r_lookup is None:
            raise FileNotFoundError(cfg["dispersion"]["r_lookup_path"])
        self.templates = episode_templates(prepared, cfg, opened_from)
        cats = sorted({t["category"] for t in self.templates})
        if isinstance(epsilon_true, dict):
            missing = [c for c in cats if c not in epsilon_true]
            if missing:
                raise ValueError(f"epsilon_true lacks {missing}")
            self.epsilon_true = {c: float(epsilon_true[c]) for c in cats}
        else:
            self.epsilon_true = {c: float(epsilon_true) for c in cats}
        if any(e >= 0 for e in self.epsilon_true.values()):
            raise ValueError("epsilon_true must be negative (design 5.6)")
        self.r_scale = float(r_scale)
        self.drift = float(level_drift_per_day)
        # the common shock every hour of an episode shares (design 5.6: the
        # reason deff deflates evidence): log-normal, drawn once per episode
        self.episode_shock_sd = float(episode_shock_sd)
        self.faults = dict(faults or {})
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------ demand

    def r_of(self, template):
        return float(lookup_r(self.r_lookup, template["subcategory"],
                              template["category"])) * self.r_scale

    def episode_shock(self):
        """The multiplier one episode's hours all carry."""
        if self.episode_shock_sd <= 0:
            return 1.0
        return float(np.exp(self.rng.normal(-0.5 * self.episode_shock_sd ** 2,
                                            self.episode_shock_sd)))

    def mu_ref_paths(self, openings, model=None):
        """`mu_ref_path` for many openings in ONE prediction: a frame of
        every (opening, hour) row, predicted once, split back. Per-episode
        prediction spent 30 ms of pandas per episode -- at 5,000 a day
        that was the run. Returns a list aligned with `openings`; each
        opening carries `template`, `grid`, `features`."""
        model = model or self.model
        if not openings:
            return []
        rows = []
        for i, o in enumerate(openings):
            tpl, (rate30, prior_rate) = o["template"], o["features"]
            for date, hour in o["grid"]:
                rows.append((i, date, hour, tpl["category"], tpl["subcategory"],
                             tpl["fc"], tpl["original_price"], rate30, prior_rate))
        frame = pd.DataFrame(rows, columns=[
            "_i", "date", "hour_of_day", "category", "subcategory", "fc",
            "original_price", "sku_ref_sales_rate_30d", "prior_episode_ref_sales_rate"])
        frame["total_discount"] = np.nan
        mu = model.predict_mu_ref(frame)
        out = [[] for _ in openings]
        for i, m in zip(frame["_i"].to_numpy(), mu):
            out[i].append(float(m))
        return out

    def demand(self, template, mu_ref, shelf_discount, day_index, episode_shock=1.0):
        """One hour's demand draw at the SHELF price: the world's level,
        the assumed elasticity, the episode's shock, drift and the shock
        fault, NB noise."""
        d_ref = reference_discount(self.cfg, template["category"])
        ratio = (1 - shelf_discount) / (1 - d_ref)
        mu = mu_ref * ratio ** self.epsilon_true[template["category"]]
        mu *= episode_shock * self.drift ** day_index
        shock = self.faults.get("demand_shock")
        if shock and day_index >= shock[0]:
            mu *= shock[1]
        mu = max(float(mu), self.cfg["pricing"]["demand_floor"])
        r = self.r_of(template)
        return int(self.rng.negative_binomial(r, r / (r + mu))), mu

    # -------------------------------------------------------------- feed

    def feed_row(self, template, date, hour, q, shelf_discount, sold, ending,
                 hours_remaining):
        """The hourly feed row in the SOURCE schema. `ending` is the
        source's convention already (the write-off sentinel at a close)."""
        disc_pct = shelf_discount * 100.0
        if self.faults.get("discount_rounding"):
            disc_pct = float(np.round(disc_pct))
        realised = template["original_price"] * (1 - shelf_discount)
        return {
            "date": pd.Timestamp(date).date(), "hour": int(hour),
            "skuseq": int(template["sku_id"]), "fc": template["fc"],
            "inventory": float(q), "discount": float(disc_pct),
            "units_sold": int(sold), "normal_asp": float(template["original_price"]),
            "final_price": float(realised) if sold > 0 else 0.0,
            "cogs_wo_vat": float(template["cost"]),
            "ending_inventory": float(ending),
            "flc_window": float(hours_remaining - 1),
            "category": template["category"], "subcategory": template["subcategory"],
        }

    def draw_fault(self, name):
        rate = self.faults.get(name)
        return bool(rate) and self.rng.random() < rate

    @staticmethod
    def feed_frame(rows):
        """Rows -> the source parquet's exact schema (write with
        `pyarrow.Table.from_pandas(df, schema=FEED_SCHEMA)`)."""
        return pd.DataFrame(rows, columns=[f.name for f in FEED_SCHEMA])
