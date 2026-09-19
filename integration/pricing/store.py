"""The append-only record: one decision per shelf-hour, one rejection per
refused shelf-hour, a quarantine for a line that is not an event.

Every emit takes the store's lock and re-reads the streams' tails first, so
two hourly runs that overlap cannot both price an hour. A torn last line
(a write that died mid-append) is quarantined and closed. The indexes a
later hour reads: `priced_hours`, `latest_by_shelf` (the last decision on a
shelf: the episode it continues and the anchor), `last_seen_by_shelf` (the
last hour seen, priced or refused: what the id rule steps from) and
`episode_paths` (the forecast a later hour of an episode is priced on)."""
import datetime
import fcntl
import json
import os
import re
from contextlib import contextmanager

import numpy as np

from pricing.keys import hour_key, rejection_id_of

DECISION_REQUIRED = [
    "decision_id", "episode_id", "is_entry", "sku_id", "fc", "category",
    "subcategory", "date", "hour_of_day", "hours_remaining", "q_remaining",
    "original_price", "cost", "d_max", "feasible_tier_count", "action_set_size",
    "optimal_price", "optimal_discount", "expected_il", "expected_denominator",
    "applied_price", "applied_discount", "is_exploration", "exploration_cost",
    "affordable_set_size", "tau_current", "delta_min",
    "epsilon_posterior_mean", "epsilon_posterior_std",
    "reference_discount", "reference_mu", "mu_ref_path", "anchor_discount",
    "dispersion_r", "baseline_model_version", "posterior_version", "config_version",
    "config_digest", "timestamp",
]
DECISION_OPTIONAL = ["sku_ref_sales_rate_30d", "prior_episode_ref_sales_rate"]
REJECTION_REQUIRED = [
    "rejection_id", "episode_id", "sku_id", "fc", "date", "hour_of_day",
    "hours_remaining", "q_remaining", "reason", "timestamp",
]
ISO_DAY = re.compile(r"^\d{4}-\d{2}-\d{2}$")
STREAMS = ("decision", "rejection")


def _json_scalar(v):
    if isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return float(v)
    raise TypeError(f"event field of type {type(v).__name__} is not JSON-serialisable")


def _is_iso_day(v):
    if not isinstance(v, str) or not ISO_DAY.match(v):
        return False
    try:
        datetime.date.fromisoformat(v)
    except ValueError:
        return False
    return True


def rejection_event(row, reason, timestamp=None):
    """The record of a refused shelf-hour; None when the row names none."""
    try:
        key = hour_key(row.get("sku_id"), row.get("fc"), row.get("date"), row.get("hour_of_day"))
    except (AttributeError, TypeError, ValueError):
        return None
    return {
        "event": "rejection", "rejection_id": rejection_id_of(key),
        "episode_id": row.get("episode_id"),
        "sku_id": key[0], "fc": key[1], "date": key[2], "hour_of_day": key[3],
        "hours_remaining": row.get("hours_remaining"),
        "q_remaining": row.get("q_remaining", row.get("q")),
        "reason": reason,
        "timestamp": timestamp or datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }


def _decision_key(evt):
    try:
        return hour_key(evt.get("sku_id"), evt.get("fc"), evt.get("date"), evt.get("hour_of_day"))
    except (TypeError, ValueError):
        return None


def _upsert_shelf(index, key, evt, **extra):
    shelf, when = key[:2], (key[2], key[3])
    held = index.get(shelf)
    if held is None or (held["date"], held["hour_of_day"]) <= when:
        index[shelf] = {"episode_id": evt.get("episode_id"), "date": key[2],
                        "hour_of_day": key[3], "hours_remaining": evt.get("hours_remaining"),
                        "q_remaining": evt.get("q_remaining"), **extra}


def _episode_path(evt):
    path = evt.get("mu_ref_path")
    if not isinstance(path, list) or evt.get("episode_id") is None:
        return None
    try:
        out = {"date": str(evt["date"]), "hour_of_day": int(evt["hour_of_day"]),
               "hours_remaining": int(evt["hours_remaining"]),
               "mu_ref_path": [float(m) for m in path], "features": None}
        if all(f in evt for f in DECISION_OPTIONAL):
            out["features"] = tuple(None if evt[f] is None else float(evt[f])
                                    for f in DECISION_OPTIONAL)
        return out
    except (KeyError, TypeError, ValueError):
        return None


def _quarantine_key(evt):
    for kind in ("decision", "rejection"):
        ident = evt.get(f"{kind}_id")
        if ident is not None:
            return (kind, ident)
    if evt.get("raw_line") is not None:
        return ("raw", evt.get("stream"), evt["raw_line"])
    return None


class EventStore:
    def __init__(self, cfg, root=None):
        self.root = root or cfg["events"]["store_dir"]
        os.makedirs(self.root, exist_ok=True)
        self.paths = {k: os.path.join(self.root, f"{k}.jsonl")
                      for k in ("decisions", "rejections", "quarantine")}
        self.quarantined_this_run = 0
        self._quarantined_ids = set()
        with self._locked():
            for _, parsed, _ in self._lines(self.paths["quarantine"]):
                if parsed is None:
                    continue
                evt = parsed.get("event")
                key = _quarantine_key(evt) if isinstance(evt, dict) else None
                if key is not None:
                    self._quarantined_ids.add(key)
            self._terminate_last_line(self.paths["quarantine"])
        self._ids = {k: set() for k in STREAMS}
        self._hour_keys = set()
        self.episode_paths = {}
        self.latest_by_shelf = {}
        self.last_seen_by_shelf = {}
        self._offsets = {k: 0 for k in STREAMS}
        self._line_counts = {k: 0 for k in STREAMS}
        self.last_refusal = None
        with self._locked():
            for kind in STREAMS:
                self._consume(kind)

    @contextmanager
    def _locked(self):
        with open(os.path.join(self.root, ".lock"), "w") as fd:
            fcntl.flock(fd, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(fd, fcntl.LOCK_UN)

    @property
    def priced_hours(self):
        return self._hour_keys

    # ------------------------------------------------------------ reading

    def _consume(self, kind):
        """Register every complete line of the stream past this store's
        offset; quarantine and close a torn last line."""
        path = self.paths[kind + "s"]
        if not os.path.exists(path):
            return
        torn, partial = [], False
        with open(path, "rb") as f:
            f.seek(self._offsets[kind])
            while True:
                line = f.readline()
                if not line:
                    break
                if not line.endswith(b"\n"):
                    partial = True
                    self._line_counts[kind] += 1
                    torn.append((self._line_counts[kind], line.decode(errors="replace")))
                    break
                self._line_counts[kind] += 1
                raw = line.decode(errors="replace").rstrip("\n")
                if not raw.strip():
                    continue
                try:
                    parsed = json.loads(raw)
                    if not isinstance(parsed, dict):
                        raise ValueError("not a JSON object")
                except ValueError:
                    torn.append((self._line_counts[kind], raw))
                    continue
                self._register_line(kind, parsed)
            self._offsets[kind] = f.tell()
        if partial:
            self._terminate_last_line(path)
            self._offsets[kind] = os.path.getsize(path)
        for i, raw in torn:
            self._quarantine({"stream": f"{kind}s", "line_no": i, "raw_line": raw},
                             [f"unparseable JSONL line {i} in {kind}s.jsonl (torn write?) -- skipped on load"])

    def _register_line(self, kind, parsed):
        ident = parsed.get(f"{kind}_id")
        if ident is None or ident in self._ids[kind]:
            return
        self._ids[kind].add(ident)
        if kind == "decision":
            self._register_decision(parsed)
        elif _decision_key(parsed) not in self._hour_keys:
            self._register_seen(parsed, priced=False)

    def _register_seen(self, evt, priced):
        key = _decision_key(evt)
        if key is not None:
            _upsert_shelf(self.last_seen_by_shelf, key, evt, priced=priced)
        return key

    def _register_decision(self, evt):
        key = self._register_seen(evt, priced=True)
        if key is not None:
            self._hour_keys.add(key)
            _upsert_shelf(self.latest_by_shelf, key, evt, applied_discount=evt.get("applied_discount"))
        path = _episode_path(evt)
        if path is not None:
            prev = self.episode_paths.get(evt["episode_id"])
            path["opened"] = prev["opened"] if prev else (path["date"], path["hour_of_day"])
            self.episode_paths[evt["episode_id"]] = path

    @staticmethod
    def _lines(path):
        if not os.path.exists(path):
            return
        with open(path, errors="replace") as f:
            for i, line in enumerate(f, start=1):
                raw = line.rstrip("\n")
                if not raw.strip():
                    continue
                try:
                    parsed = json.loads(raw)
                    if not isinstance(parsed, dict):
                        raise ValueError("not a JSON object")
                except ValueError:
                    parsed = None
                yield i, parsed, raw

    @staticmethod
    def _terminate_last_line(path):
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            return
        with open(path, "rb+") as f:
            f.seek(-1, os.SEEK_END)
            if f.read(1) != b"\n":
                f.write(b"\n")
                f.flush()
                os.fsync(f.fileno())

    # ------------------------------------------------------------ writing

    def _append(self, path, evt):
        with open(path, "a") as f:
            f.write(json.dumps(evt, default=_json_scalar) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def _write(self, kind, evt):
        path = self.paths[kind + "s"]
        self._append(path, evt)
        self._offsets[kind] = os.path.getsize(path)
        self._line_counts[kind] += 1

    def _quarantine(self, evt, problems):
        key = _quarantine_key(evt)
        if key is not None:
            if key in self._quarantined_ids:
                return
            self._quarantined_ids.add(key)
        self.quarantined_this_run += 1
        self._append(self.paths["quarantine"], {"event": evt, "problems": problems})

    def emit_decision(self, evt):
        """True when the decision landed; False (with `last_refusal`) when
        the hour is already priced, the id is held, or the event is not one."""
        with self._locked():
            for kind in STREAMS:
                self._consume(kind)
            self.last_refusal = None
            missing = [f for f in DECISION_REQUIRED if f not in evt]
            problems = [f"missing fields: {missing}"] if missing else []
            if not problems and not _is_iso_day(evt.get("date")):
                problems.append(f"date must be an ISO 'YYYY-MM-DD' string; got {evt.get('date')!r}")
            key = None if problems else _decision_key(evt)
            if not problems and key is None:
                problems.append("sku_id, fc, date and hour_of_day must name one hour of one item")
            if problems:
                self.last_refusal = "quarantined: " + "; ".join(problems)
                self._quarantine(evt, problems)
                return False
            if key in self._hour_keys:
                self.last_refusal = "already_priced: another run committed this hour first"
                return False
            if evt["decision_id"] in self._ids["decision"]:
                self.last_refusal = "duplicate decision id"
                return False
            self._write("decision", evt)
            self._ids["decision"].add(evt["decision_id"])
            self._register_decision(evt)
            return True

    def emit_rejection(self, evt):
        """Record a shelf-hour seen and not priced; never beside a decision."""
        with self._locked():
            for kind in STREAMS:
                self._consume(kind)
            missing = [f for f in REJECTION_REQUIRED if f not in evt]
            problems = [f"missing fields: {missing}"] if missing else []
            if not problems and not _is_iso_day(evt.get("date")):
                problems.append(f"date must be an ISO 'YYYY-MM-DD' string; got {evt.get('date')!r}")
            if not problems and not (isinstance(evt.get("reason"), str) and evt["reason"].strip()):
                problems.append(f"reason must be a non-empty string; got {evt.get('reason')!r}")
            if problems:
                self._quarantine(evt, problems)
                return False
            if evt["rejection_id"] in self._ids["rejection"]:
                return False
            if _decision_key(evt) in self._hour_keys:
                return False
            self._write("rejection", evt)
            self._ids["rejection"].add(evt["rejection_id"])
            self._register_seen(evt, priced=False)
            return True
