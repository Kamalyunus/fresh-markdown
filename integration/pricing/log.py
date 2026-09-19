"""The append-only log of what the hour decided: one JSONL line per decision
and one per refused shelf-hour, under `events.store_dir`.

The service never reads it; the learning lane in the owner's repository
does (its store loads these lines, dedups by id and quarantines a torn
one). An append takes a lock so two runs cannot interleave lines. A re-run
of an hour appends the same decision again -- same inputs, same decision,
same id -- and the reader keeps the first."""
import datetime
import fcntl
import json
import os

from pricing.keys import hour_key, rejection_id_of

STREAMS = ("decisions", "rejections")


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


class EventLog:
    def __init__(self, cfg, enabled=True):
        self.root = cfg["events"]["store_dir"]
        self.enabled = enabled          # False on a dry run: nothing is written
        os.makedirs(self.root, exist_ok=True)

    def append(self, stream, events):
        """Append `events` to the stream in one locked write; the count."""
        if stream not in STREAMS:
            raise ValueError(f"no such stream: {stream}")
        if not self.enabled or not events:
            return 0
        with open(os.path.join(self.root, ".lock"), "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                with open(os.path.join(self.root, f"{stream}.jsonl"), "a") as f:
                    for evt in events:
                        f.write(json.dumps(evt) + "\n")
                    f.flush()
                    os.fsync(f.fileno())
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)
        return len(events)
