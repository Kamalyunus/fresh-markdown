"""common.ab -- A/B arm assignment (design section 11).

One definition, because two would silently split the population. It lived as a
private helper inside pipeline.monitor and bootstrap.derive_thresholds reached
across a layer to import it -- which worked, but meant the noise floors and the
monitor could drift apart without anything noticing, and they are only
comparable while the assignment is identical.
"""
import hashlib


def arm(sku_id, fc, allocation):
    """Stable hash assignment at the SKU x FC unit.

    Stable across runs and machines: the same unit must land in the same arm
    for the whole experiment, or the comparison is between two moving sets.
    """
    h = hashlib.md5(f"{sku_id}|{fc}".encode()).digest()
    return "treatment" if int.from_bytes(h[:4], "big") / 2 ** 32 < allocation \
        else "control"
