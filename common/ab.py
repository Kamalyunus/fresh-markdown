"""common.ab -- A/B arm assignment (design section 11).

One definition: the noise floors and the monitor are only comparable while
the assignment is identical.
"""
import hashlib


def arm(sku_id, fc, allocation):
    """Stable hash assignment at the SKU x FC unit -- the same unit must land
    in the same arm across runs and machines for the whole experiment."""
    h = hashlib.md5(f"{sku_id}|{fc}".encode()).digest()
    return "treatment" if int.from_bytes(h[:4], "big") / 2 ** 32 < allocation \
        else "control"
