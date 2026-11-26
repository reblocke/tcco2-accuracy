"""The All-settings example shared by the classification figure and its Word caption."""

import json
from decimal import ROUND_HALF_UP, Decimal
from hashlib import sha256


def read_lookup(pages):
    receipt = json.loads((pages / "conditional-release.json").read_bytes())
    payload = (pages / "conditional-lookup.json").read_bytes()
    if sha256(payload).hexdigest() != receipt["lookup_sha256"]:
        raise ValueError("Prediction lookup differs from its release receipt")
    lookup = json.loads(payload)
    first, last = receipt["readings_mmhg"]
    readings = range(first, last + 1)
    if len(readings) != receipt["readings_per_setting"] or 45 not in readings:
        raise ValueError("Invalid reading grid for the classification example")
    probabilities = {
        setting: dict(zip(readings, (row["tails"] for row in group["rows"]), strict=True))
        for setting, group in lookup["groups"].items()
    }
    return receipt, probabilities


def positive_calls_per_100(probabilities):
    """At an exact TcCO₂ of 45, a positive call is true when PaCO₂ is at least 45."""
    p, _p52 = probabilities["all"][45]
    if p is None or not 0 <= p <= 1:
        raise ValueError("The All-settings probability at 45 mmHg is unresolved")
    true_positives = int((Decimal(str(p)) * 100).to_integral_value(rounding=ROUND_HALF_UP))
    return true_positives, 100 - true_positives
