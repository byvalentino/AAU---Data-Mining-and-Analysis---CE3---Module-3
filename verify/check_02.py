#!/usr/bin/env python3
"""Check 2 — eight impossible requests, eight refusals, the model never asked.

It also grades the contract's own numbers: the speed bounds have to be the range
Module 1 measured for this vehicle, widened by the margin the lab declares, so
that a request faster than anything recorded but slower than the margin allows is
accepted and one beyond it is not. A bound nobody can violate is documentation
rather than a contract.

And it grades the two rules the lab used to conflate. Every field is REQUIRED to
be present, because the model's door is four columns wide and the platform's
signature refuses a request carrying three (Lab 4 part (b) shows it doing so);
only the three signal strengths may be present and NULL, because a phone hears
whichever beacons are in range. A contract calling a field optional while the
platform refuses a request without it is a contract that contradicts its own
service, and that is what this module shipped before.

Finally it compares the contract's fields against `data/handoff/manifest.json` —
the table Module 2 hands over. A service that accepts a set of fields its own
input table does not promise is documenting something else.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _harness import run, close, explain                              # noqa: E402
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from lab_support import load_artefact, load_handoff                   # noqa: E402

GOOD = {"speed": 1.2, "rssi1": -70.0, "rssi2": -80.0, "rssiC": -75.0}

IMPOSSIBLE = [
    ({k: v for k, v in GOOD.items() if k != "speed"},   "speed",  "required field missing"),
    ({k: v for k, v in GOOD.items() if k != "rssi1"},   "rssi1",  "a declared column absent"),
    ({**GOOD, "speed": None},                           "speed",  "required field is None"),
    ({**GOOD, "speed": "1.2"},                          "speed",  "a number sent as text"),
    ({**GOOD, "speed": True},                           "speed",  "a boolean sent as a number"),
    ({**GOOD, "speed": 900.0},                          "speed",  "above the declared maximum"),
    ({**GOOD, "rssi1": 40.0},                           "rssi1",  "signal strength above zero"),
    ({**GOOD, "rssiC": -400.0},                         "rssiC",  "below the declared minimum"),
]


class Fixed:
    """A stand-in model that always answers the same probability.

    The decision rule is graded on a number chosen by the check rather than on
    whatever the forest happens to return, so "at the threshold" means exactly at
    the threshold.
    """

    def __init__(self, probability):
        self.probability = float(probability)

    def predict_proba(self, rows):
        return [[1.0 - self.probability, self.probability] for _ in rows]


class Spy:
    """Stands in for the model and objects loudly if anybody asks it anything."""

    def __init__(self, real):
        self.real, self.asked = real, 0

    def predict_proba(self, rows):
        self.asked += 1
        return self.real.predict_proba(rows)


def bounds(lab, contract):
    """The declared range is the measurement plus a stated margin, and it is checked.

    Not style: a contract whose bounds nothing can violate refuses nothing, and
    one set exactly at the measured extreme refuses the first slightly faster day
    that occurs. The lab states both numbers, and this is where they have to
    agree with each other.
    """
    speed = contract["speed"]
    close(speed["min"], lab.SPEED_MEASURED_MIN - lab.SPEED_MARGIN, 1e-9,
          "the contract's minimum speed is the measured minimum "
          f"({lab.SPEED_MEASURED_MIN}) less the declared margin ({lab.SPEED_MARGIN} "
          "metre per second)")
    close(speed["max"], lab.SPEED_MEASURED_MAX + lab.SPEED_MARGIN, 1e-9,
          "the contract's maximum speed is the measured maximum "
          f"({lab.SPEED_MEASURED_MAX}) plus the declared margin ({lab.SPEED_MARGIN} "
          "metre per second)")
    assert speed["max"] < 30, (
        f"the contract accepts speeds up to {speed['max']}. This vehicle was measured "
        f"at {lab.SPEED_MEASURED_MAX} metres per second over two days; a bound at thirty "
        "(one hundred and eight kilometres per hour) is a rule nothing can break, which "
        "is documentation rather than a contract.")

    # Just inside the margin: faster than anything recorded, and still served.
    inside = {**GOOD, "speed": lab.SPEED_MEASURED_MAX + lab.SPEED_MARGIN / 2}
    assert lab.validate(inside, contract) == [], (
        f"a speed of {inside['speed']:.3f} was refused. It is above everything the "
        "archive recorded and inside the margin the contract declares, which is exactly "
        "what the margin is for.")
    # Just outside it: refused, and the refusal names speed.
    outside = {**GOOD, "speed": lab.SPEED_MEASURED_MIN - 2 * lab.SPEED_MARGIN}
    complaints = lab.validate(outside, contract)
    assert complaints and any("speed" in str(c) for c in complaints), (
        f"a speed of {outside['speed']:.3f} is beyond the declared minimum "
        f"{speed['min']:.3f} and was accepted")



def presence_and_nullability(lab, contract):
    """A missing field and a null value are different failures.

    Not a matter of taste. The model was fitted on four columns and the
    platform's signature enforces four columns, so a request carrying three is
    one the service cannot answer however willing it is. A value that is *there
    and empty* is a different thing: it is a measurement that was not made, and
    Lab 4's stored median is what stands in for it. This module declared the
    signal strengths "optional", filled them when they were absent, and had the
    registered model refuse the same request in the next lab.
    """
    for field, rule in contract.items():
        assert "nullable" in rule, explain(
            f"m3:contract:nullable-missing:{field}",
            f"the contract's rule for {field!r} has no 'nullable'",
            "Presence and emptiness are two rules. Without the second, 'optional' means "
            "both at once, and the platform's signature disagrees with the door.")
        assert rule.get("required") is True, explain(
            f"m3:contract:not-required:{field}",
            f"the contract declares {field!r} not required",
            "Every one of the four is a column the model's door needs; the signature "
            "restores their order by name and cannot restore a column that is not there. "
            "Lab 4 part (b) grades the platform refusing exactly that request.")

    assert contract["speed"]["nullable"] is False, explain(
        "m3:contract:speed-nullable",
        "the contract lets speed arrive null",
        "Then the median stands in for the one measurement that is always made — speed "
        "carries a value on every row of the day this model was fitted on — and the "
        "service answers a question nobody asked.")
    for field in ("rssi1", "rssi2", "rssiC"):
        assert contract[field]["nullable"] is True, explain(
            f"m3:contract:signal-not-nullable:{field}",
            f"the contract refuses a null {field}",
            "A phone hears whichever beacons are in range: these three carry a value on "
            "roughly a tenth to a third of the rows. Refusing a null one refuses most of "
            "the traffic, and the stored median exists precisely for it.")

    empty = {**GOOD, "rssi1": None}
    assert lab.validate(empty, contract) == [], explain(
        "m3:contract:null-refused",
        f"a request with rssi1 present and null drew {lab.validate(empty, contract)}",
        "Present and empty is allowed for a signal strength, and Lab 4's prepare() fills "
        "it from the stored median. Refusing it here and filling it there is the "
        "contradiction this rule exists to remove.")


def matches_the_handoff(lab, contract):
    """The door accepts what the table upstream promises, field for field."""
    _, manifest = load_handoff()
    promised = list(manifest["feature_columns"])
    assert list(contract) == promised, explain(
        "m3:contract:handoff",
        f"the contract declares {list(contract)} and the hand-off table promises {promised}",
        "data/handoff/manifest.json is the shape Module 2's Lab 4 writes: one row per "
        "phone per window, a mask beside every filled value, the split point, the stored "
        "transform. A service whose door accepts a different set of fields from the table "
        "it is fed is documenting something that does not exist, and the mismatch is "
        "invisible until somebody retrains.")


def body(lab):
    artefact = load_artefact("v1")
    contract = lab.CONTRACT
    bounds(lab, contract)
    presence_and_nullability(lab, contract)
    matches_the_handoff(lab, contract)

    for request, field, description in IMPOSSIBLE:
        complaints = lab.validate(request, contract)
        assert complaints, f"accepted an impossible request — {description}: {request}"
        assert any(field in str(c) for c in complaints), (
            f"refused the request ({description}) but no complaint names '{field}'. "
            f"Complaints were {complaints}. A caller cannot fix what you will not name.")

    assert lab.validate(GOOD, contract) == [], (
        f"refused a valid request: {lab.validate(GOOD, contract)}")

    # The model must not be asked about a request that was never acceptable.
    spy = Spy(artefact["model"])
    watched = {"model": spy, "transform": artefact["transform"], "version": "v1"}
    for request, _, description in IMPOSSIBLE:
        response = lab.respond(request, contract, watched, 0.5)
        assert response["status"] == 422, (
            f"status was {response['status']} for an impossible request ({description}); "
            "expected 422 — the request was well formed and its contents unacceptable")
        assert "complaints" in response, "a 422 with no complaints tells the caller nothing"
    assert spy.asked == 0, explain(
        "m3:contract:model-asked",
        f"the model was asked {spy.asked} time(s) about requests that were never valid",
        "A model given nonsense returns a number — it cannot object. So the order is the "
        "whole lesson: validate, and only then ask.")

    response = lab.respond(GOOD, contract, watched, 0.5)
    assert response["status"] == 200, f"a valid request got status {response['status']}"
    assert spy.asked == 1, f"the model was asked {spy.asked} times for one valid request"
    for field in ("probability", "decision", "model_version", "threshold"):
        assert field in response, (
            f"the response has no '{field}'. Provenance goes in every response — "
            "'which model gave this answer?' is asked months later about one request.")
    assert 0.0 <= response["probability"] <= 1.0, "probability outside [0, 1]"
    assert response["decision"] in {"aboard", "not aboard"}, (
        f"decision was {response['decision']!r}")

    # The boundary, once, everywhere: aboard when the probability is *at* the
    # threshold. Asked of a stand-in model that always answers 0.42, with the
    # threshold set to 0.42 -- exact, so the assertion is about the rule and not
    # about the last bit of a floating-point number.
    fixed = {"model": Fixed(0.42), "transform": artefact["transform"], "version": "v1"}
    on_the_line = lab.respond(GOOD, contract, fixed, 0.42)
    assert on_the_line["decision"] == "aboard", (
        "a model answering exactly 0.42, asked with a threshold of exactly 0.42, was "
        f"turned into {on_the_line['decision']!r}. This module decides aboard when "
        "p >= t — the rule Lab 3 derives from the two costs — so at equality the answer "
        "is aboard.")
    assert lab.respond(GOOD, contract, fixed, 0.43)["decision"] == "not aboard", (
        "a probability of 0.42 below a threshold of 0.43 must be 'not aboard'")


run(2, "02_the_contract", "validate", body)
