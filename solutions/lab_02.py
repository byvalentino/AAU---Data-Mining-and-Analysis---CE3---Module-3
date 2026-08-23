"""Lab 2, solved — with the reasoning, not only the code.

Run it:  python3 solutions/lab_02.py        (or  python3 labs/02_the_contract.py  after apply.py)
It refuses eight impossible requests one at a time, prints them as a table with
the field each refusal names, shows the two acceptable ones -- including the one
whose signal strength is null -- answers the valid request with its provenance,
and draws the contract's accepted band against the speed the archive actually
recorded.
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from lab_support import NotSolved, load_artefact  # noqa: E402

LAB = 2

SPEED_MEASURED_MIN = -3.361
SPEED_MEASURED_MAX = 3.555
SPEED_MARGIN = 1.0

# Every field is required to be PRESENT, because the model's door is four
# columns wide and a column that is not there cannot be reordered into place.
# Only the three signal strengths may be present and NULL, because a phone hears
# whichever beacons are in range: speed carries a value on 100.0 per cent of the
# training day's rows and the signal strengths on 11.3, 25.1 and 30.7 per cent.
# A null is a measurement that was not made, and the stored median fills it.
NULLABLE_BY_MEASUREMENT = "speed is present on every row; the signal strengths are not"

CONTRACT = {
    "speed": {"required": True, "nullable": False, "type": "number",
              "min": SPEED_MEASURED_MIN - SPEED_MARGIN,
              "max": SPEED_MEASURED_MAX + SPEED_MARGIN,
              "units": "metres per second, signed"},
    "rssi1": {"required": True, "nullable": True, "type": "number",
              "min": -120.0, "max": 0.0, "units": "decibel-milliwatts"},
    "rssi2": {"required": True, "nullable": True, "type": "number",
              "min": -120.0, "max": 0.0, "units": "decibel-milliwatts"},
    "rssiC": {"required": True, "nullable": True, "type": "number",
              "min": -120.0, "max": 0.0, "units": "decibel-milliwatts"},
}


def _is_number(value) -> bool:
    """A boolean is not a number here, whatever Python thinks.

    In Python, True is an instance of int. Left alone, that means a request
    carrying `"speed": true` sails through a naive type check and reaches the
    model as 1.0 metres per second. Excluding bool explicitly is one line and it
    closes a real hole.
    """
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def validate(request: dict, contract: dict = CONTRACT) -> list:
    """Every way the request breaks the contract, each naming its field.

    Definition graded by the check:
        accept(x) ⇔ ∀f: (required_f ⇒ f ∈ x) ∧ (f ∈ x ∧ x[f] = null ⇒ nullable_f) ∧ (f ∈ x ∧ x[f] ≠ null ⇒ type(x[f]) = type_f ∧ min_f ≤ x[f] ≤ max_f); otherwise the refusal names every f that failed
        (Breck et al., 2019). Slide: "Definition — the data contract at the door".

    Present and null are two rules and not one. The field has to be there,
    because the model was fitted on four columns and the platform's signature
    will refuse a request that carries three; whether it may be EMPTY is a
    separate question, answered by whether the measurement is one that is always
    made. Speed always is. A signal strength is made only when a beacon is in
    range, so a null there is information, and the stored median stands in for it.

    Naming the field is not politeness. A refusal that says only "invalid
    request" is one the caller will retry unchanged, because they have no way to
    know what to change. Then you have two problems: a broken client and a
    service being hammered by it.

    The bounds are a decision, not a fact. Module 1 measured this vehicle at
    -3.361 to 3.555 metres per second; a contract set exactly there refuses the
    first slightly faster day that ever occurs, and one set at "minus five to
    thirty" — as this lab used to declare — refuses nothing a vehicle could do.
    The rule taken from that: bound at the measurement plus a margin you state
    out loud, so the next person can argue with the margin instead of guessing
    at the intent.
    """
    complaints = []
    for field, rule in contract.items():
        if field not in request:
            if rule.get("required"):
                complaints.append(
                    f"{field}: required, and the field is absent; the model's door is "
                    f"{len(contract)} columns wide")
            continue

        value = request[field]
        if value is None:
            # Present and empty. Allowed only where the measurement is one that
            # is not always made, and then the stored median stands in for it.
            if not rule.get("nullable"):
                complaints.append(
                    f"{field}: present but null, and this field is never empty in the "
                    f"data the model was fitted on ({rule['units']})")
            continue
        if rule.get("type") == "number" and not _is_number(value):
            complaints.append(
                f"{field}: expected a number in {rule['units']}, got "
                f"{type(value).__name__}")
            continue

        low, high = rule.get("min"), rule.get("max")
        if low is not None and value < low:
            complaints.append(
                f"{field}: {value} is below the minimum {low} ({rule['units']})")
        if high is not None and value > high:
            complaints.append(
                f"{field}: {value} is above the maximum {high} ({rule['units']})")

    # A field nobody declared is not automatically wrong, but it is worth saying
    # so: silently ignoring it is how a caller comes to believe it has an effect.
    for field in request:
        if field not in contract:
            complaints.append(f"{field}: not in the contract, so it will be ignored")
    return complaints


def respond(request: dict, contract: dict, artefact: dict, threshold: float = 0.5) -> dict:
    """Refuse at the door, or answer with provenance.

    Definition graded by the check:
        complaints ≠ ∅ ⇒ (status 422, every complaint naming its field, model calls = 0); complaints = ∅ ⇒ (status 200, probability p, decision = aboard iff p ≥ t, model_version, threshold t)
        (Fielding, Nottingham & Reschke, 2022, RFC 9110 §15.5.21). Slide:
        "Definition — refusal at the door, and provenance in the answer".

    The ordering is the lesson. Validation happens *before* the model is asked,
    because a model given nonsense returns a number rather than an objection —
    it has no way to know the request was impossible. An unchecked service
    therefore answers every request successfully, including the meaningless
    ones, and the only trace is a slightly worse metric somewhere downstream
    several weeks later.

    Provenance goes in every response, not only the interesting ones. "Which
    model gave this answer?" is a question that arrives months later about a
    single request, and it should be answerable from the answer itself rather
    than by correlating timestamps against a deployment log.

    The decision boundary is the one Lab 3 derives: aboard when the probability
    is at or above the threshold. Immaterial for a continuous score, and it is
    still written the same way in every file of this module.
    """
    complaints = validate(request, contract)
    if complaints:
        return {"status": 422, "complaints": complaints}

    transform = artefact["transform"]
    row = [[
        (float(request.get(feature, transform["medians"][feature])
               if request.get(feature) is not None else transform["medians"][feature])
         - transform["means"][feature]) / transform["stds"][feature]
        for feature in transform["features"]
    ]]
    probability = float(artefact["model"].predict_proba(row)[0][1])

    return {
        "status": 200,
        "probability": round(probability, 6),
        "decision": "aboard" if probability >= threshold else "not aboard",
        "model_version": artefact.get("version", "v1"),
        "threshold": threshold,
    }


if __name__ == "__main__":
    import pandas as pd
    import plotly.graph_objects as go
    from _narrate import narrator, show_table, save_figure

    say = narrator(LAB)
    say.info("Lab 2 — the door: eight impossible requests refused by name, two acceptable "
             "ones let through, and the valid one answered with the version that answered it")

    artefact = load_artefact("v1")
    artefact["version"] = "v1"
    say.info("loaded the approved artefact (v1): the forest and the fitted transform "
             "together, from service/artefacts/model_v1.pkl (generated)")
    say.info("the contract's speed bounds: %.3f to %.3f metres per second — Module 1's "
             "measured range for this vehicle, %.3f to %.3f, widened by a stated margin "
             "of %.1f on each side", CONTRACT["speed"]["min"], CONTRACT["speed"]["max"],
             SPEED_MEASURED_MIN, SPEED_MEASURED_MAX, SPEED_MARGIN)

    good = {"speed": 1.2, "rssi1": -70.0, "rssi2": -80.0, "rssiC": -75.0}
    impossible = [
        ("required field missing", {k: v for k, v in good.items() if k != "speed"}),
        ("a declared column absent", {k: v for k, v in good.items() if k != "rssi1"}),
        ("required field is None", {**good, "speed": None}),
        ("a number sent as text", {**good, "speed": "1.2"}),
        ("a boolean sent as a number", {**good, "speed": True}),
        ("above the declared maximum", {**good, "speed": 900.0}),
        ("signal strength above zero", {**good, "rssi1": 40.0}),
        ("below the declared minimum", {**good, "rssiC": -400.0}),
    ]

    rows = []
    for description, request in impossible:
        answer = respond(request, CONTRACT, artefact, 0.2)
        rows.append({"what was wrong": description, "status": answer["status"],
                     "complaints": len(answer["complaints"]),
                     "first complaint": answer["complaints"][0]})
    show_table(pd.DataFrame(rows), "eight impossible requests, eight refusals", logger=say)
    say.info("every refusal names its field, so the caller can act without asking us; "
             "every one is 422, because the request parsed and its contents were "
             "unacceptable — 400 would send them looking at their bytes and 500 would "
             "page us for their mistake (Fielding, Nottingham & Reschke, 2022)")
    say.info("the model was not asked once in those eight: a model given nonsense returns "
             "a number, so the order — validate, then ask — is the whole lesson")
    empty = {**good, "rssi1": None}
    say.info("and the pair that used to contradict each other: rssi1 ABSENT draws %d "
             "complaint(s) — the signature needs four columns — while rssi1 present and "
             "NULL draws %d, because a null is a measurement that was not made and Lab 4 "
             "fills it from the stored median", len(validate({k: v for k, v in good.items()
                                                              if k != "rssi1"}, CONTRACT)),
             len(validate(empty, CONTRACT)))
    filled = respond(empty, CONTRACT, artefact, 0.2)
    say.info("the null request is answered: probability %.4f, decision %r — the median "
             "stood in for the beacon that was not heard", filled["probability"],
             filled["decision"])

    answer = respond(good, CONTRACT, artefact, 0.2)
    say.info("the valid request %s", good)
    say.info("answered: probability %.4f, decision %r, model_version %r, threshold %.2f — "
             "provenance in the answer itself, not in a log somebody has to join against",
             answer["probability"], answer["decision"], answer["model_version"],
             answer["threshold"])
    say.info("the boolean case is the one people lose: isinstance(True, int) is True in "
             "Python, so an unchecked service reads `\"speed\": true` as one metre per "
             "second and answers happily")

    # What the bounds mean, against what the vehicle actually did.
    slice_path = pathlib.Path(__file__).resolve().parent.parent / "data" / "bus_slice.csv.gz"
    speeds = pd.read_csv(slice_path, low_memory=False)["speed"].dropna()
    say.info("the archive slice for this vehicle: %d speed readings, %.3f to %.3f metres "
             "per second (archive)", len(speeds), speeds.min(), speeds.max())
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=speeds, nbinsx=80, marker_color="#52514E",
                               name="archive readings"))
    for value, colour, label in (
            (SPEED_MEASURED_MIN, "#2A78D6", "measured minimum"),
            (SPEED_MEASURED_MAX, "#2A78D6", "measured maximum"),
            (CONTRACT["speed"]["min"], "#E07B39", "contract minimum"),
            (CONTRACT["speed"]["max"], "#E07B39", "contract maximum")):
        fig.add_vline(x=value, line_color=colour, line_dash="dash",
                      annotation_text=f"{label} {value:.3f}", annotation_textangle=-90)
    fig.add_trace(go.Scatter(x=[900.0], y=[1], mode="markers+text", text=["refused: 900"],
                             textposition="middle left", marker=dict(color="#C0392B", size=12),
                             name="a request the contract refuses"))
    fig.update_layout(template="plotly_white",
                      title="The contract is the measurement plus a stated margin",
                      xaxis_title="speed (metres per second, signed)",
                      yaxis_title="archive readings (count)", xaxis_type="log")
    fig.update_xaxes(type="linear", range=[-6, 8])
    save_figure(fig, "contract_bounds", LAB, logger=say)

    say.info("what the check grades: eight refusals each naming their field, no complaint "
             "for the valid request or the null one, status 422 with complaints on every refusal, the "
             "model asked exactly once — for the valid request — and probability, "
             "decision, model_version and threshold in the 200 response")
