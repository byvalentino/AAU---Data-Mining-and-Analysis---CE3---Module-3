"""Lab 2 — The contract at the door.

Why this lab exists: a model cannot object. Given a speed of nine hundred metres
per second it returns a probability, so the only thing standing between a
nonsense request and a confident answer is a rule checked before the model is
asked. You write that rule, refuse eight impossible requests by name, and make
every answer carry the version that produced it.
Where it sits: Block 2 — "The data dictionary, with consequences", "Four kinds of
rule, and the one people forget" and "Refusing well", and the definition slides
"Definition — the data contract at the door" and "Definition — refusal at the
door, and provenance in the answer".
What the check grades: all eight impossible requests draw at least one complaint
naming the offending field, the two acceptable ones draw none, every refusal is
status 422 with its complaints attached, the model — replaced by a spy that
counts — is asked exactly once, for the valid request, and the 200 response
carries the probability, the decision, the model version and the threshold. The
contract's fields are also compared against the table Module 2 hands over: a
service that accepts fields its own input table does not promise is a service
documenting something else.
Needs: isinstance, dict.get, model.predict_proba.

Twenty-five minutes.

Module 1 wrote a data dictionary: for every field, the unit, the source, the
valid range, the owner. That document had no consequences. A contract is the
same declaration with consequences attached — the service states what it will
accept and refuses everything else *before the model is asked*.

Why before. A model given nonsense returns a number. It does not know the
request was impossible; it has no way to know. So an unchecked service answers
every request successfully, including the ones that are meaningless, and the
only trace is a slightly worse metric somewhere downstream weeks later.

What you write: validate(request, contract).

    Return a list of complaints, one per violated rule, empty if the request is
    acceptable. Each complaint must name the offending **field** — a refusal
    that does not say which field is a refusal somebody will retry unchanged.

    Five kinds of rule, all in the contract you are given:

      required   the field must be PRESENT. All four are, because the model's
                 door needs four columns -- read the note above CONTRACT.
      nullable   whether a present field may carry null. Speed may not; the
                 three signal strengths may, and a null is what Lab 4's stored
                 median fills.
      type       "number" -- bool is not a number here, and a string is not
                 either, however numeric it looks
      range      minimum and maximum, inclusive, checked only on a value that
                 is actually there
      units      declared for the reader; not enforced, but printed in the
                 refusal so the caller can see what was expected

    Present and null are different failures. Conflating them is what made an
    earlier version of this lab contradict itself, and the note above CONTRACT
    says how.

Then write: respond(request, contract, artefact, threshold).

    If validate returns complaints, return
        {"status": 422, "complaints": [...]}
    and **do not touch the model**. The check watches; asking the model on an
    invalid request fails the lab.

    Otherwise return
        {"status": 200, "probability": float, "decision": "aboard"|"not aboard",
         "model_version": str, "threshold": float}

    Provenance in every response, not only the failures. "Which model gave this
    answer?" has to be answerable from the answer itself, months later, without
    a log search.

Status 422 rather than 400 or 500: the request was well-formed and its contents
were unacceptable (Fielding, Nottingham & Reschke, 2022). 500 would claim the
fault is ours; 400 says the bytes were malformed. Say what actually happened.

The check passes when all eight impossible requests draw at least one complaint
naming the offending field, when the valid request and the one whose signal
strength is null draw none at all, when every impossible request gets status 422
with its complaints attached, when the model is asked exactly once — for the
valid request, and never for an invalid one — and when the 200 response carries
probability, decision, model_version and threshold.
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from lab_support import NotSolved, load_artefact  # noqa: E402

LAB = 2

# Present, and null: why every field is required and only some may be empty
# ---------------------------------------------------------------------------
# An earlier version of this lab declared the three signal strengths "optional"
# and meant two different things by the word at once. The service filled a
# missing one from the stored median and answered; the registered model in Lab 4
# part (b) REFUSED the same request, because a model signature is a set of
# columns and a column that is not there cannot be reordered into place. One
# request, servable by one cure and refused by the other, and neither file said
# so.
#
# The resolution, and it is a distinction worth carrying out of this room:
#
#   a missing FIELD   is a request the service cannot answer. The model was
#                     fitted on four columns and needs four columns; there is
#                     nothing to put in the fourth position and nothing that
#                     says so. Refuse it, and name the field.
#   a null VALUE      is a measurement that was not made. The field is there,
#                     the caller is telling you it is empty, and the stored
#                     median -- fitted on the training rows, never on live
#                     traffic -- goes in its place (Lab 4, prepare()).
#
# Which fields may be empty is a measurement rather than a preference. On the
# day this service is trained, speed carries a value on 100.0 per cent of rows
# and the three signal strengths on 11.3, 25.1 and 30.7 per cent: a phone hears
# whichever beacons are in range and no others. A contract refusing a null
# signal strength would refuse most of the traffic; one accepting a null speed
# would answer a question nobody asked, out of a median.

# The declared range for speed, and where it comes from. This field is the
# phone's own speed, not the vehicle's -- but a phone aboard moves at the
# vehicle's speed, so the vehicle's measured range bounds it too. Module 1
# measured this vehicle's speed over both days of the archive slice this
# repository ships: -3.361 to 3.555 metres per second (Module 1/slides/
# measured.json, speed_range_m_per_s; data/README.md). A contract set exactly
# at a measured extreme refuses the first faster day that ever happens, so the
# bound is the measurement widened by a margin, and the margin is stated
# rather than folded in: one metre per second on each side, about a third of
# the observed spread.
#
# An earlier version of this lab declared -5 to 30 metres per second "from the
# measurement". Thirty metres per second is 108 kilometres per hour, which this
# vehicle never approached; a bound nothing can violate is documentation, not a
# contract.
SPEED_MEASURED_MIN = -3.361
SPEED_MEASURED_MAX = 3.555
SPEED_MARGIN = 1.0

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


def validate(request: dict, contract: dict = CONTRACT) -> list:
    """Every way `request` breaks `contract`. Empty list means acceptable.

    Definition graded by the check:
        accept(x) ⇔ ∀f: (required_f ⇒ f ∈ x) ∧ (f ∈ x ∧ x[f] = null ⇒ nullable_f) ∧ (f ∈ x ∧ x[f] ≠ null ⇒ type(x[f]) = type_f ∧ min_f ≤ x[f] ≤ max_f); otherwise the refusal names every f that failed
        (Breck et al., 2019). Choices: a missing field and a null value are
        different failures, because the model's door needs four columns and the
        stored median fills an empty one; bool is not a number, however Python
        counts it; the bounds are inclusive and are checked only on a value that
        is there; speed's bounds are Module 1's measured range for this vehicle
        widened by a stated margin of 1 metre per second on each side. Slide:
        "Definition — the data contract at the door".
    Needs: isinstance, dict.get
    """
    # TODO: check required, type, and range. Name the field in every complaint.
    raise NotSolved("validate(request, contract) still raises instead of returning complaints")


def respond(request: dict, contract: dict, artefact: dict, threshold: float = 0.5) -> dict:
    """Refuse at the door, or answer with provenance.

    Definition graded by the check:
        complaints ≠ ∅ ⇒ (status 422, every complaint naming its field, model calls = 0); complaints = ∅ ⇒ (status 200, probability p, decision = aboard iff p ≥ t, model_version, threshold t)
        (Fielding, Nottingham & Reschke, 2022, RFC 9110 §15.5.21). Choices: 422
        rather than 400 (the bytes parsed) or 500 (the fault is not ours); the
        decision boundary is p ≥ t, as Lab 3 derives it; the version travels in
        every response, not only the failures. Slide: "Definition — refusal at
        the door, and provenance in the answer".
    Needs: model.predict_proba, dict
    """
    # TODO: validate first. Only ask the model if there is nothing to complain about.
    raise NotSolved("respond(request, contract, artefact, threshold) still raises "
                    "instead of returning a response")


if __name__ == "__main__":
    artefact = load_artefact("v1")
    good = {"speed": 1.2, "rssi1": -70.0, "rssi2": -80.0, "rssiC": -75.0}
    print(respond(good, CONTRACT, artefact))
    print(respond({**good, "speed": 900.0}, CONTRACT, artefact))
