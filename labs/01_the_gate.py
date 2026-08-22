"""Lab 1 — The gate, and the record behind it.

Why this lab exists: a model reaches production because somebody moved a
pointer, and a gate is a rule attached to that move; months later, "which model
answered, trained on what, with which settings?" has to be a lookup rather than
an investigation. You build the pointer and the gate in fifty lines, then record
the same two models in a real registry -- settings, metrics, environment,
signature -- and move the alias the way release moves it.
Where it sits: Block 1 — "Release is moving a pointer" and "Aliases: promotion
is a pointer move", and the definition slides "Definition — release by
indirection", "Definition — the gate", "Definition — model signature" and
"Definition — release by alias".
What the check grades: part (a) — the worse and the tied candidate are refused
with the registry unchanged and a reason quoting a measured score, a better one
is promoted, rollback steps back to v1; part (b) — in a fresh store, two runs
named v1 and v2 carry the module's metrics, the alias `champion` sits on the
version with the higher accuracy, and its signature carries the four feature
names in order; part (c) — the verdict, on six sets of evidence whose right call
is not the same, with a reason built out of the numbers you were handed.
Needs: mlflow (start_run, log_params, log_metrics, mlflow.scikit-learn, infer_signature,
    MlflowClient), pandas.

Twenty-five minutes.

Part (a) — the fifty-line registry.

A model reaches production because somebody moved a pointer. That is the whole
mechanism, and it is worth making explicit: the service loads "the approved
model" by *name*, the registry says which *version* that name currently means,
and releasing is changing one line. Rollback is changing it back.

What that buys you is the question you can answer months later — "which model
gave this answer?" — as a lookup rather than an investigation.

And it gives you somewhere to put a rule. The rule here is a gate: a candidate
is promoted only if it measures better than the model it would replace, on the
agreed metric, on the agreed data.

Refusing a broken model is easy. This lab asks you to refuse a model that is:

    five times the trees        more capacity
    two hundred times the size  more disk, more memory
    seventeen times the work    more cost per request
    nine points worse           on data from a later stretch of time

Everything about it says "newer and bigger". The only thing that matters says
"worse". Read `service/artefacts/metrics.json` and see for yourself.

What you write: promote_if_better(registry, candidate, metrics).

    Return (promoted: bool, reason: str).

    Promote only if the candidate's value of the agreed metric is strictly
    greater than the approved model's. On promotion, write the registry so that
    `approved` names the candidate and `history` keeps every version that has
    ever been approved, in order, so that rollback is possible.

    On refusal, change nothing, and give a reason a person can act on. "Refused"
    is not a reason. "candidate v2 scored 0.7291 against approved v1 at 0.8180"
    is.

Also write: rollback(registry) — move `approved` back to the previous entry in
history, and return the version now approved.

Part (b) — the same two models, in a real registry.

The fifty lines above are the idea. MLflow is the idea kept the way industry
keeps it: every training run recorded *as it happens* -- parameters, metrics,
the environment, the data window and its checksum, and the artefact itself with
a **signature**: the column names, their types and their order, inferred when
the model is logged and enforced when it is asked. The signature is the
mechanism that stops columns being switched around at serving time; Lab 4 shows
it refusing.

What you write: log_training_run(name, model, X, y_pred, params, metrics).

    Inside one run named `name`: log `params`, log `metrics`, then log `model`
    with `mlflow.sklearn.log_model(...)`, a signature inferred from `X` and
    `y_pred`, `X.head(2)` as the input example, and the registered model name
    REGISTERED_MODEL, so that the run becomes a *version* of one named model.
    Return that version number.

    `model` is the artefact as one object -- lab_support.as_pipeline(artefact):
    the stored transform first, then the forest, so the preparation travels
    with the model. Ask it for probabilities, not labels: pass
    `pyfunc_predict_fn="predict_proba"`. Record the environment as it actually
    is: `pip_requirements=environment_pins()`.

Then: promote_by_alias(client, name, version, alias="champion").

    Move the alias onto `version`. That is release. Rollback is the same call
    with the previous version, and the service loads `models:/aboard@champion`
    without ever knowing a version number.

Part (c) — the verdict a person signs.

The gate in part (a) compares one number, agreed before there was a candidate,
and that narrowness is its whole value. It is not the release decision. The
release decision is made by a person who can see everything the gate refuses to
look at, and who has to say one of three words and defend it: **promote**,
**hold**, or **roll back**.

What you write: promotion_verdict(evidence).

    `evidence` is EVIDENCE_KEYS — ten numbers you measured with the other three
    labs: the two accuracies, what the model in service scored on the day it was
    promoted, what the candidate and the model in service cost at the priced
    threshold of Lab 3, what the best policy that uses no model at all costs on
    the same rows, the ninety-fifth percentile latency of Lab 4 against the
    budget the service agreement allows, and the two sizes on disk.

    Return (call, reason). The call is one of VERDICT_CALLS. The reason is the
    argument, and the check grades it as hard as it grades the call: every
    number in it must be one of the numbers you were handed, to the digit, and
    it must name at least two of the quantities it weighed. A sentence copied
    off a slide fails, because the numbers on the slide are not the numbers in
    your hand.

    The unit the decision is made in is the one the operator pays in. Block
    three spent five slides establishing that accuracy weights every mistake
    equally and the operator does not, so a candidate that is more accurate and
    dearer at the priced threshold is a candidate to hold. That is not a trick
    question; it is the module's argument, and one of the six sets of evidence
    is exactly it.

    Write this part last. The quantities it weighs are measured by Lab 3 (the
    two realised costs and the no-model floor) and by Lab 4 (the ninety-fifth
    percentile), so part (c) is the ten minutes at the end of the day when the
    room has them in front of it. The check grades it whenever you run it.

The check passes when the worse candidate and the tied candidate are both refused
with the registry left exactly as it was; when the refusal's reason runs to more
than fifteen characters and quotes one of the two measured scores; when a
genuinely better candidate is promoted, `approved` names it and `history` ends
with it; when rollback(registry) both returns "v1" and leaves `approved` at
"v1"; in a fresh store of its own, when your two runs carry the module's
metrics, `champion` points at the more accurate version, and that version's
signature names speed, rssi1, rssi2, rssiC in that order; and when your verdict
gives the right call on six sets of evidence whose right calls differ, keeps
giving it when the check moves one number at a time, and argues for each of them
out of the numbers it was handed.
"""
from __future__ import annotations

import json
import pathlib
import sys
import tempfile

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from lab_support import (NotSolved, load_metrics, registry_path, load_artefact,   # noqa: E402
                         load_table, as_pipeline, open_store, environment_pins,
                         REGISTERED_MODEL, CHAMPION, FEATURES)

LAB = 1
METRIC = "accuracy"          # the agreed metric, agreed once, in the open
HIGHER_IS_BETTER = True


def promote_if_better(registry: dict, candidate: str, metrics: dict):
    """Promote `candidate` only if it measures better than what is approved.

    Args:
        registry:  {"approved": version, "history": [version, ...]} — modify in place.
        candidate: the version being proposed, e.g. "v2".
        metrics:   {version: {"accuracy": float, ...}} from metrics.json.

    Returns:
        (promoted, reason)

    Definition graded by the check:
        promote(candidate) ⇔ m(candidate) > m(approved) + δ, with δ = 0 here (a tie is refused) and δ = 0.01 in Module 5
        (Zaharia et al., 2018). Choices: m is METRIC, agreed before any candidate;
        the margin δ is a choice — nought here, 0.01 in Module 5, and Module 4's
        Wilson interval on the label count is where a principled δ comes from.
        Slide: "Definition — the gate".
    Needs: dict access, string formatting — no library.
    """
    # TODO: compare on METRIC, promote or refuse, and say why.
    raise NotSolved("promote_if_better(registry, candidate, metrics) still raises "
                    "instead of returning (promoted, reason)")


def rollback(registry: dict) -> str:
    """Move `approved` back to the previous approved version. Return it.

    Definition graded by the check:
        serve(name) = artefact[approved]; promote: approved ← candidate, history ← history + [candidate]; rollback: history ← history[:−1], approved ← history[−1]
        (Zaharia et al., 2018; Schelter et al., 2018). Choices: history is
        append-only, and rollback is the release move reversed, not a separate
        procedure. Slide: "Definition — release by indirection".
    Needs: list.pop — no library.
    """
    # TODO: step back through history.
    raise NotSolved("rollback(registry) still raises instead of returning a version")


def log_training_run(name: str, model, X, y_pred, params: dict, metrics: dict) -> int:
    """Record one training run in the store MLflow currently points at; return the version.

    Args:
        name:    the run name, "v1" or "v2".
        model:   the artefact as one scikit-learn object (lab_support.as_pipeline).
        X:       raw rows with all four features present — the signature is inferred from them.
        y_pred:  model.predict_proba(X), so the output schema is inferred too.
        params:  what was set before training (trees, depth, seed, data window, checksum).
        metrics: what was measured after (accuracy, size in bytes).

    Definition graded by the check:
        S = (name_i, type_i)_{i=1..k}, recorded with the model; accept(x) ⇔ ∀i: name_i ∈ columns(x) ∧ type(x[name_i]) = type_i; input = (x[name_1], …, x[name_k])
        (Zaharia et al., 2018). Choices: the signature is inferred from `X` and
        `y_pred` at logging time; probabilities, not labels
        (pyfunc_predict_fn="predict_proba"); the environment is recorded as the
        versions actually running (environment_pins()). Slide: "Definition — model
        signature".
    Needs: mlflow.start_run, mlflow.log_params, mlflow.log_metrics, mlflow.scikit-learn,
        mlflow.models.infer_signature
    """
    # TODO: one run, params, metrics, then log_model with signature and
    #       registered_model_name=REGISTERED_MODEL; return info.registered_model_version.
    raise NotSolved("log_training_run(name, model, X, y_pred, params, metrics) still "
                    "raises instead of returning the registered version")


def promote_by_alias(client: MlflowClient, name: str, version, alias: str = CHAMPION) -> str:
    """Move `alias` of the registered model `name` onto `version`. Return the version now aliased.

    Definition graded by the check:
        alias: name@alias → version; promote: alias[champion] ← version; rollback: alias[champion] ← previous version; serve(models:/name@champion)
        (Zaharia et al., 2018). Choices: the alias is `champion`; the service
        loads models:/aboard@champion and never a version number; rollback is the
        same call with the previous version. Slide: "Definition — release by
        alias".
    Needs: MlflowClient.set_registered_model_alias,
        MlflowClient.get_model_version_by_alias
    """
    # TODO: set the alias, then read it back and return the version it names.
    raise NotSolved("promote_by_alias(client, name, version, alias) still raises "
                    "instead of moving the alias")


# --------------------------------------------------------------------------
# Part (c) — the verdict a person signs
# --------------------------------------------------------------------------
# The gate is a rule. The verdict is a decision, and a decision has a unit. This
# module spent block three establishing that the operator pays in vehicle-hour
# equivalents rather than in accuracy points, so the verdict is priced in the
# first and not in the second. Everything here is declared in the open, before
# there is a candidate anybody is fond of, which is the only moment at which
# these numbers can be agreed honestly.

VERDICT_CALLS = ("promote", "hold", "roll back")

# The ten quantities a release decision rests on. Every one is something the
# four labs of this module measure: the accuracies from Lab 1's gate, the
# realised costs from Lab 3, the ninety-fifth percentile from Lab 4, the sizes
# from service/artefacts/metrics.json.
EVIDENCE_KEYS = (
    "candidate_accuracy",              # the candidate, on the agreed rows
    "approved_accuracy",               # the model in service, same rows, measured now
    "approved_accuracy_at_promotion",  # what it scored the day the alias moved to it
    "candidate_realised_cost",         # Lab 3's K(t) for the candidate, at the priced threshold
    "approved_realised_cost",          # the same bill for the model in service
    "trivial_baseline_cost",           # the cheaper policy that never asks a model
    "candidate_latency_p95",           # Lab 4's nearest-rank percentile, in milliseconds
    "latency_budget",                  # what the service agreement allows, in milliseconds
    "candidate_megabytes",             # on disk
    "approved_megabytes",              # on disk
)

# How far the model in service may fall below what it scored on the day it was
# promoted before the answer is "put the previous version back". A choice, in
# accuracy points, stated rather than discovered: two points is wider than the
# label noise this data plausibly carries and narrower than any drop worth
# living with. Module 4's Wilson interval on the label count is where a
# principled number comes from, and Module 5 is where it is applied weekly.
REGRESSION_MARGIN = 0.02

# The percentile the service agreement is written against, and the one Lab 4
# reports. Ninety-five is a reporting convention rather than a measurement: it
# says one request in twenty is at least this slow. An agreement written against
# a mean is an agreement about nobody.
LATENCY_PERCENTILE = 95.0


def promotion_verdict(evidence: dict) -> tuple[str, str]:
    """Say what to do with this release, and why. Return (call, reason).

    `call` is one of VERDICT_CALLS:

        "promote"    the candidate replaces the model in service. It is cheaper
                     than the model in service on the rows you both measured,
                     cheaper than the best policy that uses no model at all, and
                     inside the latency budget the service agreement allows.
        "hold"       nothing moves. The candidate has not earned the risk of a
                     release — it is dearer, or level, or it does not clear the
                     no-model floor, or it is too slow.
        "roll back"  the model *in service* has fallen below what it scored on
                     the day it was promoted by more than REGRESSION_MARGIN. Put
                     the previous version back. A candidate is not the answer to
                     a service that has already broken, and rollback is the same
                     move as release, run backwards.

    `reason` is the argument, in one or two sentences, and it is graded:

      * every number in it must be one of the numbers in `evidence`, to the
        digit. A figure remembered from a slide is exactly the habit this course
        exists to break, and the numbers on the slides are not yours;
      * it must name at least two of EVIDENCE_KEYS in words — "the candidate's
        realised cost", "the trivial baseline cost". A verdict rests on a
        comparison and a comparison has two sides;
      * it must name the quantity that actually decided this case. A reason that
        would fit any of the six sets of evidence is an argument about none of
        them.

    Notice what the call is decided in. Not accuracy: accuracy weights every
    mistake equally, which is the assumption block three spent five slides
    rejecting. A candidate that is more accurate *and* dearer at the priced
    threshold is held, and if that feels wrong, the feeling is the lesson.

    Definition graded by the check:
        verdict(evidence) = (call, reason), call ∈ {promote, hold, roll back}; roll back ⇔ m(approved, now) < m(approved, at promotion) − δ_r; promote ⇔ ¬roll back ∧ K(candidate) < K(approved) ∧ K(candidate) < K(trivial) ∧ p95(candidate) ≤ budget; otherwise hold
        (Provost & Fawcett, 2013, ch. 7-8; Sculley et al., 2015). m is the agreed
        metric of the gate card, K the realised cost of Lab 3's card, p95 the
        nearest-rank percentile of Lab 4's, and δ_r is REGRESSION_MARGIN. Choices:
        the decision is priced in what the operator pays and not in accuracy;
        the no-model floor has to be cleared as well as the model in service,
        because a saving against the habit alone is the flattering half; a tie
        is not an improvement, exactly as in the gate. Slide: "Definition — the
        promotion verdict".

    Needs: the evidence dictionary, the constants declared above, and a sentence
        you would be willing to defend to somebody who was not in the room
    """
    # TODO: weigh the evidence, return one of VERDICT_CALLS and the argument for it.
    raise NotSolved("promotion_verdict(evidence) still raises instead of returning "
                    "(call, reason)")


if __name__ == "__main__":
    registry = json.loads(registry_path().read_text())
    metrics = load_metrics()
    print("approved before:", registry["approved"])
    print(promote_if_better(registry, "v2", metrics))
    print("approved after: ", registry["approved"])
