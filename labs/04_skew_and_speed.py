"""Lab 4 — Skew, and honest speed.

Why this lab exists: the most unsettling failure in the course is the one where
nothing goes wrong — the same model, prepared differently at serving time,
returns a well-formed answer to every request and a worse one than yesterday.
You cause it, cure it twice (once by discipline, once by the platform's
signature), report speed by a percentile that describes somebody's experience,
and prove that the batch path and the request path are one implementation.
Where it sits: Block 4 — "The failure where nothing goes wrong", "The cure is
one preparation, used twice", "Honest speed", and the definition slides
"Definition — training–serving skew", "Definition — model signature",
"Definition — percentile latency, nearest rank" and "Definition — one
preparation, two paths".
What the check grades: cause_and_cure_skew() exchanges two fields in every
request, reports the share of decisions that changed and the two accuracies to
the check's own arithmetic, reports zero failed requests, and reports that the
cure changes nothing back; prepare() returns one row inside a list built from the
stored constants in the stored order, unchanged when the request's keys arrive
reversed and filled from the stored median when a field is missing;
percentile_latency() gives the nearest-rank duration at every percentile asked;
batch_predict() agrees with the request-by-request path to within 1e-9 on four
hundred rows; and ask_registered() returns the right probability for a request
whose keys are reversed and None for a request with a renamed column, while the
raw pickle silently accepts the swapped row and answers wrongly.
Needs: math, numpy, pandas, mlflow.exceptions.MlflowException.

Twenty-five minutes.

The failure in this lab is the most unsettling in the course, because nothing
goes wrong. Every request succeeds. Every response is well-formed. The service
returns a probability, the log shows two hundred healthy requests a second, and
the answers are quietly worse than they were yesterday.

The cause: the model was trained on columns in one order and is being asked in
another. A model does not know its columns by name. It knows position one,
position two, position three. Hand it signal strength where it expects speed
and it will confidently interpret one as the other.

That is **training-serving skew**: the input at serving time is prepared
differently from the input at training time. Column order is the crudest form;
the same failure hides in a different fill value, a scaler refitted on live
data, or a unit converted in one path and not the other.

Part (a) — cause it, then cure it.

You cause it first, because a failure you have produced is a failure you
recognise. The deck's sharpest number -- two columns exchanged, every request
answered with status 200, twenty-seven per cent of decisions different -- is a
number this function produces. Do not take it from the slide; measure it.

What you write: cause_and_cure_skew(frame, artefact, exchange, threshold).

    Send every row of `frame` to the service twice.

    1. Build the request as the caller would: a dictionary of the four declared
       fields, a value or None for each.
    2. **Cause it.** Rebuild the request with the two fields in `exchange`
       swapped over each other in the dictionary's own order -- a caller whose
       client library serialises its keys differently, which is all it takes.
       Then build the model's input the way a great deal of serving code does:
       loop over the REQUEST's keys, scale each field by its own stored mean and
       standard deviation, and put it at the position it arrived in. Every value
       is right. Two of them are in the wrong seat.
    3. Ask the model. Note what does not happen: no exception, no complaint, no
       slow request, nothing in a log. Count the failures anyway and report
       them, because the count being nought is the lesson.
    4. Measure. What share of decisions changed at `threshold`, and what the
       accuracy was before and after, against `frame[TARGET] == ABOARD`.
    5. **Cure it.** Send the very same reordered requests through prepare(),
       which loops over the stored order and never over the request's keys, and
       measure the share of decisions that changed now. It is nought, exactly.

    Ask the model once per path rather than once per row: collect the rows and
    make three calls in all. A model asked four hundred times in a loop is a
    demonstration of patience.

Then: prepare(request, transform).

    Build the model's input from a request dictionary, using the transform's
    stored feature order and its stored constants. Nothing about the incoming
    dictionary's key order may reach the model. Fill a missing field with the
    stored median rather than refusing or guessing.

    Return that row *inside a list*: [[value, value, value, value]]. A fitted
    model is asked about a batch of rows, so one request is a batch of one.

Then: percentile_latency(durations, percentile).

    Report the 95th percentile, not the mean. A mean hides the tail, and the
    tail is what your users experience: if one request in twenty takes a second,
    a mean of forty milliseconds is a true number that describes nobody's
    experience. Use nearest rank on the sorted durations -- no interpolation,
    so the answer is always a duration that actually happened.

Then: batch_predict(frame, artefact) — the same answers, computed all at once.

    The batch path exists because per-request work is wasteful when the answers
    are wanted for a whole table. The check confirms your batch agrees with the
    request-by-request path exactly. If the two disagree, the batch path is a
    second implementation of the model, and now you have two things to keep
    correct.

Part (b) — the cure by the platform.

Lab 1 registered the same model with a **signature**: the column names, their
types and their order, inferred at logging and enforced at prediction. Load
the champion from the store and ask it with a *named* request instead of a
positional row.

What you write: ask_registered(request, registered).

    `registered` is the pyfunc model behind models:/aboard@champion
    (lab_support.load_registered()). Put the request into a one-row pandas
    DataFrame -- its keys become column names, in whatever order they came --
    and call registered.predict(frame). The platform matches columns by name,
    restores the stored order, and hands the artefact its own preparation. Return
    the probability of aboard: column 1 of the result, exactly as predict_proba.
    If the platform refuses the request (a renamed or missing column raises
    MlflowException: Failed to enforce schema), return None -- the service turns
    that into a 422 naming the schema.

    Three outcomes to see for yourself, in the check and in the solution: the
    raw pickle silently accepts a row with two positions swapped and answers
    wrongly; the registered model, given the same request with its keys
    reversed, answers correctly; given a renamed column, it refuses.

The check passes when cause_and_cure_skew() reports the share of changed
decisions and the two accuracies that the check measures for itself on the same
rows and the same exchange, reports no failed requests, reports exactly nought
changed after the cure, and gives a different answer for a different exchange;
when prepare() returns one row inside a list whose values come from the stored
constants, unchanged when the request's keys arrive reversed and
filled from the stored median when a field is missing; when percentile_latency()
gives the nearest-rank duration at every percentile it is asked for, not only at
95; when batch_predict() agrees with the request-by-request path to within
1e-9 on all four hundred rows; and when ask_registered() answers the reversed
request to within 1e-9 of the correctly prepared pickle path and returns None
for the renamed column, while the pickle accepts the swapped row without a word.
"""
from __future__ import annotations

import math
import pathlib
import sys
import time

import numpy as np
import pandas as pd
from mlflow.exceptions import MlflowException

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from lab_support import (NotSolved, load_artefact, load_table, load_registered,   # noqa: E402
                         CHAMPION, FEATURES)

LAB = 4

# What the two mistakes cost is Lab 3's business; this is the cutoff those two
# prices produce, written once here so that the skew is measured in decisions
# rather than in probabilities. A quarter of a probability's movement is
# invisible until it crosses a line, and the line is what the operator pays for.
PRICED_THRESHOLD = 0.2

# The truth column of the table, and the value that means aboard.
TARGET, ABOARD = "label2", "IN"

# The two fields to exchange. Speed and the first signal strength: one is metres
# per second and the other decibel-milliwatts, so a human reading the request
# would see the mistake at once and the model cannot.
SKEW_EXCHANGE = ("speed", "rssi1")


def cause_and_cure_skew(frame, artefact: dict, exchange=SKEW_EXCHANGE,
                        threshold: float = PRICED_THRESHOLD) -> dict:
    """Cause training-serving skew on purpose, measure it, then cure it.

    Returns a dictionary with, at least:

        requests                              how many were sent
        failures                              how many did not return an answer
        decisions_changed_percent             at `threshold`, cause against truth-blind
        accuracy_before                       before the exchange, at `threshold`
        accuracy_after                        after it
        decisions_changed_percent_after_cure  after prepare() is used instead

    Definition graded by the check:
        changed(π) = |{i : 1[p_i ≥ t] ≠ 1[p_i^π ≥ t]}| / n, where p_i^π is the same request with two declared fields exchanged and the row assembled in the order the keys arrived; failures(π) = 0; and re-assembling in the stored order gives changed = 0
        (Breck et al., 2019; Huyen, 2022, ch. 7). π is the exchange, t the
        threshold. Choices: the decision rule is p ≥ t, as everywhere in this
        module; the share is of decisions and not of probabilities, because a
        probability that moves without crossing the cutoff costs nothing; each
        field keeps its OWN stored mean and standard deviation and only its
        position is wrong, which is the ordinary form of this bug. Slide:
        "Definition — causing the skew, and measuring it".
    Needs: numpy, pandas
    """
    # TODO: build the requests, exchange two fields, assemble in arrival order,
    #       ask the model, count what changed, then assemble in the stored order.
    raise NotSolved("cause_and_cure_skew(frame, artefact, exchange, threshold) still "
                    "raises instead of returning what the exchange changed")


def prepare(request: dict, transform: dict):
    """One request, as the model expects it: stored order, stored constants.

    Return one row inside a list -- [[value, value, value, value]] -- not four
    bare numbers. The model is asked about a batch of rows and a single request
    is a batch of one; hand it a flat list and scikit-learn refuses with
    "Expected 2D array, got 1D array instead".

    Definition graded by the check:
        skew ⇔ prepare_serve(x) ≠ prepare_train(x); cure: input_j = ((x[f_j] if present else median_{f_j}) − mean_{f_j}) / std_{f_j} for f_j in the stored order, j = 1..k
        (Breck et al., 2019). Choices: the order is transform["features"], the
        fill is the stored median, the scale is the stored mean and population
        standard deviation (ddof = 0), all fitted on the training rows by
        service/models.py. Slide: "Definition — training–serving skew".
    Needs: dict.get, float — no library.
    """
    # TODO: build the row in transform["features"] order, fill, scale, wrap it.
    raise NotSolved("prepare(request, transform) still raises instead of returning "
                    "one row inside a list")


def percentile_latency(durations, percentile: float = 95.0) -> float:
    """Nearest-rank percentile of `durations`, so the answer really happened.

    Definition graded by the check:
        Q(p) = x_(⌈p · n / 100⌉), where x_(1) ≤ … ≤ x_(n) are the sorted durations
        (Hyndman & Fan, 1996, definition 1; Dean & Barroso, 2013). Choices:
        nearest rank, never interpolation, so the number reported is a duration
        that occurred; the 95th percentile is the reporting convention. Slide:
        "Definition — percentile latency, nearest rank".
    Needs: sorted, math
    """
    # TODO: sort, take the nearest rank.
    raise NotSolved("percentile_latency(durations, percentile) still raises instead of "
                    "returning a duration")


def batch_predict(frame, artefact: dict):
    """Probabilities for a whole table, agreeing exactly with the per-request path.

    Definition graded by the check:
        batch(frame)_i = model(prepare(row_i)) for every row i, |difference| < 10⁻⁹
        (Breck et al., 2019; Huyen, 2022). Choices: one implementation of the
        preparation — the stored order, the stored median, the stored scale —
        applied to a whole frame; the model asked once. Slide: "Definition — one
        preparation, two paths".
    Needs: pandas, model.predict_proba
    """
    # TODO: apply the stored transform to the frame, then predict once.
    raise NotSolved("batch_predict(frame, artefact) still raises instead of returning "
                    "probabilities")


def ask_registered(request: dict, registered):
    """The registered model, asked by name: the probability of aboard, or None if refused.

    Definition graded by the check:
        S = (name_i, type_i)_{i=1..k}, recorded with the model; accept(x) ⇔ ∀i: name_i ∈ columns(x) ∧ type(x[name_i]) = type_i; input = (x[name_1], …, x[name_k])
        (Zaharia et al., 2018). Choices: the request travels as a one-row pandas
        DataFrame, keys in the caller's order; the answer is column 1 of
        registered.predict(frame), the probability of aboard; a refusal
        (MlflowException) becomes None. Slide: "Definition — model signature".
    Needs: pandas, registered.predict, mlflow.exceptions.MlflowException
    """
    # TODO: one-row frame from the request; predict; column 1; None on MlflowException.
    raise NotSolved("ask_registered(request, registered) still raises instead of "
                    "returning a probability or None")


if __name__ == "__main__":
    artefact = load_artefact("v1")
    table = load_table().tail(500)
    print("batch mean probability:", float(batch_predict(table, artefact).mean()).__round__(4))
    print("95th percentile of [1,2,3,...,100]:", percentile_latency(list(range(1, 101)), 95))
