"""Lab 4, solved — with the reasoning, not only the code.

Run it:  python3 solutions/lab_04.py        (or  python3 labs/04_skew_and_speed.py  after apply.py)
It causes the skew over the whole test period and measures what it changed,
narrates both cures, the percentiles and the batch-equals-request proof, and
draws the skew histograms and the latency percentiles.
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

PRICED_THRESHOLD = 0.2
TARGET, ABOARD = "label2", "IN"
SKEW_EXCHANGE = ("speed", "rssi1")


def cause_and_cure_skew(frame, artefact: dict, exchange=SKEW_EXCHANGE,
                        threshold: float = PRICED_THRESHOLD) -> dict:
    """Produce the failure, count it, and put it right.

    Definition graded by the check:
        changed(π) = |{i : 1[p_i ≥ t] ≠ 1[p_i^π ≥ t]}| / n, where p_i^π is the same request with two declared fields exchanged and the row assembled in the order the keys arrived; failures(π) = 0; and re-assembling in the stored order gives changed = 0
        (Breck et al., 2019; Huyen, 2022, ch. 7). Slide: "Definition — causing
        the skew, and measuring it".

    The bug is one word. `for name in request` instead of
    `for name in transform["features"]`. Each field is scaled by its own stored
    mean and standard deviation, so every number handed to the model is a
    perfectly good number -- it is simply in the wrong seat. That is why nothing
    catches it: a range check passes, a type check passes, a null check passes,
    the request returns 200 in the usual few milliseconds, and the only thing
    that is wrong is the answer.

    Count the failures even though the count is nought. A student who has
    written `failures = 0` into a report and then watched a quarter of the
    decisions change has learned the shape of this failure better than any
    slide can teach it: the dashboard is green, the error rate is zero, the
    latency is unchanged, and the service is wrong.
    """
    fitted = artefact["transform"]
    stored = list(fitted["features"])
    first, second = exchange

    def scaled(name, value):
        if value is None or value != value:                 # absent, or not a number
            value = fitted["medians"][name]
        return (float(value) - fitted["means"][name]) / fitted["stds"][name]

    honest, caused, cured, failures = [], [], [], 0
    for _, record in frame.iterrows():
        request = {name: (float(record[name]) if record[name] == record[name] else None)
                   for name in stored}

        # The cause: the same four fields, two of them exchanged in the
        # dictionary's own order. Nothing about the request is invalid.
        order = [second if name == first else first if name == second else name
                 for name in stored]
        reordered = {name: request[name] for name in order}

        try:
            honest.append([scaled(name, request[name]) for name in stored])
            # The bug: the row is assembled in the order the keys arrived.
            caused.append([scaled(name, reordered[name]) for name in reordered])
            # The cure: the stored order, whatever order the caller sent.
            cured.append(prepare(reordered, fitted)[0])
        except Exception:                                   # never happens, and that is the point
            failures += 1

    model = artefact["model"]
    before = model.predict_proba(np.asarray(honest, dtype=float))[:, 1] >= threshold
    after = model.predict_proba(np.asarray(caused, dtype=float))[:, 1] >= threshold
    healed = model.predict_proba(np.asarray(cured, dtype=float))[:, 1] >= threshold
    truth = (frame[TARGET].to_numpy() == ABOARD).astype(int)

    return {
        "requests": int(len(frame)),
        "failures": int(failures),
        "exchange": tuple(exchange),
        "threshold": float(threshold),
        "decisions_changed_percent": float((before != after).mean()) * 100,
        "accuracy_before": float((before.astype(int) == truth).mean()),
        "accuracy_after": float((after.astype(int) == truth).mean()),
        "decisions_changed_percent_after_cure": float((before != healed).mean()) * 100,
    }


def prepare(request: dict, transform: dict):
    """Build the model's input from the transform, never from the request.

    Definition graded by the check:
        skew ⇔ prepare_serve(x) ≠ prepare_train(x); cure: input_j = ((x[f_j] if present else median_{f_j}) − mean_{f_j}) / std_{f_j} for f_j in the stored order, j = 1..k
        (Breck et al., 2019). Slide: "Definition — training–serving skew".

    The single line that matters is the loop being over `transform["features"]`
    rather than over the request's own keys. A model does not know its columns
    by name — it knows position one, position two, position three. Iterate over
    whatever order the caller happened to send and one day a client library
    changes its serialiser, the keys arrive alphabetically, signal strength
    lands where speed is expected, and every request still succeeds.

    That is what makes training-serving skew so unpleasant: there is no error to
    find. The service is healthy, the latency is fine, the responses are
    well-formed, and the answers are wrong. The only symptom is a metric drifting
    somewhere downstream, and by then the cause is weeks behind you.

    Filling a missing optional field with the *stored* median matters for the
    same reason. Compute a median from live traffic and the service is preparing
    input differently from how the model was taught — the same failure wearing
    different clothes.
    """
    row = []
    for feature in transform["features"]:
        value = request.get(feature)
        if value is None:
            value = transform["medians"][feature]
        row.append((float(value) - transform["means"][feature]) / transform["stds"][feature])
    return [row]


def percentile_latency(durations, percentile: float = 95.0) -> float:
    """Nearest rank, so the number reported is a duration that really happened.

    Definition graded by the check:
        Q(p) = x_(⌈p · n / 100⌉), where x_(1) ≤ … ≤ x_(n) are the sorted durations
        (Hyndman & Fan, 1996, definition 1; Dean & Barroso, 2013). Slide:
        "Definition — percentile latency, nearest rank".

    Report the mean and you describe nobody. If ninety-four requests take ten
    milliseconds and six take a second, the mean is sixty-nine milliseconds — a
    true number that no user has ever experienced. The ninety-fifth percentile
    says: one request in twenty is at least this slow. That is the sentence an
    operations team can act on, and it is why Dean and Barroso talk about the
    tail and not the average.

    Nearest rank rather than interpolation because an interpolated percentile is
    a number that never occurred. For latency, where the question is "how bad
    does it get", a real observation is the more honest answer.
    """
    ordered = sorted(float(d) for d in durations)
    if not ordered:
        return float("nan")
    rank = max(1, math.ceil(percentile / 100 * len(ordered)))
    return ordered[min(rank, len(ordered)) - 1]


def batch_predict(frame, artefact: dict):
    """The whole table at once, using the same transform as the single path.

    Definition graded by the check:
        batch(frame)_i = model(prepare(row_i)) for every row i, |difference| < 10⁻⁹
        (Breck et al., 2019; Huyen, 2022). Slide: "Definition — one preparation, two paths".

    The rule this obeys: one implementation of the preparation, used by both
    paths. It is tempting to write a fast vectorised version for the batch and
    keep the simple one for requests — and then there are two things that must
    agree forever, maintained by different people under different pressures.
    The check compares the two paths element by element for exactly this reason.
    """
    transform = artefact["transform"]
    prepared = pd.DataFrame(index=frame.index)
    for feature in transform["features"]:
        values = pd.Series(frame[feature], index=frame.index, dtype="float64")
        values = values.fillna(transform["medians"][feature])
        prepared[feature] = (values - transform["means"][feature]) / transform["stds"][feature]
    # .to_numpy(): the model was fitted on plain arrays, and handing it a named
    # frame is a mismatch sklearn will warn about -- and a mismatch is exactly
    # what this lab is about, so it is not one to leave in place.
    return artefact["model"].predict_proba(
        prepared[transform["features"]].to_numpy())[:, 1]


def ask_registered(request: dict, registered):
    """The platform's cure: columns matched by name, order restored, strangers refused.

    Definition graded by the check:
        S = (name_i, type_i)_{i=1..k}, recorded with the model; accept(x) ⇔ ∀i: name_i ∈ columns(x) ∧ type(x[name_i]) = type_i; input = (x[name_1], …, x[name_k])
        (Zaharia et al., 2018). Slide: "Definition — model signature".

    The request goes in as a one-row frame, so its keys are column names. The
    signature recorded in Lab 1 says which names, which types, in which order;
    the platform reorders the columns to match and hands the artefact its own
    preparation, so a caller who reversed the keys gets the right answer. A
    caller who renamed a column gets a refusal, not a wrong number — the
    opposite of what the raw pickle does with a swapped row.
    """
    frame = pd.DataFrame([request])
    try:
        answer = registered.predict(frame)
    except MlflowException:
        return None                                   # refused: the service says 422
    return float(np.asarray(answer)[0][1])            # column 1: aboard, as predict_proba


if __name__ == "__main__":
    import plotly.graph_objects as go
    from _narrate import narrator, show_table, save_figure

    say = narrator(LAB)
    say.info("Lab 4 — cause the skew, cure it twice, report speed honestly, and prove the "
             "batch path is the request path")

    artefact = load_artefact("v1")
    transform = artefact["transform"]
    table = load_table()
    test = table.iloc[int(len(table) * 0.7):].reset_index(drop=True)
    say.info("loaded the champion (v1) and the table: %d rows, generated (seed 20200122); "
             "the later 30 per cent, %d rows, is the test period", len(table), len(test))

    # The skew, on one request: reversed keys, same row -- and what a positional
    # swap does to the same model.
    complete = test.dropna(subset=FEATURES)
    # The first complete row where exchanging two positions moves the answer by
    # more than a tenth. On many rows the forest is unmoved, and a demonstration
    # of a change in the third decimal teaches that the failure is small.
    for index in range(len(complete)):
        natural = {f: float(complete.iloc[index][f]) for f in FEATURES}
        straight = prepare(natural, transform)
        exchanged = [list(straight[0])]
        exchanged[0][0], exchanged[0][-1] = exchanged[0][-1], exchanged[0][0]
        if abs(float(artefact["model"].predict_proba(exchanged)[0][1])
               - float(artefact["model"].predict_proba(straight)[0][1])) > 0.1:
            break
    reversed_keys = {f: natural[f] for f in reversed(FEATURES)}
    say.info("one request with all four fields, %s", natural)
    say.info("prepare() with keys in the stored order    -> %s", [round(v, 4) for v in straight[0]])
    say.info("prepare() with keys reversed               -> %s (identical: the loop is over "
             "the stored order, never the request's keys)",
             [round(v, 4) for v in prepare(reversed_keys, transform)[0]])
    right = float(artefact["model"].predict_proba(straight)[0][1])
    wrong = float(artefact["model"].predict_proba(exchanged)[0][1])
    say.info("the raw pickle, row prepared correctly: probability of aboard %.4f", right)
    say.info("the raw pickle, speed and rssiC exchanged by position: %.4f — accepted without "
             "a word, no exception, status would be 200; at the priced threshold 0.20 that "
             "is the decision %r instead of %r", wrong,
             "aboard" if wrong >= 0.2 else "not aboard",
             "aboard" if right >= 0.2 else "not aboard")

    # The whole test period, caused in the request rather than in a matrix: the
    # number on the deck, produced here.
    report = cause_and_cure_skew(test, artefact, SKEW_EXCHANGE, PRICED_THRESHOLD)
    say.info("caused: %s and %s exchanged in every one of %d requests, the row assembled in "
             "the order the keys arrived", report["exchange"][0], report["exchange"][1],
             report["requests"])
    say.info("measured: %.1f per cent of decisions changed at the priced threshold %.2f; "
             "accuracy %.4f -> %.4f; requests that failed: %d",
             report["decisions_changed_percent"], report["threshold"],
             report["accuracy_before"], report["accuracy_after"], report["failures"])
    say.info("cured: the same reordered requests through prepare(), which loops over the "
             "stored order — decisions changed now: %.1f per cent",
             report["decisions_changed_percent_after_cure"])
    other = cause_and_cure_skew(test, artefact, ("rssi2", "rssiC"), PRICED_THRESHOLD)
    say.info("a different pair, rssi2 and rssiC: %.1f per cent — the damage is a property of "
             "which two columns, not a constant",
             other["decisions_changed_percent"])

    # The histograms, from the same two paths the report measured.
    prepared = np.column_stack([
        ((test[f].astype(float).fillna(transform["medians"][f]) - transform["means"][f])
         / transform["stds"][f]).to_numpy() for f in FEATURES])
    before = artefact["model"].predict_proba(prepared)[:, 1]
    swapped = prepared.copy()
    swapped[:, [0, 1]] = swapped[:, [1, 0]]
    after = artefact["model"].predict_proba(swapped)[:, 1]
    fig = go.Figure()
    edges = np.linspace(0, 1, 41)
    fig.add_trace(go.Histogram(x=before, xbins=dict(start=0, end=1, size=0.025),
                               name="columns as trained", marker_color="#2A78D6", opacity=0.9))
    fig.add_trace(go.Histogram(x=after, xbins=dict(start=0, end=1, size=0.025),
                               name="two columns swapped", marker_color="#E07B39", opacity=0.75))
    fig.update_layout(barmode="overlay", title="Every request succeeded. Both times.",
                      xaxis_title="predicted probability of aboard",
                      yaxis_title="requests (count)", legend=dict(x=0.35, y=0.98))
    save_figure(fig, "skew_histograms", LAB, logger=say)

    # The platform's cure.
    registered = load_registered(CHAMPION)
    say.info("loaded models:/aboard@%s from the module's MLflow store; its signature: %s",
             CHAMPION, registered.metadata.get_input_schema().input_names())
    say.info("registered model, keys reversed  -> %.4f (reordered by name; equals the "
             "correctly prepared pickle path %.4f)", ask_registered(reversed_keys, registered), right)
    renamed = {("rssi_c" if f == "rssiC" else f): v for f, v in natural.items()}
    say.info("registered model, rssiC renamed  -> %s (refused: MlflowException, failed to "
             "enforce schema; the service returns 422)", ask_registered(renamed, registered))
    missing = {f: v for f, v in natural.items() if f != "rssi1"}
    say.info("registered model, rssi1 missing  -> %s (refused: every feature is a required "
             "column; an absent value inside a present column is what the median fills)",
             ask_registered(missing, registered))

    # Honest speed: nearest rank on durations that happened, on this machine.
    durations = []
    for index in range(200):
        row = prepared[index % len(prepared)].reshape(1, -1)
        start = time.perf_counter()
        artefact["model"].predict_proba(row)
        durations.append((time.perf_counter() - start) * 1000)
    say.info("200 single requests to the champion on this machine, today (not a slide "
             "number): mean %.2f ms, median %.2f ms, 95th percentile %.2f ms, 99th %.2f ms — "
             "nearest rank, so each is a duration that happened", float(np.mean(durations)),
             percentile_latency(durations, 50), percentile_latency(durations, 95),
             percentile_latency(durations, 99))
    tail = [10.0] * 94 + [1000.0] * 6
    say.info("the textbook case, 94 requests at 10 ms and 6 at 1000 ms: mean %.1f ms "
             "(describes nobody), 95th percentile %.0f ms (one request in twenty is at least "
             "this slow)", float(np.mean(tail)), percentile_latency(tail, 95))
    ordered = sorted(durations)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(1, len(ordered) + 1)), y=ordered, mode="lines",
                             name="sorted durations", line=dict(color="#2A78D6")))
    for p, colour in ((50, "#52514E"), (95, "#E07B39"), (99, "#C0392B")):
        rank = max(1, math.ceil(p / 100 * len(ordered)))
        fig.add_trace(go.Scatter(x=[rank], y=[ordered[rank - 1]], mode="markers+text",
                                 text=[f"p{p} = {ordered[rank - 1]:.2f} ms"], textposition="top left",
                                 marker=dict(color=colour, size=11), name=f"{p}th percentile"))
    fig.add_hline(y=float(np.mean(durations)), line_dash="dash", line_color="#52514E",
                  annotation_text=f"mean {np.mean(durations):.2f} ms", annotation_position="bottom right")
    fig.update_layout(title="Latency by nearest rank, 200 requests on this machine (not a slide number)",
                      xaxis_title="rank of the request, slowest to the right (1..n)",
                      yaxis_title="duration (milliseconds)")
    save_figure(fig, "latency_percentiles", LAB, logger=say)

    # One preparation, two paths.
    batch = batch_predict(test, artefact)
    single = np.asarray([float(artefact["model"].predict_proba(prepare(
        {f: (float(r[f]) if r[f] == r[f] else None) for f in FEATURES}, transform))[0][1])
        for _, r in test.iterrows()])
    say.info("batch path against request path over %d rows: largest difference %.2e "
             "(one implementation of the preparation, used twice)", len(test),
             float(np.max(np.abs(batch - single))))
    show_table(pd.DataFrame({"path": ["request, one row at a time", "batch, whole table"],
                             "mean probability": [single.mean(), batch.mean()],
                             "rows": [len(single), len(batch)]}), "the two paths", logger=say)

    say.info("what the check grades: the caused skew measured to the check's own arithmetic, "
             "no failed requests and exactly nought changed after the cure; prepare() "
             "identical for reversed keys and filled from the "
             "stored median; nearest-rank percentiles at 50, 90, 95, 99, 100; batch equals "
             "request within 1e-9; the pickle accepts the swapped row and answers wrongly, "
             "the registered model answers the reversed request rightly and refuses the "
             "renamed column")
