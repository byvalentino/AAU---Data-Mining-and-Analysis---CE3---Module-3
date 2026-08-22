#!/usr/bin/env python3
"""Check 4 — the skew caused and cured, honest latency, and one implementation.

Part (a) begins with the failure itself: the student exchanges two fields in
every request, and the check measures the same thing its own way -- by exchanging
two columns of the prepared matrix -- so that the two methods have to agree. If
they do, the student has just produced the module's sharpest number rather than
read it off a slide. Then the cure by discipline: the stored order, the stored constants,
nearest-rank percentiles, and the batch path agreeing with the request path.
Part (b) grades the cure by the platform: the request asked of the registered
model by name, answered when the keys arrive reversed and refused when a column
is renamed or absent — while the raw pickle accepts the exchanged row without a
word. It reads the module's own store; run identifiers are random, so nothing
here compares one.
"""
import sys, pathlib, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _harness import run, close, not_ready, explain                   # noqa: E402
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
try:
    from lab_support import (load_artefact, load_table, load_registered,   # noqa: E402
                             CHAMPION, FEATURES)
    import numpy as np                                                # noqa: E402
    import pandas as pd                                               # noqa: E402
except ImportError as unready:
    not_ready(unready)


class Watched:
    """The registered model, with a note of how it was asked.

    Part (b) is about the platform doing the work: the signature matches columns
    by name and restores their order. A student who prepares the row positionally
    and asks the pickle instead would get the same number for the reversed
    request, so the check records whether `registered.predict` was called at all,
    and with what. Everything else is delegated untouched.
    """

    def __init__(self, model):
        self.model = model
        self.asked_with = []

    def predict(self, frame, *arguments, **keywords):
        self.asked_with.append(frame)
        return self.model.predict(frame, *arguments, **keywords)

    def __getattr__(self, name):
        return getattr(self.model, name)



def reference_skew(lab, artefact, frame, exchange, threshold):
    """The same measurement by another route: two columns of the prepared matrix.

    The student causes the skew in the request -- two fields exchanged, the row
    assembled in the order the keys arrived. This computes it by preparing every
    row correctly and then exchanging two columns of the matrix, which is how
    `slides/make_figs.py` measured the number on the deck. Two routes to one
    quantity: if they disagree, one of them is wrong, and the check says which
    numbers it got.
    """
    transform = artefact["transform"]
    features = list(transform["features"])
    prepared = np.column_stack([
        ((frame[f].astype(float).fillna(transform["medians"][f]) - transform["means"][f])
         / transform["stds"][f]).to_numpy() for f in features])
    first, second = features.index(exchange[0]), features.index(exchange[1])
    swapped = prepared.copy()
    swapped[:, [first, second]] = swapped[:, [second, first]]

    before = artefact["model"].predict_proba(prepared)[:, 1] >= threshold
    after = artefact["model"].predict_proba(swapped)[:, 1] >= threshold
    truth = (frame[lab.TARGET].to_numpy() == lab.ABOARD).astype(int)
    return {"decisions_changed_percent": float((before != after).mean()) * 100,
            "accuracy_before": float((before.astype(int) == truth).mean()),
            "accuracy_after": float((after.astype(int) == truth).mean())}


def caused_and_cured(lab, artefact, table):
    """The failure, produced by the student, measured against the check's own arithmetic."""
    frame = table.iloc[int(len(table) * 0.7):].reset_index(drop=True)
    report = lab.cause_and_cure_skew(frame, artefact, lab.SKEW_EXCHANGE, lab.PRICED_THRESHOLD)
    assert isinstance(report, dict), (
        f"cause_and_cure_skew returned {type(report).__name__}; it returns a dictionary of "
        "what the exchange changed")
    for key in ("requests", "failures", "decisions_changed_percent", "accuracy_before",
                "accuracy_after", "decisions_changed_percent_after_cure"):
        assert key in report, (
            f"the report has no {key!r}. The six keys are in the docstring, and each is "
            "one line of arithmetic over the decisions you already have.")

    assert int(report["requests"]) == len(frame), (
        f"you reported {report['requests']} requests over a frame of {len(frame)} rows")
    assert int(report["failures"]) == 0, explain(
        "m3:skew:failures",
        f"you reported {report['failures']} failed request(s)",
        "Not one of them fails, and that is the whole point of this lab. Every request is "
        "well formed, every field is inside its declared range, the model answers in the "
        "usual few milliseconds, and the answers are different. A count of nought here is "
        "the sentence to remember.")

    expected = reference_skew(lab, artefact, frame, lab.SKEW_EXCHANGE, lab.PRICED_THRESHOLD)
    close(report["decisions_changed_percent"], expected["decisions_changed_percent"], 1e-6,
          "the share of decisions that changed when "
          f"{lab.SKEW_EXCHANGE[0]} and {lab.SKEW_EXCHANGE[1]} were exchanged. The check "
          "measures it by exchanging two columns of the prepared matrix; you should get the "
          "same number by exchanging two fields of the request and assembling the row in "
          "the order the keys arrived")
    close(report["accuracy_before"], expected["accuracy_before"], 1e-9,
          "the accuracy before the exchange, at the priced threshold")
    close(report["accuracy_after"], expected["accuracy_after"], 1e-9,
          "the accuracy after the exchange, at the priced threshold")
    assert report["accuracy_after"] < report["accuracy_before"], (
        f"you reported accuracy {report['accuracy_after']:.4f} after the exchange against "
        f"{report['accuracy_before']:.4f} before. Exchanging two columns cannot improve a "
        "model; check which decisions you compared against the truth.")
    close(report["decisions_changed_percent_after_cure"], 0.0, 1e-12, explain(
        "m3:skew:cure",
        "the cure did not put every decision back",
        "prepare() loops over transform['features'] and never over the request's keys, so "
        "the order the caller sent them in cannot reach the model at all. Not nearly "
        "nought: nought."))

    # A different pair, so that the first answer cannot have been typed in.
    other = ("rssi2", "rssiC")
    again = lab.cause_and_cure_skew(frame, artefact, other, lab.PRICED_THRESHOLD)
    reference = reference_skew(lab, artefact, frame, other, lab.PRICED_THRESHOLD)
    close(again["decisions_changed_percent"], reference["decisions_changed_percent"], 1e-6,
          f"exchanging {other[0]} and {other[1]} instead changes a different share of the "
          "decisions, and the function has to measure it rather than report the pair it was "
          "written for")
    assert abs(again["decisions_changed_percent"]
               - report["decisions_changed_percent"]) > 1e-6, (
        "fixture error: the two exchanges changed the same share of decisions")


def body(lab):
    artefact = load_artefact("v1")
    transform = artefact["transform"]
    features = transform["features"]
    table = load_table().tail(400).reset_index(drop=True)

    # --- prepare() must use the stored order, whatever order the caller sent.
    # A row with every declared field present, so the check's own arithmetic below
    # is not comparing against a fill.
    complete = table.dropna(subset=features)
    assert len(complete) > 0, "fixture error: no row carries all four features"
    row = complete.iloc[0]
    natural = {f: float(row[f]) for f in features}
    shuffled = {f: natural[f] for f in reversed(features)}          # same data, reversed keys
    assert list(shuffled) != list(natural), "fixture error: the keys did not reverse"

    straight = lab.prepare(natural, transform)
    reversed_keys = lab.prepare(shuffled, transform)

    # Said here, in a sentence, rather than left to scikit-learn to raise sixty
    # lines further down. A flat row is the natural reading of "build the row",
    # and the traceback it earns names a reshape rather than the contract.
    assert np.asarray(straight, dtype=float).ndim == 2, (
        "prepare() returned a flat row of numbers. The model is asked about a batch "
        "of rows, so one request is a batch of one: return [[...]] rather than [...].")

    assert np.allclose(np.asarray(straight, dtype=float),
                       np.asarray(reversed_keys, dtype=float)), (
        "the same request with its keys in a different order produced a different "
        "input to the model. That is training-serving skew, and it is the failure "
        "where nothing goes wrong: every request succeeds and every answer is worse. "
        "Loop over transform['features'], never over the request's own keys.")

    # And the values must be the stored transform's, not recomputed.
    expected = [(natural[f] - transform["means"][f]) / transform["stds"][f]
                for f in features]
    close(float(np.asarray(straight, dtype=float).ravel()[0]), expected[0], 1e-6,
          "prepare() scaled with something other than the stored mean and standard "
          "deviation")

    # A missing optional field is filled from the stored median, not refused.
    gap = dict(natural); gap[features[-1]] = None
    filled = np.asarray(lab.prepare(gap, transform), dtype=float).ravel()[-1]
    close(filled, (transform["medians"][features[-1]] - transform["means"][features[-1]])
          / transform["stds"][features[-1]], 1e-6,
          "a missing optional field must be filled with the *stored* median")

    # --- cause it, and cure it. Graded after prepare() because the cure IS
    # prepare(), so a broken one should be reported as a broken prepare().
    caused_and_cured(lab, artefact, load_table())

    # --- percentile, nearest rank.
    close(lab.percentile_latency(list(range(1, 101)), 95), 95.0, 1e-9,
          "the 95th percentile of 1..100 by nearest rank is 95")
    # Ninety-four requests at 10 ms and six at 1000 ms. The mean is 69.4 -- a true
    # number describing nobody. The 95th percentile is 1000, which is the sentence
    # an operations team can act on.
    tail = [10.0] * 94 + [1000.0] * 6
    close(lab.percentile_latency(tail, 95), 1000.0, 1e-9,
          "with six requests in a hundred taking a second, the 95th percentile is a "
          "second. A mean would report 69.4, which describes nobody.")
    close(lab.percentile_latency([5.0], 95), 5.0, 1e-9, "a single duration")
    assert abs(lab.percentile_latency(tail, 95) - float(np.mean(tail))) > 1, (
        "percentile_latency returned the mean. The tail is what users experience.")

    # Ninety-five is the only percentile asked for above, so hard-coding it costs
    # nothing. The argument has to do something.
    for percentile, expected_rank in ((50, 50.0), (90, 90.0), (99, 99.0), (100, 100.0)):
        close(lab.percentile_latency(list(range(1, 101)), percentile), expected_rank,
              1e-9,
              f"the {percentile}th percentile of 1..100 by nearest rank is "
              f"{expected_rank:.0f}. The `percentile` argument is not decoration — an "
              "operations team asks for the median as often as for the tail")

    # --- batch and single must agree exactly, because they must be one implementation.
    batch = np.asarray(lab.batch_predict(table, artefact), dtype=float)
    assert len(batch) == len(table), (
        f"batch_predict returned {len(batch)} probabilities for {len(table)} rows")

    single = []
    for _, record in table.iterrows():
        request = {f: (float(record[f]) if record[f] == record[f] else None) for f in features}
        single.append(float(artefact["model"].predict_proba(lab.prepare(request, transform))[0][1]))
    single = np.asarray(single)

    worst = float(np.max(np.abs(batch - single)))
    assert worst < 1e-9, (
        f"the batch path and the request path disagree by up to {worst:.2e}. They must "
        "be one implementation of the preparation used two ways — two implementations "
        "is two things that have to stay correct forever, maintained under different "
        "pressures.")

    part_b(lab, artefact, transform, complete)


def part_b(lab, artefact, transform, complete):
    """The platform's cure: the signature, enforced at prediction time.

    Three outcomes, and the check wants all three. The raw pickle accepts a row
    whose positions were exchanged without a word and answers wrongly; the
    registered model, asked by name with the keys reversed, answers correctly;
    a renamed column is refused rather than guessed at.

    The store is the module's own -- setup.sh wrote it and check 0 verified it.
    Nothing here compares a run identifier: identifiers are random, content is
    not.
    """
    features = transform["features"]

    # A row where exchanging two positions actually changes the answer, so the
    # first outcome has something to show. Chosen by scanning in order, so the
    # same row is chosen on every machine.
    chosen = None
    for index in range(len(complete)):
        request = {f: float(complete.iloc[index][f]) for f in features}
        row = np.asarray(lab.prepare(request, transform), dtype=float)
        exchanged = row.copy()
        exchanged[0, 0], exchanged[0, -1] = exchanged[0, -1], exchanged[0, 0]
        right = float(artefact["model"].predict_proba(row)[0][1])
        # Outcome one: no exception, no complaint, status 200, wrong number.
        quietly = float(artefact["model"].predict_proba(exchanged)[0][1])
        if abs(quietly - right) > 0.1:
            chosen = (request, right, quietly)
            break
    assert chosen is not None, (
        "fixture error: exchanging speed and rssiC changed no answer by more than 0.1 on "
        "any complete row of the test tail, so outcome one has nothing to show")
    natural, right, quietly = chosen
    assert abs(quietly - right) > 0.1, (
        f"fixture error: the pickle answered {quietly:.4f} against {right:.4f}")

    # Outcome two: the same request, keys reversed, asked of the registered model.
    registered = Watched(load_registered(CHAMPION))
    reversed_keys = {f: natural[f] for f in reversed(features)}
    assert list(reversed_keys) != list(natural), "fixture error: the keys did not reverse"
    answer = lab.ask_registered(reversed_keys, registered)
    assert answer is not None, (
        "ask_registered returned None for a request carrying all four fields under their "
        "own names. The signature reorders columns by name; only a column it does not "
        "know is refused.")
    close(float(answer), right, 1e-9,
          "ask_registered on the reversed request did not give the probability of aboard. "
          "The answer is column 1 of registered.predict(frame) -- the same layout as "
          f"predict_proba -- and it must equal the correctly prepared pickle path ({right:.4f})")
    assert registered.asked_with, (
        "ask_registered never called registered.predict(). Part (b) is the platform doing "
        "the work: preparing the row yourself and asking the pickle is part (a) again, and "
        "it is exactly the discipline the signature exists to replace.")
    frame = registered.asked_with[0]
    assert isinstance(frame, pd.DataFrame), (
        f"registered.predict was called with {type(frame).__name__}. Send a one-row pandas "
        "DataFrame: its column names are what the signature matches on.")
    assert len(frame) == 1, (
        f"registered.predict was called with {len(frame)} rows; one request is one row")

    # A second row, so that a constant is not an answer.
    other = None
    for index in range(len(complete)):
        candidate = {f: float(complete.iloc[index][f]) for f in features}
        row = np.asarray(lab.prepare(candidate, transform), dtype=float)
        value = float(artefact["model"].predict_proba(row)[0][1])
        if abs(value - right) > 0.02:
            other = (candidate, value)
            break
    assert other is not None, "fixture error: every complete row scores within 0.02 of the first"
    second, expected = other
    close(float(lab.ask_registered({f: second[f] for f in reversed(features)}, registered)),
          expected, 1e-9,
          "ask_registered gave the same answer to a different request. It is returning a "
          "number of its own rather than the model's")

    # Outcome three: a renamed column, and a missing one, are refused -- and the
    # refusal arrives as None, which the service turns into 422 naming the schema.
    renamed = {("rssi_c" if f == "rssiC" else f): v for f, v in reversed_keys.items()}
    assert lab.ask_registered(renamed, registered) is None, (
        "a request with rssiC renamed to rssi_c was answered instead of refused. The "
        "signature knows four names; a fifth is not a column it can restore the order of, "
        "and guessing is how a service answers confidently about the wrong thing. Catch "
        "MlflowException and return None.")
    missing = {f: v for f, v in reversed_keys.items() if f != "rssi1"}
    assert lab.ask_registered(missing, registered) is None, (
        "a request with rssi1 absent was answered instead of refused. Every feature is a "
        "required column at the model's door; an absent *value* inside a present column is "
        "what the stored median fills, and that is a different thing.")


run(4, "04_skew_and_speed", "prepare", body)
