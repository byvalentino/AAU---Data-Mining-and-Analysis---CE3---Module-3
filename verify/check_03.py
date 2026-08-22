#!/usr/bin/env python3
"""Check 3 — the threshold comes from the costs, it costs less than the habit,
and the probabilities it thresholds are measured against what actually happened.

The arithmetic first, with no data: the ratio at twenty-nine pairs of costs, and
the direction people invert. Then the realised cost of those decisions on the
model's own output. Then calibration, which this module taught across five slides and used to grade
nowhere: the student's bands against the check's own, on the service's output,
and then on two invented models built so that the MORE accurate of the two is
the badly calibrated one. Accuracy and calibration are different properties of
the same numbers, and a service that prices its decisions needs the second.

Then, only if the student wrote it, the optional departure
where the price of a miss is not a constant -- graded against the same exact
Poisson sums the module measured, so Module 3's slide and Module 4's proof agree
to the last printed digit.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _harness import run, close, not_ready, explain                   # noqa: E402
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
try:
    from lab_support import NotSolved, load_artefact, load_table      # noqa: E402
    import numpy as np                                                # noqa: E402
    from scipy import stats                                           # noqa: E402
except ImportError as unready:
    not_ready(unready)

# The shared worked example: one departure, twelve seats, demand Poisson with
# mean ten, a passenger left standing priced at four. Module 3's measured.json
# and Module 4's carry the same entry, and this is the number both print.
SEATS, MEAN_DEMAND, PRICE_MISS = 12, 10.0, 4.0
MEASURED_MEAN_COST = 2.124


def exact_mean_cost(capacity, mean_demand, price):
    """The same sum the module measured: exact over the Poisson mass, no simulation."""
    demand = np.arange(0, int(mean_demand) + 200)
    return price * float((np.maximum(0, demand - capacity)
                          * stats.poisson.pmf(demand, mean_demand)).sum())


def part_a(lab):
    close(lab.threshold_from_costs(1.0, 1.0), 0.5, 1e-9,
          "equal costs must give one half — that is the only case where 0.5 is right")
    close(lab.threshold_from_costs(1.0, 4.0), 0.2, 1e-9,
          "a false negative four times worse gives 1/(1+4) = 0.2")
    close(lab.threshold_from_costs(3.0, 1.0), 0.75, 1e-9,
          "a false positive three times worse gives 3/(3+1) = 0.75")
    close(lab.threshold_from_costs(0.0, 5.0), 0.0, 1e-9,
          "a costless false positive gives zero — always say aboard")

    # The direction people get wrong: a dearer false negative LOWERS the cutoff.
    dear_miss = lab.threshold_from_costs(1.0, 9.0)
    assert dear_miss < 0.5, (
        f"with a false negative nine times dearer you returned {dear_miss:.3f}. When "
        "missing a passenger is the expensive mistake, you should say 'aboard' more "
        "readily, not less — the threshold goes down.")

    # Five named pairs are five entries in a table, and a table of five answers
    # is not a derivation. Sweep a spread of costs and insist one expression
    # holds across all of them — at which point the table would have to be the
    # expression written out.
    for cost_fp in (0.5, 1.0, 2.5, 7.0, 12.0):
        for cost_fn in (0.25, 1.0, 3.0, 8.0, 20.0):
            close(lab.threshold_from_costs(cost_fp, cost_fn),
                  cost_fp / (cost_fp + cost_fn), 1e-9,
                  f"threshold_from_costs({cost_fp}, {cost_fn}) — the cutoff is "
                  "C_fp / (C_fp + C_fn) for every pair of costs, not only the "
                  "round ones")


def scored_test_set():
    """The champion's probabilities on the later stretch of time, and the truth."""
    artefact = load_artefact("v1")
    table = load_table()
    test = table.iloc[int(len(table) * 0.7):]
    transform = artefact["transform"]
    prepared = np.column_stack([
        ((test[f].astype(float).fillna(transform["medians"][f]) - transform["means"][f])
         / transform["stds"][f]).to_numpy()
        for f in transform["features"]])
    probabilities = artefact["model"].predict_proba(prepared)[:, 1]
    truth = (test["label2"] == "IN").astype(int).to_numpy()
    return probabilities, truth


def part_b(lab, probabilities, truth):
    cost_fp, cost_fn = lab.COST_FALSE_POSITIVE, lab.COST_FALSE_NEGATIVE
    priced = lab.threshold_from_costs(cost_fp, cost_fn)

    at_priced = lab.realised_cost(probabilities, truth, priced, cost_fp, cost_fn)
    at_habit = lab.realised_cost(probabilities, truth, 0.5, cost_fp, cost_fn)

    assert at_priced >= 0, "a total cost cannot be negative"
    assert at_priced < at_habit, (
        f"the priced threshold {priced:.2f} cost {at_priced:,.0f} and the habitual 0.5 "
        f"cost {at_habit:,.0f}. The priced one should cost less on this data — check "
        "that realised_cost counts false positives and false negatives the right way "
        "round.")

    # A sanity anchor: at threshold 0, nothing is ever called 'not aboard',
    # so there can be no false negatives.
    everything_aboard = lab.realised_cost(probabilities, truth, 0.0, cost_fp, cost_fn)
    false_positives_only = int((truth == 0).sum()) * cost_fp
    close(everything_aboard, false_positives_only, 1e-6,
          "at threshold 0 every row is called aboard, so the cost is exactly the "
          "false positives")

    # The boundary, stated once and used everywhere: aboard when p is *at* the
    # threshold, not only above it. Two rows sitting exactly on the cutoff, both
    # not aboard, are therefore two false positives.
    on_the_line = lab.realised_cost(np.array([0.2, 0.2]), np.array([0, 0]), 0.2, 1.0, 4.0)
    close(on_the_line, 2.0, 1e-9,
          "two rows whose probability is exactly the threshold, neither aboard. This "
          "module decides aboard when p ≥ t — the same rule the derivation ends on and "
          "the one the slides, the figures and the other labs use — so both are false "
          "positives and the cost is 2. Using p > t gives 0 here")

    # And it must be the count of mistakes, not a rate: doubling the rows doubles
    # the bill.
    doubled = lab.realised_cost(np.concatenate([probabilities, probabilities]),
                                np.concatenate([truth, truth]), priced, cost_fp, cost_fn)
    close(doubled, 2 * at_priced, 1e-6,
          "the same rows twice cost twice as much. realised_cost is the total over the "
          "period, not an average per row — that is why it is not called an expectation")



# --------------------------------------------------------------------------
# Calibration
# --------------------------------------------------------------------------
# Two invented models, built so that the accurate one is the badly calibrated
# one. Nothing subtle: the first answers 0.6 to every request and is right about
# 88 rows in a hundred, so a threshold of one half calls everything aboard and is
# right 88 per cent of the time -- while the single band it occupies promises 0.6
# and delivers 0.88. The second answers 0.3 to half the rows and 0.7 to the
# other half, and those shares are exactly what happens, so it keeps its word
# everywhere and is right 70 per cent of the time.
CALIBRATION_ROWS = 1000


def accurate_but_uncalibrated():
    """Every request scored 0.6; 88 rows in a hundred are aboard."""
    scores = np.full(2 * CALIBRATION_ROWS, 0.6)
    truth = np.zeros(2 * CALIBRATION_ROWS, dtype=int)
    truth[:int(0.88 * len(truth))] = 1
    return scores, truth


def calibrated_but_less_accurate():
    """Half scored 0.3 with 30 per cent aboard, half scored 0.7 with 70 per cent."""
    scores = np.concatenate([np.full(CALIBRATION_ROWS, 0.3),
                             np.full(CALIBRATION_ROWS, 0.7)])
    truth = np.zeros(2 * CALIBRATION_ROWS, dtype=int)
    truth[:int(0.30 * CALIBRATION_ROWS)] = 1
    truth[CALIBRATION_ROWS:CALIBRATION_ROWS + int(0.70 * CALIBRATION_ROWS)] = 1
    return scores, truth


def reference_bands(scores, truth, edges, minimum_rows):
    """The check's own binning, so nothing here is compared against a slide."""
    scores, truth = np.asarray(scores, float), np.asarray(truth, float)
    bands = {}
    for index in range(len(edges) - 1):
        low, high = float(edges[index]), float(edges[index + 1])
        last = index == len(edges) - 2
        inside = (scores >= low) & (scores <= high if last else scores < high)
        if int(inside.sum()) < minimum_rows:
            continue
        bands[f"p{int(round(low * 100)):02d}_{int(round(high * 100)):02d}"] = {
            "rows": int(inside.sum()),
            "predicted": float(scores[inside].mean()),
            "actual": float(truth[inside].mean())}
    return bands


def accuracy_at(scores, truth, threshold=0.5):
    return float(((np.asarray(scores) >= threshold).astype(int)
                  == np.asarray(truth)).mean())


def part_d(lab, probabilities, truth):
    edges, floor = lab.CALIBRATION_EDGES, lab.CALIBRATION_MINIMUM_ROWS

    # 1. On the service's own output: the bands the deck's diagram draws.
    answer = lab.reliability(probabilities, truth, edges, floor)
    assert isinstance(answer, dict) and "bands" in answer and "largest_gap" in answer, (
        f"reliability() returned {answer!r}; it returns "
        "{'bands': {label: {'rows', 'predicted', 'actual'}}, 'largest_gap': float}")
    expected = reference_bands(probabilities, truth, edges, floor)
    assert set(answer["bands"]) == set(expected), explain(
        "m3:reliability:labels",
        f"your bands are {sorted(answer['bands'])} and the check's are {sorted(expected)}",
        "The label of a band is its two edges as whole percentages, 'p60_80' for 0.6 to "
        "0.8. A band holding fewer rows than the floor is dropped rather than reported: a "
        "point on a reliability diagram carries the same weight whether it rests on six "
        "rows or a thousand.")
    for label, band in expected.items():
        got = answer["bands"][label]
        assert int(got["rows"]) == band["rows"], explain(
            f"m3:reliability:rows:{label}",
            f"band {label} holds {got['rows']} rows and the check counted {band['rows']}",
            "The bands are half-open, [low, high), except the LAST, which is closed at the "
            "top. Left half-open all the way along it silently drops every probability of "
            "exactly one — and those are the rows this forest is most confident and most "
            "nearly right about.")
        close(float(got["predicted"]), band["predicted"], 1e-9,
              f"band {label}: the mean probability the model promised")
        close(float(got["actual"]), band["actual"], 1e-9,
              f"band {label}: the share of those rows that were actually aboard")
    close(float(answer["largest_gap"]),
          max(abs(b["predicted"] - b["actual"]) for b in expected.values()), 1e-9,
          "the largest distance from the diagonal, over the bands you kept")

    # 2. The two invented models. The point is the ORDER of the two properties.
    loud_scores, loud_truth = accurate_but_uncalibrated()
    honest_scores, honest_truth = calibrated_but_less_accurate()
    loud = lab.reliability(loud_scores, loud_truth, edges, floor)
    honest = lab.reliability(honest_scores, honest_truth, edges, floor)

    close(float(honest["largest_gap"]), 0.0, 1e-9, explain(
        "m3:reliability:calibrated",
        "the calibrated model was reported as having a gap",
        "It scores 0.3 on rows that are aboard three times in ten and 0.7 on rows that are "
        "aboard seven times in ten. Every band keeps its word exactly, so the largest "
        "distance from the diagonal is nought."))
    close(float(loud["largest_gap"]), 0.28, 1e-9, explain(
        "m3:reliability:uncalibrated",
        f"you reported a largest gap of {float(loud['largest_gap']):.4f} for the model that "
        "answers 0.6 to everything",
        "Every request is scored 0.6 and 88 rows in a hundred are aboard, so the one band "
        "it occupies promises 0.6 and delivers 0.88. That is a gap of 0.28 — and this is "
        "the MORE accurate of the two models."))

    # And the sentence the whole exercise exists for.
    assert accuracy_at(loud_scores, loud_truth) > accuracy_at(honest_scores, honest_truth), (
        "fixture error: the badly calibrated model is no longer the more accurate one")
    assert float(loud["largest_gap"]) > float(honest["largest_gap"]), explain(
        "m3:reliability:ordering",
        "the accurate model was not reported as the worse calibrated one",
        f"At a threshold of one half the first is right "
        f"{accuracy_at(loud_scores, loud_truth) * 100:.0f} per cent of the time and the "
        f"second {accuracy_at(honest_scores, honest_truth) * 100:.0f} per cent, and it is "
        "the first whose probabilities mean nothing. Accuracy and calibration are different "
        "properties of the same numbers, and a price multiplies a probability.")

    # 3. The two edge rules, each on its own.
    at_one = np.array([1.0] * (floor + 1) + [0.5] * (floor + 1))
    kept = lab.reliability(at_one, np.ones(len(at_one), dtype=int), edges, floor)
    assert "p80_100" in kept["bands"] and kept["bands"]["p80_100"]["rows"] == floor + 1, explain(
        "m3:reliability:closed-top",
        "a probability of exactly one was dropped",
        "The last band is closed at the top. This module's own calibration table lost 86 "
        "of 3,241 test rows to that bug, and they were the best-calibrated rows it had.")
    small = np.concatenate([np.full(floor - 1, 0.1), np.full(floor + 10, 0.5)])
    thin = lab.reliability(small, np.ones(len(small), dtype=int), edges, floor)
    assert "p00_20" not in thin["bands"], explain(
        "m3:reliability:floor",
        f"a band of {floor - 1} rows was reported, and the floor is {floor}",
        "A band below the floor is dropped rather than drawn. Plotted, it looks exactly "
        "like a band of a thousand rows and is read as a failure of the model rather than "
        "as a shortage of evidence.")


def part_c(lab):
    """Optional: the departure where the price of a miss is not a constant."""
    try:
        answer = lab.cost_at_mean_versus_mean_cost(SEATS, MEAN_DEMAND, PRICE_MISS)
    except NotSolved:
        print("  (optional) cost_at_mean_versus_mean_cost() not attempted — not graded. "
              "Write it and this check grades it too.")
        return

    assert answer is not None and len(answer) == 2, (
        f"cost_at_mean_versus_mean_cost returned {answer!r}; it returns two numbers, "
        "(cost at the mean demand, mean of the cost over the demand)")
    cost_at_mean, mean_cost = float(answer[0]), float(answer[1])

    close(cost_at_mean, PRICE_MISS * max(0.0, MEAN_DEMAND - SEATS), 1e-9,
          f"the cost computed at the average demand: {MEAN_DEMAND:.0f} passengers in "
          f"{SEATS} seats leaves nobody standing, so it is nought. That is the whole "
          "point of the example — the plan checked on the average looks perfect")
    assert mean_cost >= cost_at_mean, (
        f"you returned a mean cost of {mean_cost:.3f}, below the cost at the mean demand "
        f"({cost_at_mean:.3f}). For a convex cost the average of the costs is at least "
        "the cost of the average, never less (Jensen, 1906) — so this is arithmetic, not "
        "an unlucky sample. Check the direction of max(0, demand - capacity).")
    close(mean_cost, MEASURED_MEAN_COST, 1e-3,
          f"the mean cost of the shared {SEATS}-seat departure is {MEASURED_MEAN_COST} "
          "vehicle-hour equivalents — the number on Module 3's slide and in Module 4's "
          "proof. Sum the Poisson mass exactly rather than simulating; a simulation "
          "gives a different answer on every run and cannot be put on a slide")

    # A second capacity, so the answer is a computation and not the slide's number
    # typed in. At sixteen seats the shortfall is rare and the gap is small.
    other = lab.cost_at_mean_versus_mean_cost(16, MEAN_DEMAND, PRICE_MISS)
    close(float(other[1]), exact_mean_cost(16, MEAN_DEMAND, PRICE_MISS), 1e-6,
          "with sixteen seats the mean cost is a different number, and the function has "
          "to compute it rather than return what the slide says for twelve")
    assert float(other[1]) < mean_cost, (
        "more seats cannot cost more standing passengers")


def body(lab):
    part_a(lab)
    probabilities, truth = scored_test_set()
    part_b(lab, probabilities, truth)
    part_d(lab, probabilities, truth)
    part_c(lab)


run(3, "03_the_threshold", "threshold_from_costs", body)
