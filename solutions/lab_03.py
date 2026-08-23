"""Lab 3, solved — with the reasoning, not only the code.

Run it:  python3 solutions/lab_03.py        (or  python3 labs/03_the_threshold.py  after apply.py)
It derives the threshold from the two prices, measures what it costs against the
habit and against the two policies that use no model at all, draws the
cost-against-threshold curve with those three policies marked, and closes on the
departure where the price of a miss is not a constant.
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from lab_support import NotSolved, load_artefact, load_table  # noqa: E402

LAB = 3

COST_FALSE_POSITIVE = 1.0
COST_FALSE_NEGATIVE = 4.0

SEATS = 12
MEAN_DEMAND = 10.0

CALIBRATION_EDGES = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
CALIBRATION_MINIMUM_ROWS = 50


def threshold_from_costs(cost_false_positive: float, cost_false_negative: float) -> float:
    """The cutoff follows from the consequences, not from the classifier.

    Definition graded by the check:
        t* = C_fp / (C_fp + C_fn); decide aboard ⇔ p ≥ t*
        (Elkan, 2001, pp. 973-978). Slide: "Definition — the cost-sensitive threshold".

    Decide "aboard" when the expected cost of doing so is no higher than the
    expected cost of not. With p the probability of being aboard:

        say aboard      : correct costs nothing, wrong costs C_fp, and it is
                          wrong with probability (1 - p)     ->  (1 - p) . C_fp
        say not aboard  : wrong costs C_fn, with probability p  ->  p . C_fn

    Say aboard when (1 - p) . C_fp <= p . C_fn, which rearranges to

        p >= C_fp / (C_fp + C_fn)

    Note what is absent from that expression: the model. The threshold is a
    property of the operator's consequences (Elkan, 2001). Two services using
    the identical model should use different cutoffs if their mistakes cost
    different amounts, and a service whose costs change should move its cutoff
    without retraining anything.

    Here a passenger left standing costs four times a wasted vehicle-hour, so
    the cutoff is 1 / 5 = 0.2, not 0.5. We are willing to be wrong about "aboard"
    four times as often, because that mistake is a quarter as expensive.

    One half is not a neutral default. It is the specific claim that both
    mistakes cost the same, made by someone who did not check.

    The boundary itself, p exactly equal to t*, is a choice with no consequence
    and it still has to be made once: at equality the two expected costs are the
    same number, so either answer is defensible. This module says "aboard", and
    the stub, the checks, the figures and the slides all say it.
    """
    total = cost_false_positive + cost_false_negative
    if total <= 0:
        raise ValueError("costs must not both be zero")
    return cost_false_positive / total


def realised_cost(probabilities, truth, threshold: float,
                  cost_false_positive: float, cost_false_negative: float) -> float:
    """What this cutoff cost, counted rather than argued.

    Definition graded by the check:
        K(t) = C_fp · #{i : p_i ≥ t and y_i = 0} + C_fn · #{i : p_i < t and y_i = 1}
        (Elkan, 2001; Provost & Fawcett, 2013, ch. 7). Slide: "Definition —
        realised cost of a threshold".

    The name is the lesson. The derivation above is a claim about expectations,
    made before anything happens. This is the bill afterwards: the decisions this
    threshold produced on the rows we have, priced. It was called expected_cost
    in an earlier version of this lab, and it was never an expectation -- it is a
    total over a finite test period, and a student who reads "expected" learns the
    wrong word for a quantity they can see.

    Measuring it is the only way to know whether the costs you were given
    describe the world you are in.
    """
    probabilities = np.asarray(probabilities, dtype=float)
    truth = np.asarray(truth).astype(int)
    said_aboard = probabilities >= threshold

    false_positives = int((said_aboard & (truth == 0)).sum())
    false_negatives = int((~said_aboard & (truth == 1)).sum())
    return (false_positives * cost_false_positive
            + false_negatives * cost_false_negative)



def reliability(probabilities, truth, edges=CALIBRATION_EDGES,
                minimum_rows: int = CALIBRATION_MINIMUM_ROWS) -> dict:
    """The promise against the outcome, band by band, and the worst distance between them.

    Definition graded by the check:
        B_j = {i : e_j ≤ p_i < e_(j+1)}, the last band closed at the top; predicted_j = mean_{i ∈ B_j} p_i, actual_j = mean_{i ∈ B_j} y_i, kept when |B_j| ≥ r; largest gap = max_j |predicted_j − actual_j|
        (DeGroot & Fienberg, 1983; Niculescu-Mizil & Caruana, 2005). Slide:
        "Definition — reliability in bands, and the largest gap".

    Two lines of it are decisions rather than arithmetic, and both were got
    wrong here once.

    **The last band is closed at the top.** Written as `low <= p < high` all the
    way along, the table silently drops every probability of exactly one -- and
    those are the rows a forest is most confident and most nearly right about.
    Excluding the best-calibrated rows from a calibration table is not a
    rounding decision.

    **A band too small to believe is dropped rather than drawn.** A point on a
    reliability diagram carries the same visual weight whether it rests on six
    rows or a thousand, and the eye reads the six as a failure of the model
    rather than as a shortage of evidence.

    And the property itself is worth stating in the room's own currency: take
    every request the model scored 0.7 and count how many were aboard. A
    calibrated model gives you seven in ten. Ranking and calibration are
    different virtues -- a model can order every request perfectly and still
    promise numbers that mean nothing, which is why a threshold derived from
    PRICES is a starting point on an uncalibrated model rather than a guarantee
    (Niculescu-Mizil & Caruana, 2005).
    """
    scores = np.asarray(probabilities, dtype=float)
    outcomes = np.asarray(truth, dtype=float)
    bands, gaps = {}, []
    for index in range(len(edges) - 1):
        low, high = float(edges[index]), float(edges[index + 1])
        last = index == len(edges) - 2
        inside = (scores >= low) & (scores <= high if last else scores < high)
        if int(inside.sum()) < minimum_rows:
            continue
        promised, happened = float(scores[inside].mean()), float(outcomes[inside].mean())
        bands[f"p{int(round(low * 100)):02d}_{int(round(high * 100)):02d}"] = {
            "rows": int(inside.sum()), "predicted": promised, "actual": happened}
        gaps.append(abs(promised - happened))
    return {"bands": bands, "largest_gap": max(gaps) if gaps else 0.0}


def cost_at_mean_versus_mean_cost(capacity: int, mean_demand: float,
                                  price_false_negative: float):
    """The plan made on the average demand, against the average of the plans.

    Definition graded by the check:
        g(E[D]) = C_fn · max(0, E[D] − c) and E[g(D)] = C_fn · Σ_{d>c} (d − c) · P(D = d), with D ~ Poisson(λ)
        (Jensen, 1906; Module 4 block two proves the inequality). Slide:
        "Definition — cost at the mean versus mean cost".

    Everything above this function priced a mistake at a constant: a wasted
    vehicle-hour is one, a passenger left standing is four, whatever the day
    looks like. Under two constants the expected cost of a decision is a straight
    line in the probability, and the threshold rule is the whole story.

    Let the price grow with the size of the miss and the story changes. Twelve
    seats, demand around ten: at exactly ten passengers nobody is left behind and
    the cost is nought, so a plan checked against the average demand looks
    perfect. Average the cost over the days instead and it is not nought, because
    the days with sixteen passengers cost far more than the days with four save.
    The hinge max(0, D - c) bends upwards, and for a bending cost the average of
    the costs is at least the cost of the average (Jensen, 1906).

    Exactly, not by simulation: demand is Poisson, so the sum over the mass
    function is a closed computation that gives every machine the same number.
    """
    demand = np.arange(0, int(mean_demand) + 200)          # the tail past 200 above the mean is below 1e-300
    mass = stats.poisson.pmf(demand, mean_demand)
    cost_at_mean = price_false_negative * max(0.0, mean_demand - capacity)
    mean_cost = price_false_negative * float(
        (np.maximum(0, demand - capacity) * mass).sum())
    return cost_at_mean, mean_cost


if __name__ == "__main__":
    import plotly.graph_objects as go
    import pandas as pd
    from _narrate import narrator, show_table, save_figure

    say = narrator(LAB)
    say.info("Lab 3 — the cutoff comes from the two prices, and then it is measured "
             "against the habit and against using no model at all")

    artefact = load_artefact("v1")
    transform = artefact["transform"]
    table = load_table()
    test = table.iloc[int(len(table) * 0.7):]
    say.info("loaded the champion (v1) and the table: %d rows, generated (seed 20200122); "
             "the later 30 per cent, %d rows, is the test period — split by time, because "
             "a random split would let the model see the future", len(table), len(test))

    prepared = np.column_stack([
        ((test[f].astype(float).fillna(transform["medians"][f]) - transform["means"][f])
         / transform["stds"][f]).to_numpy() for f in transform["features"]])
    probabilities = artefact["model"].predict_proba(prepared)[:, 1]
    truth = (test["label2"] == "IN").astype(int).to_numpy()
    say.info("scored the test period: %d rows, %d recorded aboard, %d not",
             len(truth), int(truth.sum()), int((truth == 0).sum()))

    priced = threshold_from_costs(COST_FALSE_POSITIVE, COST_FALSE_NEGATIVE)
    say.info("prices from the operator: a false positive costs %.0f wasted vehicle-hour, a "
             "false negative %.0f — so the cutoff is C_fp / (C_fp + C_fn) = %.3f, and the "
             "model never entered that arithmetic", COST_FALSE_POSITIVE,
             COST_FALSE_NEGATIVE, priced)

    at_priced = realised_cost(probabilities, truth, priced,
                              COST_FALSE_POSITIVE, COST_FALSE_NEGATIVE)
    at_half = realised_cost(probabilities, truth, 0.5,
                            COST_FALSE_POSITIVE, COST_FALSE_NEGATIVE)
    always_aboard = realised_cost(probabilities, truth, 0.0,
                                  COST_FALSE_POSITIVE, COST_FALSE_NEGATIVE)
    always_not = realised_cost(probabilities, truth, 1.01,
                               COST_FALSE_POSITIVE, COST_FALSE_NEGATIVE)
    say.info("realised cost at the priced threshold %.2f: %.0f (vehicle-hour equivalents, "
             "on the test period)", priced, at_priced)
    say.info("realised cost at the habitual 0.50: %.0f — the priced cutoff saves %.1f per "
             "cent against the habit", at_half, (1 - at_priced / at_half) * 100)
    say.info("and against no model at all: say aboard to everybody %.0f, say not aboard to "
             "everybody %.0f — the floor is %.0f, so the model plus its threshold saves "
             "%.1f per cent against the best thing you could do without it", always_aboard,
             always_not, min(always_aboard, always_not),
             (1 - at_priced / min(always_aboard, always_not)) * 100)
    say.info("note which comparison flatters: one half is dearer than answering aboard "
             "without asking the model, so a saving quoted only against the habit hides "
             "that the habit was worse than doing nothing")

    show_table(pd.DataFrame({
        "policy": ["priced threshold", "the habit, one half", "always aboard, no model",
                   "always not aboard, no model"],
        "threshold": [round(priced, 3), 0.5, 0.0, float("nan")],
        "realised cost": [at_priced, at_half, always_aboard, always_not],
        "accuracy": [float(((probabilities >= t).astype(int) == truth).mean())
                     for t in (priced, 0.5, 0.0, 1.01)]}),
        "four policies, priced", logger=say)
    say.info("accuracy disagrees with cost on purpose: it weighs both mistakes the same, "
             "which is the assumption this block spent its time rejecting")

    # Calibration: the assumption the derivation made without saying so.
    diagram = reliability(probabilities, truth, CALIBRATION_EDGES, CALIBRATION_MINIMUM_ROWS)
    show_table(pd.DataFrame([{"band": label, **band}
                             for label, band in diagram["bands"].items()]),
               "promised against happened, by band (bands under "
               f"{CALIBRATION_MINIMUM_ROWS} rows dropped)", logger=say)
    say.info("largest distance from the diagonal: %.3f — this model over-predicts almost "
             "everywhere, so the derived threshold is a principled starting point rather "
             "than a guaranteed optimum (Niculescu-Mizil & Caruana, 2005)",
             diagram["largest_gap"])
    say.info("and the pair the check grades, because accuracy and calibration are different "
             "properties of the same numbers:")
    loud = np.full(2000, 0.6)
    loud_truth = np.zeros(2000, dtype=int); loud_truth[:1760] = 1
    quiet = np.concatenate([np.full(1000, 0.3), np.full(1000, 0.7)])
    quiet_truth = np.zeros(2000, dtype=int)
    quiet_truth[:300] = 1; quiet_truth[1000:1700] = 1
    for name, scores, outcomes in (("answers 0.6 to everything", loud, loud_truth),
                                   ("answers 0.3 and 0.7, and means it", quiet, quiet_truth)):
        measured = reliability(scores, outcomes, CALIBRATION_EDGES, CALIBRATION_MINIMUM_ROWS)
        say.info("  %-34s accuracy at one half %.2f, largest gap %.2f", name,
                 float(((scores >= 0.5).astype(int) == outcomes).mean()),
                 measured["largest_gap"])
    say.info("the more accurate of those two is the one whose probabilities mean nothing, "
             "and a price multiplies a probability")

    fig = go.Figure()
    labels = list(diagram["bands"])
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash",
                             color="#52514E"), name="perfect calibration"))
    fig.add_trace(go.Scatter(
        x=[diagram["bands"][b]["predicted"] for b in labels],
        y=[diagram["bands"][b]["actual"] for b in labels], mode="markers+text",
        text=labels, textposition="top center", name="this service",
        marker=dict(color="#2A78D6",
                    size=[8 + 22 * (diagram["bands"][b]["rows"] / max(
                        diagram["bands"][x]["rows"] for x in labels)) for b in labels])))
    fig.update_layout(template="plotly_white",
                      title="Promised against happened, band by band",
                      xaxis_title="mean probability promised (dimensionless)",
                      yaxis_title="share actually aboard (dimensionless)")
    fig.update_xaxes(range=[0, 1]); fig.update_yaxes(range=[0, 1])
    save_figure(fig, "reliability_bands", LAB, logger=say)

    thresholds = np.linspace(0.0, 1.0, 201)
    costs = [realised_cost(probabilities, truth, t, COST_FALSE_POSITIVE, COST_FALSE_NEGATIVE)
             for t in thresholds]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=thresholds, y=costs, mode="lines", name="the model, at each threshold",
                             line=dict(color="#2A78D6", width=2)))
    fig.add_hline(y=always_aboard, line_dash="dash", line_color="#E07B39",
                  annotation_text=f"always aboard, no model: {always_aboard:,.0f}",
                  annotation_position="top right")
    for value, colour, name in ((priced, "#E07B39", "priced"), (0.5, "#52514E", "the habit")):
        here = realised_cost(probabilities, truth, value, COST_FALSE_POSITIVE, COST_FALSE_NEGATIVE)
        fig.add_trace(go.Scatter(x=[value], y=[here], mode="markers+text",
                                 text=[f"{name} {value:.2f} → {here:,.0f}"],
                                 textposition="top center", marker=dict(size=12, color=colour),
                                 name=f"{name} ({value:.2f})"))
    fig.update_layout(template="plotly_white",
                      title="One half is a claim that both mistakes cost the same",
                      xaxis_title="threshold on the predicted probability (decide aboard when p ≥ t)",
                      yaxis_title="realised cost of the decisions (vehicle-hour equivalents)")
    save_figure(fig, "cost_by_threshold", LAB, logger=say)

    cost_at_mean, mean_cost = cost_at_mean_versus_mean_cost(SEATS, MEAN_DEMAND,
                                                            COST_FALSE_NEGATIVE)
    say.info("the departure where the price of a miss is not a constant: %d seats, demand "
             "Poisson with mean %.0f, a passenger left standing priced at %.0f",
             SEATS, MEAN_DEMAND, COST_FALSE_NEGATIVE)
    say.info("cost computed at the average demand: %.3f — at ten passengers in twelve seats "
             "nobody is left behind, so the plan looks perfect", cost_at_mean)
    say.info("average of the cost over the demand: %.3f vehicle-hour equivalents — the gap "
             "is %.3f, and it is not noise: max(0, D - c) bends, so the average of the costs "
             "is at least the cost of the average (Jensen, 1906; Module 4 proves it)",
             mean_cost, mean_cost - cost_at_mean)
    say.info("the closure: the capacity that balances the two prices is the "
             "C_fn / (C_fp + C_fn) = %.1f quantile of demand, %d seats — the same ratio as "
             "this lab's threshold, read as a demand quantile rather than a probability",
             COST_FALSE_NEGATIVE / (COST_FALSE_POSITIVE + COST_FALSE_NEGATIVE),
             int(stats.poisson.ppf(COST_FALSE_NEGATIVE
                                   / (COST_FALSE_POSITIVE + COST_FALSE_NEGATIVE), MEAN_DEMAND)))

    say.info("what the check grades: the ratio C_fp / (C_fp + C_fn) at twenty-nine pairs of "
             "costs, the realised cost of the priced threshold below that of one half, the "
             "cost at threshold nought equal to the not-aboard rows, the reliability bands "
             "against its own arithmetic and against two invented models — one accurate "
             "and badly calibrated, one calibrated and less accurate — and, if written, the "
             "mean cost at least the cost at the mean and equal to the module's measured "
             "%.3f", mean_cost)
