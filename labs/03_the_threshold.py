"""Lab 3 — What a mistake costs.

Why this lab exists: a model returns a probability and a service must return a
decision, so somebody has to choose the cutoff — and the cutoff almost everybody
uses, one half, is the specific claim that the two mistakes cost the same. You
derive the cutoff from the two prices instead, measure what it saves against the
habit and against using no model at all, and then meet the case the derivation
does not cover: a price that grows with the size of the miss.
Where it sits: Block 3 — "The threshold is a property of the costs, not of the
model", "Price against doing nothing, as well as against the habit" and "When the
price of a miss is not a constant", and the definition slides "Definition — the
cost-sensitive threshold", "Definition — realised cost of a threshold" and
"Definition — cost at the mean versus mean cost".
What the check grades: threshold_from_costs() returns C_fp / (C_fp + C_fn) for
all twenty-nine pairs of costs it is given, not only the round ones;
reliability() bins the model's own probabilities the way the deck's diagram does
and reports the largest distance from the diagonal, on the service's output and
on two invented models — one accurate and badly calibrated, one calibrated and
less accurate;
realised_cost() prices the derived threshold below one half on the model's own
output and prices a threshold of nought at exactly the number of rows that were
not aboard; and, if you write the optional cost_at_mean_versus_mean_cost(), the
mean cost it returns is at least the cost at the mean demand and equals the
module's measured value for the shared twelve-seat departure.
Needs: numpy, scipy.

Twenty-five minutes.

A model returns a probability. A service returns a decision. Turning one into
the other requires a cutoff, and the cutoff most people use is one half —
because it is the middle, not because anybody priced anything.

One half is correct in exactly one circumstance: when the two mistakes cost the
same. They almost never do.

Our service answers "is this passenger aboard?" Consider what each mistake does:

    a false positive   we say aboard, they are not. The vehicle is recorded
                       fuller than it is; dispatch may send a shuttle nobody
                       needs. Cost: one wasted vehicle-hour.

    a false negative   we say not aboard, they are. The vehicle is recorded
                       emptier than it is; a passenger is left at a stop
                       because the system believes there is room where there is
                       not. Cost: a person waiting in January, and a complaint.

If the second is four times worse than the first, one half is not a defensible
cutoff and nobody chose it.

What you write: threshold_from_costs(cost_false_positive, cost_false_negative).

    The arithmetic, from expected value. Decide "aboard" when the expected cost
    of doing so is no higher than the expected cost of not:

        p . 0 + (1-p) . C_fp   <=   p . C_fn + (1-p) . 0

    which rearranges to p >= C_fp / (C_fp + C_fn). That ratio is the threshold.
    Note what it does not contain: the model. The cutoff is a property of the
    consequences, not of the classifier (Elkan, 2001).

    Sanity checks you can do in your head: equal costs give one half; a false
    negative four times worse gives one fifth; a false positive that costs
    nothing gives zero, and the service should say "aboard" every time.

Then write: realised_cost(probabilities, truth, threshold, cost_fp, cost_fn).

    The word matters. This is not an expectation: it is the total the decisions
    that threshold produces actually cost on the rows you have, counted after
    the fact. An expected cost would be an average over a distribution of
    outcomes that has not happened yet. The check uses this one to confirm your
    derived threshold costs less than one half — measured, on the model's own
    output, rather than argued.

Then write: reliability(probabilities, truth, edges, minimum_rows).

    The derivation above assumed something without saying so: that when the
    model says 0.7, the thing happens about seven times in ten. That property is
    called **calibration**, and prices need it, because a price multiplies a
    probability. Measure whether this model has it.

    Sort the probabilities into bands between `edges`, and for each band report
    how many rows fell in it, the mean probability the model promised, and the
    share that were actually aboard. Report the largest distance between the two
    across the bands. That is the reliability diagram of block three, as a
    number.

    Two bands' worth of arithmetic, and one uncomfortable result: the check
    hands you a model that is more accurate than another and much worse
    calibrated. Accuracy and calibration are different properties of the same
    numbers, and a service that prices its decisions needs the second.

Then, optional and graded only if you write it:
cost_at_mean_versus_mean_cost(capacity, mean_demand, price_false_negative).

    The two prices above are constants, so the expected cost of a decision is a
    straight line in the probability and the threshold rule is the whole story.
    Now let the price of a miss grow with the size of the miss: one departure,
    `capacity` seats, demand D uncertain with mean `mean_demand`. The passengers
    left standing are max(0, D - capacity), and that hinge bends.

    Return two numbers: the cost computed at the average demand, and the average
    of the cost over the demand. They are not the same number, and the second is
    never the smaller — Jensen's inequality (Jensen, 1906), which Module 4
    proves and which reuses this exact departure.

The check passes when threshold_from_costs() returns C_fp / (C_fp + C_fn) for
every one of the twenty-nine pairs of costs it tries — not only the round ones —
when realised_cost() prices the derived threshold below the habitual one half
on the model's own output, and prices a threshold of zero at exactly the number
of rows that were not aboard, and when reliability() reproduces the check's own
bands on the service's output, puts a probability of exactly one in the last
band, drops a band too small to believe, and reports a gap of nought for a
calibrated model and a large one for an accurate model that is not calibrated. Write the optional function and it is graded too:
mean cost at least the cost at the mean, and the module's measured gap for the
twelve-seat departure to within a thousandth.
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from lab_support import NotSolved, load_artefact, load_table  # noqa: E402

LAB = 3

# What the two mistakes cost, in the same unit. Priced by the operator, not by
# the data scientist -- which is the point of the block.
COST_FALSE_POSITIVE = 1.0    # a shuttle sent that was not needed
COST_FALSE_NEGATIVE = 4.0    # a passenger left standing at a stop

# The departure Modules 3 and 4 share: one vehicle, twelve seats, demand
# uncertain around ten passengers. Poisson because it is the standard law for
# independent arrivals in a fixed window, and because a named law lets both
# modules compute the same number exactly rather than simulate it.
SEATS = 12
MEAN_DEMAND = 10.0

# The bands of the reliability diagram, and the smallest band worth believing.
# Equal width rather than equal count, so the picture is read off the axis
# rather than off a legend; half-open, [low, high), so that no row lands in two
# of them -- except the last, which is CLOSED at the top. Left half-open it
# excludes every probability of exactly one, and those are the rows this model
# is most confident and most nearly right about: dropping the best-calibrated
# band from a calibration table is the kind of quiet exclusion this course
# exists to teach against. Fifty rows is the floor: a band of six promises
# nothing anybody should believe, and plotting it as a point gives it the same
# visual weight as a band of a thousand.
CALIBRATION_EDGES = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
CALIBRATION_MINIMUM_ROWS = 50


def threshold_from_costs(cost_false_positive: float, cost_false_negative: float) -> float:
    """The probability at or above which "aboard" is the cheaper decision.

    Definition graded by the check:
        t* = C_fp / (C_fp + C_fn); decide aboard ⇔ p ≥ t*
        (Elkan, 2001, pp. 973-978). Choices: the boundary is decided in favour of
        "aboard" (p ≥ t*, not p > t*); for a continuous score the two rules
        differ on a set of probability nought, and this module picks one and
        uses it in the stub, the check, the slides and the figures. Slide:
        "Definition — the cost-sensitive threshold".
    Needs: no library — two costs in, one number out
    """
    # TODO: two costs in, one number out. No model involved.
    raise NotSolved("threshold_from_costs(cost_fp, cost_fn) still raises instead of "
                    "returning a threshold")


def realised_cost(probabilities, truth, threshold: float,
                  cost_false_positive: float, cost_false_negative: float) -> float:
    """What this cutoff cost on the rows you have — counted, not expected.

    Args:
        probabilities: predicted probability of "aboard", one per row.
        truth:         1 where actually aboard, 0 where not.

    Definition graded by the check:
        K(t) = C_fp · #{i : p_i ≥ t and y_i = 0} + C_fn · #{i : p_i < t and y_i = 1}
        (Elkan, 2001; Provost & Fawcett, 2013, ch. 7). Choices: the total over
        the test period, not an average and not an expectation; the decision rule
        is p ≥ t, the same one threshold_from_costs derives; correct decisions
        are priced at nought. Slide: "Definition — realised cost of a threshold".
    Needs: numpy
    """
    # TODO: decide, count the two kinds of mistake, price them.
    raise NotSolved("realised_cost(...) still raises instead of returning a cost")



def reliability(probabilities, truth, edges=CALIBRATION_EDGES,
                minimum_rows: int = CALIBRATION_MINIMUM_ROWS) -> dict:
    """What the model promised against what happened, band by band.

    Args:
        probabilities: predicted probability of "aboard", one per row.
        truth:         1 where actually aboard, 0 where not.
        edges:         the band boundaries, in order.
        minimum_rows:  bands holding fewer rows than this are dropped.

    Returns:
        {"bands": {label: {"rows": int, "predicted": float, "actual": float}},
         "largest_gap": float}

        The label of the band between low and high is "pLL_HH" -- the two edges
        as whole percentages, two digits each, so the band from 0.6 to 0.8 is
        "p60_80" and the first is "p00_20".

    Definition graded by the check:
        B_j = {i : e_j ≤ p_i < e_(j+1)}, the last band closed at the top; predicted_j = mean_{i ∈ B_j} p_i, actual_j = mean_{i ∈ B_j} y_i, kept when |B_j| ≥ r; largest gap = max_j |predicted_j − actual_j|
        (DeGroot & Fienberg, 1983; Niculescu-Mizil & Caruana, 2005). e_j are
        `edges` and r is `minimum_rows`. Choices: bands of equal width rather
        than equal count; half-open except the last, which is closed so that a
        probability of exactly one is not quietly dropped; a band below the floor
        is dropped rather than reported, because a handful of rows promises
        nothing. Slide: "Definition — reliability in bands, and the largest gap".
    Needs: numpy
    """
    # TODO: put each row in its band, average the promise and the outcome, and
    #       report the largest distance between them.
    raise NotSolved("reliability(probabilities, truth, edges, minimum_rows) still raises "
                    "instead of returning the bands and the largest gap")


def cost_at_mean_versus_mean_cost(capacity: int, mean_demand: float,
                                  price_false_negative: float):
    """Optional, and graded if you write it: the plan on the average against the average of the plans.

    Definition graded by the check:
        g(E[D]) = C_fn · max(0, E[D] − c) and E[g(D)] = C_fn · Σ_{d>c} (d − c) · P(D = d), with D ~ Poisson(λ)
        (Jensen, 1906; Module 4 block two proves the inequality). Choices: demand
        is Poisson with mean λ = mean_demand, so the sum is exact and no seed is
        involved; only the shortfall is priced here, because the empty seat is
        the other module's half of the example. Slide: "Definition — cost at the
        mean versus mean cost".
    Needs: scipy, numpy
    """
    # TODO: return (cost at the mean demand, mean of the cost over the demand).
    raise NotSolved("cost_at_mean_versus_mean_cost(capacity, mean_demand, "
                    "price_false_negative) still raises instead of returning two costs")


if __name__ == "__main__":
    priced = threshold_from_costs(COST_FALSE_POSITIVE, COST_FALSE_NEGATIVE)
    print(f"priced threshold: {priced:.3f}   (the habit: 0.500)")
