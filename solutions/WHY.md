# Why these solutions look like this

## Lab 1 — the gate is valuable because it is narrow

`promote_if_better` deliberately cannot see how large the candidate is, how new
it is, what it cost to train, or who is keen on it. It compares one number on
one agreed metric.

Every one of those other properties is an argument somebody will make in a
meeting. The gate exists so the argument was settled in the open, before there
was a candidate to be enthusiastic about. Our candidate has five times the
trees, two hundred and fifty times the size and four times the latency — and is
nine percentage points worse. A person eyeballing that table refuses it perhaps
three times in four. A gate refuses it every time.

Rollback is the release mechanism run backwards, not a separate emergency
procedure. A rollback path that works differently from the release path is one
nobody has tested.

The margin is where the gate's honesty lives. This module uses nought — a tie is
refused, because swapping a running model for no measured gain spends risk and
buys nothing. Module 5 uses 0.01, because it promotes on a schedule and a margin
stops the alias oscillating on noise. Neither number is principled by itself:
the principled margin is an interval around the metric, and Module 4 computes
one from the count of labels behind it (Wilson, 1927).

Part (b) does the same thing with MLflow, and the point of doing both is to see
which parts were the idea and which were the product. The idea: a run is a
record written as it happens, a signature is the schema of the model's door, and
a release is an alias moving. The product: a store other people can reach, a
permission model, and a lineage graph. `log_training_run` returns the registered
version rather than the run identifier, because identifiers are random and
nothing in this module ever compares one — the checks compare content.

Part (c) is a decision rather than a rule, and it is priced in a different unit
from part (a) on purpose. The gate compares the one number agreed before there
was a candidate; the verdict is what a person signs, and the unit it is signed in
is the one the operator pays in. So `promotion_verdict` never compares accuracies
to decide a release: a candidate that is more accurate and dearer at the priced
threshold is held, and if that feels wrong then the feeling is block three's
lesson arriving from a fourth direction.

Three things about the order of its tests are worth more than the code.

The health of the model *in service* is asked first, and it does not look at the
candidate at all. A service that has fallen below what it promised is not
repaired by releasing something new into it: the release path and the recovery
path would run at once, at the worst moment, and afterwards nobody could say
which of the two moved the number.

Being cheaper than what is running is not enough. The honest question is whether
the service earns its keep, and the comparison that answers it is against the
best policy that never asks a model — 1,663 vehicle-hour equivalents here
against the habit's 2,054. Quote the saving against the habit alone and you have
quoted the flattering half.

And the reason is graded as hard as the call, because a verdict without an
argument is a coin flip that the failure message would otherwise explain. Every
number in it has to be one the student measured; a sentence lifted off a slide
fails on the first figure, which is the habit this course exists to break.

## Lab 2 — validate before you predict

A model given nonsense returns a number. It cannot object; it has no way to know
the request was impossible. So an unchecked service answers every request
successfully, including the meaningless ones, and the only trace is a slightly
worse metric weeks downstream.

Two details worth keeping: exclude `bool` from "is a number", because in Python
`True` is an `int` and `{"speed": true}` otherwise reaches the model as 1.0
metres per second. And name the field in every complaint — a refusal that says
only "invalid request" is one the caller retries unchanged.

Provenance goes in every response, not only the failures. "Which model gave this
answer?" arrives months later about one request, and should be answerable from
the answer rather than by correlating timestamps against a deployment log.

The bounds are a decision, and the solution states it twice — once as two
constants and once in the refusal it prints. Module 1 measured this vehicle at
−3.361 to 3.555 metres per second; the contract accepts that range widened by
one metre per second on each side. An earlier version of this lab declared −5 to
30 "from the measurement", and thirty metres per second is one hundred and eight
kilometres per hour: a bound nothing can violate refuses nothing, which is
documentation rather than a contract. Bounding at the measured extreme is the
opposite mistake — the first slightly faster day is then refused — so the repair
is a margin you say out loud.

## Lab 3 — the threshold is a property of the consequences

    t* = C_fp / (C_fp + C_fn),  and the service decides aboard when p ≥ t*

The model is not in that expression. Two services with the identical model
should use different cutoffs if their mistakes cost different amounts, and a
service whose costs change should move its cutoff without retraining anything.

One half is not a neutral default. It is the specific claim that both mistakes
cost the same, made by somebody who did not check.

The boundary — at the threshold, not only above it — is immaterial for a
continuous score and is still chosen once, here, so that the stub, the checks,
the figures, the slides and the notebook cannot drift apart. It is written the
same way in all of them.

`realised_cost` is not called `expected_cost`, which is what it was called
before. It is a total over a test period that has happened, not an average over
outcomes that have not. A student who keeps the two words apart will not later
average a cost over a test set and present it as a forecast.

The optional `cost_at_mean_versus_mean_cost` is where the module's own
derivation stops being the whole story. Two constant prices make the expected
cost a straight line in the probability. Price a miss by its size — passengers
left standing, max(0, D − c) — and the line bends, and the cost of the average
day stops being the average cost of the days: nought against 2.124 vehicle-hour
equivalents on the shared twelve-seat departure. It is computed by exact Poisson
sums rather than simulated, because Module 4 proves the inequality on the same
numbers and two modules cannot share a number that moves with a seed.

## Lab 4 — the failure where nothing goes wrong

`prepare` loops over `transform["features"]`, never over the request's keys. A
model knows position one, position two, position three — not names. Iterate over
the caller's order and one day a client library serialises alphabetically,
signal strength lands where speed is expected, every request still succeeds, and
every answer is wrong.

There is no error to find. The service is healthy, the latency is fine, the
responses are well-formed. The only symptom is a metric drifting somewhere
downstream, and by then the cause is weeks behind you. That is training-serving
skew, and it is why the transform travels with the model as one artefact.

Report the 95th percentile, not the mean, by nearest rank so the number reported
is a duration that happened. Ninety-four requests at ten milliseconds and six at
a second give a mean of 69 milliseconds — a true number describing nobody.

Part (b) is the same cure bought from the platform. The signature recorded in
Lab 1 matches columns by name, restores the stored order, and refuses a column
it was not trained with, so a request with its keys reversed is answered
correctly and one with a renamed column is refused rather than guessed at. Both
cures are taught because they fail differently: discipline protects the batch
job somebody writes next year in another repository, and enforcement protects
you from your own discipline lapsing. A team with only the first has a
convention; a team with only the second has a rule it does not understand.

## Lab 2, again — present and null are two rules

An earlier version of this lab called the three signal strengths *optional* and
meant two things by the word at once. The service filled a missing one from the
stored median and answered; the registered model in Lab 4 refused the identical
request, because a signature is a set of columns and a column that is not there
cannot be reordered into place. One request, servable by one cure and refused by
the other, and neither file said so.

A missing **field** is a request the service cannot answer: the model was fitted
on four columns and there is nothing to put in the fourth position. A null
**value** is a measurement that was not made: the field is there, the caller is
saying it is empty, and the stored median stands in for it.

Which fields may be empty is a measurement and not a preference. On the training
day speed carries a value on 100.0 per cent of rows and the three signal
strengths on 11.3, 25.1 and 30.7 per cent, because a phone always knows how fast
it is moving and hears only the beacons in range. A contract refusing a null
signal strength would refuse most of the traffic; one accepting a null speed
would answer a question nobody asked, out of a median.

## Lab 3, again — calibration is a different property from accuracy

`reliability` is six lines and one uncomfortable result. Two decisions inside it
were got wrong here once. The last band is **closed** at the top: written as
`low <= p < high` all the way along, the table silently drops every probability
of exactly one, and those were 995 of this module's 3,241 test rows and the
best-calibrated ones it had. And a band below the floor of fifty rows is dropped
rather than drawn, because a point on a reliability diagram weighs the same on
the eye whether it rests on six rows or a thousand, and the eye reads the six as
a failure of the model rather than as a shortage of evidence.

The result the check insists on: a model answering 0.6 to every request, right 88
per cent of the time, has a gap of 0.28; a model answering 0.3 and 0.7 and
meaning both is right 70 per cent of the time with a gap of nought. The more
accurate of the two is the one whose probabilities are worthless. Ranking is what
accuracy and a threshold sweep use; calibration is what a *price* needs, because
a price multiplies a probability.

## Lab 4, again — cause it before you cure it

The bug in `cause_and_cure_skew` is one word: `for name in request` where the
cure says `for name in transform["features"]`. Every field is still scaled by its
own stored mean and standard deviation, so every number reaching the model is a
perfectly good number — it is simply in the wrong seat. That is why nothing
catches it. The range check passes, the type check passes, the null check passes,
the request returns 200 in the usual few milliseconds, and the only thing wrong
is the answer.

Counting the failures matters even though the count is nought. A student who has
written `failures: 0` beside `27.1 per cent of decisions changed` has met this
failure rather than been told about it, and the check measures the same share its
own way — by exchanging two columns of the prepared matrix — so the two routes to
the number have to agree.

## References

- Sculley, D. et al. (2015). *Hidden Technical Debt in Machine Learning Systems.* NeurIPS 28.
- Elkan, C. (2001). *The Foundations of Cost-Sensitive Learning.* IJCAI, 973–978.
- Provost, F. & Fawcett, T. (2013). *Data Science for Business*, ch. 7–8. O'Reilly.
- Zaharia, M. et al. (2018). *Accelerating the Machine Learning Lifecycle with MLflow.* IEEE Data Engineering Bulletin 41(4), 39–45.
- Schelter, S. et al. (2018). *On Challenges in Machine Learning Model Management.* IEEE Data Engineering Bulletin 41(4), 5–15.
- Breck, E. et al. (2019). *Data Validation for Machine Learning.* MLSys 1.
- Fielding, R., Nottingham, M. & Reschke, J. (2022). *HTTP Semantics.* Request for Comments 9110.
- Hyndman, R. J. & Fan, Y. (1996). *Sample Quantiles in Statistical Packages.* The American Statistician 50(4), 361–365.
- Dean, J. & Barroso, L. A. (2013). *The Tail at Scale.* Communications of the ACM 56(2), 74–80.
- Jensen, J. L. W. V. (1906). *Sur les fonctions convexes et les inégalités entre les valeurs moyennes.* Acta Mathematica 30, 175–193.
- Wilson, E. B. (1927). *Probable inference, the law of succession, and statistical inference.* JASA 22 — through Module 4.
