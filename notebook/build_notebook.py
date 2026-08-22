#!/usr/bin/env python3
"""Build Module 3's demonstration notebook, then execute it.

    python "Module 3/notebook/build_notebook.py"

Unlike Modules 1 and 2, this notebook opens no archive file at all. It runs
entirely on the module's own service — the stand-in model trained from Module 2's
generated table, and the local MLflow store beside it — because everything blocks
one to four demonstrate is about the service, not about the data.

That means it executes anywhere, including in continuous integration, and every
number it prints is `generated`.

It is executed here, with the working directory set to `Module 3/exercises` —
which is where `tools/check_notebook.py --run` sets it too — and the outputs are
saved into the notebook, so what is committed is what ran.
"""
from __future__ import annotations

import sys
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

HERE = Path(__file__).resolve().parent
EXERCISES = HERE.parent / "exercises"
OUTPUT = HERE / "Module3_demonstration.ipynb"
MARKDOWN, CODE = "markdown", "code"

CELLS = [
(MARKDOWN, """# Module 3 — Deploying a model as a service

**Data Mining and Analysis (course code CE3) · Aalborg University, Copenhagen**

Four blocks: record and release, the contract at the door, what a mistake costs,
and skew.

> **The model here is a stand-in.** The four modelling sessions between Module 2
> and Module 3 produce the model this course would rather serve. Until that
> artefact is handed over, this notebook uses two small forests trained from
> Module 2's table in seven seconds. Every number below is therefore `generated`,
> not `archive` — and when the real artefact arrives, it drops into the registry
> and nothing here changes. That substitutability is block one's whole point.

Every concept the labs grade appears here in the same three places as on the
slides and in the stubs: the definition, the formula, and the source."""),

(MARKDOWN, """## Hook

A model scores 0.82 in a notebook. Six months later somebody emails: *your system
said the shuttle was full at 08:40 on 14 March, and it was not.*

What do you need to have built, back when you deployed, to answer that?"""),

(CODE, '''import json, math, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

# Runs whether the working directory is the module's exercises directory (which
# is where tools/check_notebook.py --run puts it) or the repository root.
EXERCISES = next(p for p in (Path.cwd(), Path.cwd() / "Module 3" / "exercises",
                             Path.cwd().parent / "exercises")
                 if (p / "lab_support.py").exists())
FIGURES = EXERCISES.parent / "notebook" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(EXERCISES))
sys.path.insert(0, str(EXERCISES / "data"))

from service.models import (load_artefact, load_metrics, build_table, apply_transform,
                            open_store, load_registered, REGISTERED_MODEL, CHAMPION,
                            FEATURES, EXPERIMENT)

BLUE, ORANGE, GREY, RED = "#2A78D6", "#E07B39", "#52514E", "#C0392B"

def show(fig, name):
    """Inline for the reader, and a portable copy under notebook/figures/."""
    fig.update_layout(template="plotly_white", width=980, height=520,
                      margin=dict(l=70, r=40, t=60, b=60))
    fig.write_image(str(FIGURES / f"{name}.png"), scale=2)
    fig.show()

metrics = load_metrics()
print(f"{'version':8} {'kind':24} {'accuracy':>9} {'megabytes':>11} {'trees':>7}")
for version, facts in metrics.items():
    model = load_artefact(version)["model"]
    print(f"{version:8} {facts['kind']:24} {facts['accuracy']:9.4f} "
          f"{facts['size_bytes']/1e6:11.2f} {len(model.estimators_):7}")'''),

(MARKDOWN, """## Core Concept

### Release is moving a pointer

The service loads *the approved model*, never `model_v7.pkl`. The registry says
what "approved" currently means. Releasing changes one entry; rolling back
changes it back. Three things follow cheaply: instant rollback, an audit trail,
and service code that never mentions a version number.

**Definition — release by indirection.** The service loads the artefact named by
the registry's approved entry rather than a file name, so releasing is writing a
new value into that entry and rolling back is stepping the entry back.

    serve(name) = artefact[approved]; promote: approved ← candidate,
    history ← history + [candidate]; rollback: history ← history[:−1],
    approved ← history[−1]

*(Zaharia et al., 2018; Schelter et al., 2018)*

**Definition — the gate.** Promote a candidate only when it beats the model in
service on the agreed metric, on the agreed data, by at least an agreed margin.

    promote(candidate) ⇔ m(candidate) > m(approved) + δ,
    with δ = 0 here (a tie is refused) and δ = 0.01 in Module 5

The margin is a choice, not a fact. The principled one comes from how uncertain
the metric is — Module 4's interval on the label count (Wilson, 1927)."""),

(CODE, '''registry = {"approved": "v1", "history": ["v1"]}
METRIC = "accuracy"

def promote_if_better(registry, candidate, metrics, margin=0.0):
    approved = registry["approved"]
    if metrics[candidate][METRIC] <= metrics[approved][METRIC] + margin:
        return False, (f"candidate {candidate} scored {metrics[candidate][METRIC]:.4f} "
                       f"against approved {approved} at {metrics[approved][METRIC]:.4f}")
    registry["approved"] = candidate
    registry["history"].append(candidate)
    return True, f"promoted {candidate}"

size_ratio = metrics["v2"]["size_bytes"] / metrics["v1"]["size_bytes"]
print(f"the candidate is {size_ratio:.0f} times the size of the approved model")
print("promote_if_better ->", promote_if_better(registry, "v2", metrics))
print("approved is still:", registry["approved"])
print("\\nand a tie, with the margin at nought:")
tie = {"v1": metrics["v1"], "v2": dict(metrics["v2"], accuracy=metrics["v1"]["accuracy"])}
print("  ", promote_if_better({"approved": "v1", "history": ["v1"]}, "v2", tie))'''),

(MARKDOWN, """The candidate is 258 times the size, five times the trees, and nine percentage
points worse on data from a later stretch of time. Extra capacity went into
memorising the training period.

The gate is valuable precisely because it cannot see any of the persuasive
things — not the size, not the novelty, not who is keen on it. It compares one
number, agreed in advance.

### The same idea as industry keeps it

`registry.json` above is fifty lines. Beside it this module keeps the same idea
in MLflow: one local store (a single database file, no server), one run per
training with its parameters, metrics and environment, and a **model signature**
recorded with the artefact.

**Definition — model signature.** A schema recorded with the model at logging
time: the name and type of every input column, in the order the model was fitted
on. A request is accepted only if every declared name is present with the
declared type, and the platform restores the stored order before the model sees
the row.

    S = (name_i, type_i)_{i=1..k}, recorded with the model;
    accept(x) ⇔ ∀i: name_i ∈ columns(x) ∧ type(x[name_i]) = type_i;
    input = (x[name_1], …, x[name_k])

*(Zaharia et al., 2018)*

**Definition — release by alias.** Release through a registry is the move of one
alias from the version it names to another version of the same registered model.

    alias: name@alias → version; promote: alias[champion] ← version;
    rollback: alias[champion] ← previous version; serve(models:/name@champion)"""),

(CODE, '''client = open_store()                      # the module's own store: exercises/mlruns.db
experiment = client.get_experiment_by_name(EXPERIMENT)
runs = {r.info.run_name: r for r in client.search_runs([experiment.experiment_id])}

print("what each training run recorded (identifiers are random and never compared):")
for name in sorted(runs):
    run = runs[name]
    print(f"  run {name}: accuracy {run.data.metrics['accuracy']:.4f}, "
          f"{run.data.params['n_estimators']} trees, depth {run.data.params['max_depth']}, "
          f"seed {run.data.params['random_state']}")
    print(f"           data window {run.data.params['data_window']}")
    print(f"           checksum    {run.data.params['data_checksum'][:16]}...")

named = client.get_model_version_by_alias(REGISTERED_MODEL, CHAMPION)
print(f"\\nalias '{CHAMPION}' names version {named.version} of '{REGISTERED_MODEL}'")
champion = load_registered(CHAMPION)
print("signature, in order:", champion.metadata.get_input_schema().input_names())
print("required:          ", champion.metadata.get_input_schema().required_input_names())'''),

(MARKDOWN, """## Worked Example

### The contract at the door

**Definition — the data contract at the door.** A declaration of what the service
accepts — for every field, whether it must be present, whether it may be empty,
its type, its permitted range and its unit — enforced before the model is asked.

    accept(x) ⇔ ∀f: (required_f ⇒ f ∈ x) ∧ (f ∈ x ∧ x[f] = null ⇒ nullable_f) ∧
    (f ∈ x ∧ x[f] ≠ null ⇒ type(x[f]) = type_f ∧ min_f ≤ x[f] ≤ max_f);
    otherwise the refusal names every f that failed

*(Breck et al., 2019; Fielding, Nottingham & Reschke, 2022)*

Present and null are two rules and not one, and conflating them is what this
module shipped before. Every field must be **present**, because the model's door
is four columns wide and the signature above refuses a request carrying three —
watch it do so further down. Only the three signal strengths may be present and
**null**, because a phone hears whichever beacons are in range: on the training
day speed carries a value on 100.0 per cent of the rows and the three signal
strengths on 11.3, 25.1 and 30.7 per cent. A null is a measurement that was not
made, and the stored median stands in for it.

The speed bounds are Module 1's measured range for this vehicle, −3.361 to 3.555
metres per second, widened by a stated margin of one metre per second on each
side. A bound nothing can violate is documentation, not a contract."""),

(CODE, '''CONTRACT = {
    "speed": {"required": True, "nullable": False, "type": "number",
              "min": -4.361, "max": 4.555, "units": "metres per second, signed"},
    "rssi1": {"required": True, "nullable": True, "type": "number",
              "min": -120.0, "max": 0.0, "units": "decibel-milliwatts"},
    "rssi2": {"required": True, "nullable": True, "type": "number",
              "min": -120.0, "max": 0.0, "units": "decibel-milliwatts"},
    "rssiC": {"required": True, "nullable": True, "type": "number",
              "min": -120.0, "max": 0.0, "units": "decibel-milliwatts"},
}

def is_number(value):
    # A boolean is an integer in Python, so `true` would arrive as one metre per second.
    return isinstance(value, (int, float)) and not isinstance(value, bool)

def validate(request, contract=CONTRACT):
    complaints = []
    for field, rule in contract.items():
        if field not in request:
            if rule["required"]:
                complaints.append(f"{field}: required, and the field is absent")
            continue
        value = request[field]
        if value is None:
            # Present and empty: allowed only where the measurement is one that
            # is not always made, and then the stored median stands in for it.
            if not rule["nullable"]:
                complaints.append(f"{field}: present but null, and this field is "
                                  "never empty in the data the model was fitted on")
            continue
        if not is_number(value):
            complaints.append(f"{field}: expected a number in {rule['units']}, "
                              f"got {type(value).__name__}"); continue
        if value < rule["min"]:
            complaints.append(f"{field}: {value} is below the minimum {rule['min']}")
        if value > rule["max"]:
            complaints.append(f"{field}: {value} is above the maximum {rule['max']}")
    return complaints

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
refusals = pd.DataFrame([
    {"what was wrong": what, "status": 422, "complaint": validate(request)[0]}
    for what, request in impossible])
print(refusals.to_string(index=False))
print(f"\\nthe valid request draws {len(validate(good))} complaints, "
      "and only then is the model asked")
print(f"rssi1 absent draws {len(validate({k: v for k, v in good.items() if k != 'rssi1'}))} "
      f"complaint(s); rssi1 present and null draws "
      f"{len(validate({**good, 'rssi1': None}))} — the distinction the platform's "
      "signature enforces further down")'''),

(MARKDOWN, """### A probability is not a decision

**Definition — the cost-sensitive threshold.** The probability at which the two
decisions cost the same in expectation: the price of a false positive divided by
the sum of the two prices. The classifier appears nowhere in it.

    t* = C_fp / (C_fp + C_fn); decide aboard ⇔ p ≥ t*

*(Elkan, 2001, pp. 973–978; Provost & Fawcett, 2013, ch. 7)*

**Definition — realised cost of a threshold.** The total the decisions produced
by a threshold actually cost on the rows you have. A bill, not an expectation.

    K(t) = C_fp · #{i : p_i ≥ t and y_i = 0} + C_fn · #{i : p_i < t and y_i = 1}"""),

(CODE, '''artefact = load_artefact("v1")
table = build_table()
test = table.iloc[int(len(table) * 0.7):]
prepared = apply_transform(test, artefact["transform"]).to_numpy()
probabilities = artefact["model"].predict_proba(prepared)[:, 1]
truth = (test["label2"] == "IN").astype(int).to_numpy()

COST_FP, COST_FN = 1.0, 4.0     # a wasted vehicle-hour; a passenger left standing

def realised_cost(threshold):
    said = probabilities >= threshold                 # aboard at or above the threshold
    return (int((said & (truth == 0)).sum()) * COST_FP
            + int(((~said) & (truth == 1)).sum()) * COST_FN)

priced = COST_FP / (COST_FP + COST_FN)
always_aboard, always_not = realised_cost(0.0), realised_cost(1.01)
print(f"test period: {len(truth):,} rows, {int(truth.sum()):,} aboard, "
      f"{int((truth == 0).sum()):,} not")
print(f"priced threshold = C_fp / (C_fp + C_fn) = {priced:.2f}")
print(f"\\ncost at 0.50 (the habit) : {realised_cost(0.5):,.0f}")
print(f"cost at {priced:.2f} (priced)      : {realised_cost(priced):,.0f}")
print(f"saved against the habit  : {(1 - realised_cost(priced)/realised_cost(0.5))*100:.1f} per cent")
print(f"\\nand against no model at all: always aboard {always_aboard:,.0f}, "
      f"always not aboard {always_not:,.0f}")
print(f"saved against that floor : "
      f"{(1 - realised_cost(priced)/min(always_aboard, always_not))*100:.1f} per cent")
print("\\nQuote both, or you have quoted the flattering half.")'''),

(CODE, '''grid = np.linspace(0.0, 1.0, 201)
costs = [realised_cost(t) for t in grid]
fig = go.Figure()
fig.add_trace(go.Scatter(x=grid, y=costs, mode="lines", line=dict(color=BLUE, width=2.5),
                         name="the model, at each threshold"))
fig.add_trace(go.Scatter(x=[0, 1], y=[always_aboard, always_aboard], mode="lines",
                         line=dict(color=ORANGE, dash="dash"),
                         name=f"always aboard, no model: {always_aboard:,.0f}"))
for value, colour, label in ((priced, ORANGE, "priced"), (0.5, GREY, "the habit")):
    fig.add_trace(go.Scatter(x=[value], y=[realised_cost(value)], mode="markers+text",
                             marker=dict(size=13, color=colour), showlegend=False,
                             text=[f"{label} {value:.2f} → {realised_cost(value):,.0f}"],
                             textposition="top center"))
fig.update_layout(title="One half is a claim that both mistakes cost the same",
                  xaxis_title="threshold on the predicted probability (aboard when p ≥ t)",
                  yaxis_title="realised cost (vehicle-hour equivalents)")
show(fig, "cost_by_threshold")'''),

(MARKDOWN, """### And accuracy went down

Both of these are true at once, and there is no contradiction: the priced
threshold is *less accurate* and costs less. Accuracy weights every mistake
equally, which is exactly the assumption we rejected."""),

(CODE, '''for threshold, name in ((0.5, "the habit"), (priced, "priced")):
    said = (probabilities >= threshold).astype(int)
    print(f"{name:10} threshold {threshold:.2f}  accuracy {(said == truth).mean():.4f}  "
          f"cost {realised_cost(threshold):,.0f}")
print(f"{'no model':10} say aboard      accuracy {truth.mean():.4f}  cost {always_aboard:,.0f}")
print("\\nIf accuracy is the number on the dashboard, the wrong model wins.")'''),

(MARKDOWN, """### The assumption underneath: calibration

**Definition — calibration.** A score is calibrated when it means what it says:
among all the requests given score s, the event happens a share s of the time.

    P(Y = 1 | score = s) = s for every s in [0, 1]

*(DeGroot & Fienberg, 1983; Niculescu-Mizil & Caruana, 2005)*

**Definition — reliability in bands, and the largest gap.** The measurement of
that property on data you have.

    B_j = {i : e_j ≤ p_i < e_(j+1)}, the last band closed at the top;
    predicted_j = mean_{i ∈ B_j} p_i, actual_j = mean_{i ∈ B_j} y_i,
    kept when |B_j| ≥ r; largest gap = max_j |predicted_j − actual_j|

The threshold derivation assumed the first. The reliability diagram measures it:
what was promised against what happened, band by band. Lab 3 grades it, on this
service and on two invented models where the more accurate of the two is the
badly calibrated one — a pair worth showing the room if anybody argues that
accuracy implies calibration."""),

(CODE, '''edges = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
bands = []
for low, high in zip(edges[:-1], edges[1:]):
    last = high == edges[-1]
    inside = (probabilities >= low) & ((probabilities <= high) if last
                                       else (probabilities < high))
    if inside.sum() >= 50:
        bands.append({"band": f"{low:.1f}-{high:.1f}", "rows": int(inside.sum()),
                      "predicted": round(float(probabilities[inside].mean()), 3),
                      "actual": round(float(truth[inside].mean()), 3)})
bands = pd.DataFrame(bands)
bands["gap"] = (bands["predicted"] - bands["actual"]).round(3)
print(bands.to_string(index=False))

fig = go.Figure()
fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="perfect calibration",
                         line=dict(color=GREY, dash="dash")))
fig.add_trace(go.Scatter(x=bands["predicted"], y=bands["actual"], mode="markers+lines",
                         marker=dict(size=12 + 22 * bands["rows"] / bands["rows"].max(),
                                     color=BLUE), line=dict(color=BLUE, width=1.5),
                         name="this model, by band"))
fig.update_layout(title="Promised against happened: the reliability diagram",
                  xaxis_title="mean predicted probability of aboard (dimensionless)",
                  yaxis_title="share of those rows that were aboard (dimensionless)")
show(fig, "reliability")
print("\\nBelow the diagonal is over-prediction. The derived threshold is therefore a")
print("principled starting point on this model, not a guaranteed optimum.")'''),

(MARKDOWN, """### When the price of a miss is not a constant

Both prices above are constants, so the expected cost of a decision is a straight
line in the probability and the threshold rule is the whole story. Now price a
miss by its size.

**Definition — cost at the mean versus mean cost.** For one departure with c
seats and uncertain demand D, the cost at the mean demand is the shortfall cost
computed once at the average; the mean cost is the average of that cost over the
whole distribution.

    g(E[D]) = C_fn · max(0, E[D] − c) and
    E[g(D)] = C_fn · Σ_{d>c} (d − c) · P(D = d), with D ~ Poisson(λ)

*(Jensen, 1906; Module 4 block two proves the inequality and reuses this exact
departure.)*"""),

(CODE, '''SEATS, MEAN_DEMAND = 12, 10.0
demand = np.arange(0, int(MEAN_DEMAND) + 200)
mass = stats.poisson.pmf(demand, MEAN_DEMAND)

cost_at_mean = COST_FN * max(0.0, MEAN_DEMAND - SEATS)
mean_cost = COST_FN * float((np.maximum(0, demand - SEATS) * mass).sum())
fractile = COST_FN / (COST_FP + COST_FN)
print(f"{SEATS} seats, demand Poisson with mean {MEAN_DEMAND:.0f}, "
      f"a passenger left standing priced at {COST_FN:.0f}")
print(f"cost computed at the average demand : {cost_at_mean:.3f}")
print(f"average of the cost over the demand : {mean_cost:.3f}")
print(f"the gap (Jensen)                    : {mean_cost - cost_at_mean:.3f}")
print(f"probability demand exceeds capacity : "
      f"{1 - stats.poisson.cdf(SEATS, MEAN_DEMAND):.3f}")
print(f"the balancing capacity is the {fractile:.1f} quantile of demand: "
      f"{int(stats.poisson.ppf(fractile, MEAN_DEMAND))} seats — the same ratio as the")
print("threshold above, read as a demand quantile (the newsvendor's critical fractile,")
print("Arrow, Harris & Marschak, 1951)")

show_demand = np.arange(0, 21)
fig = make_subplots(specs=[[{"secondary_y": True}]])
fig.add_trace(go.Bar(x=show_demand, y=stats.poisson.pmf(show_demand, MEAN_DEMAND),
                     marker_color=GREY, opacity=0.45, name="probability of that demand"),
              secondary_y=True)
fig.add_trace(go.Scatter(x=show_demand, y=COST_FN * np.maximum(0, show_demand - SEATS),
                         mode="lines+markers", line=dict(color=BLUE, width=3),
                         name="cost of the day: 4 × max(0, D − 12)"), secondary_y=False)
fig.add_trace(go.Scatter(x=[0, 20], y=[mean_cost, mean_cost], mode="lines",
                         line=dict(color=RED, dash="dash"),
                         name=f"mean cost over the days: {mean_cost:.3f}"), secondary_y=False)
fig.add_trace(go.Scatter(x=[MEAN_DEMAND], y=[cost_at_mean], mode="markers",
                         marker=dict(color=ORANGE, size=15),
                         name=f"cost at the mean demand: {cost_at_mean:.3f}"), secondary_y=False)
fig.update_layout(title="The cost of the average day is not the average cost of the days",
                  xaxis_title="demand at the departure (passengers)")
fig.update_yaxes(title_text="cost of the day (vehicle-hour equivalents)", secondary_y=False)
fig.update_yaxes(title_text="probability of that demand (dimensionless)", secondary_y=True,
                 showgrid=False)
show(fig, "jensen")'''),

(MARKDOWN, """### The failure where nothing goes wrong

**Definition — training–serving skew.** The input is prepared one way when the
model is fitted and another way when it is asked, so the same model answers a
different question in service.

    skew ⇔ prepare_serve(x) ≠ prepare_train(x); cure: input_j =
    ((x[f_j] if present else median_{f_j}) − mean_{f_j}) / std_{f_j}
    for f_j in the stored order, j = 1..k

*(Breck et al., 2019; Zaharia et al., 2018)*

Two columns exchanged in the preparation. A model does not know its columns by
name — it knows position one, position two, position three.

Lab 4 has the student *cause* this rather than read about it: two fields
exchanged in every request, the row assembled in the order the keys arrived, and
the share of decisions that come out different counted. The cell below reaches
the same number from the model's side, by exchanging two columns of the prepared
matrix."""),

(CODE, '''swapped = prepared.copy()
swapped[:, [0, 1]] = swapped[:, [1, 0]]
skewed = artefact["model"].predict_proba(swapped)[:, 1]

before = (probabilities >= priced).astype(int)
after = (skewed >= priced).astype(int)
print(f"decisions changed   : {(before != after).mean()*100:.1f} per cent")
print(f"accuracy before     : {(before == truth).mean():.4f}")
print(f"accuracy after      : {(after == truth).mean():.4f}")
print("requests that failed: 0\\nexceptions raised   : 0\\nlatency change      : none")

fig = go.Figure()
fig.add_trace(go.Histogram(x=probabilities, xbins=dict(start=0, end=1, size=0.025),
                           marker_color=BLUE, name="columns as trained"))
fig.add_trace(go.Histogram(x=skewed, xbins=dict(start=0, end=1, size=0.025),
                           marker_color=ORANGE, opacity=0.75, name="two columns swapped"))
fig.update_layout(barmode="overlay", title="Every request succeeded. Both times.",
                  xaxis_title="predicted probability of aboard (dimensionless)",
                  yaxis_title="requests in the test period (count)")
show(fig, "skew")
print("\\nThere is no error to find. The only symptom is in the answers themselves,")
print("which is why Module 5 monitors the output distribution and not just the")
print("error rate.")'''),

(MARKDOWN, """### The cure, twice: stored order, and the signature

The discipline is to loop over the *stored* feature order. The enforcement is the
signature recorded with the model: it matches columns by name, restores their
order, and refuses a column it was not trained with.

Three outcomes, on one request."""),

(CODE, '''complete = test.dropna(subset=FEATURES)
transform = artefact["transform"]

def prepared_row(request):
    return [[(request[f] - transform["means"][f]) / transform["stds"][f]
             for f in transform["features"]]]

# The first complete row where exchanging two positions moves the answer by more
# than a tenth: on many rows the model is unmoved, and a demonstration that shows
# a change of one part in a thousand teaches the wrong lesson.
for index in range(len(complete)):
    natural = {f: float(complete.iloc[index][f]) for f in FEATURES}
    row = prepared_row(natural)
    exchanged = [list(row[0])]
    exchanged[0][0], exchanged[0][-1] = exchanged[0][-1], exchanged[0][0]
    right = float(artefact["model"].predict_proba(row)[0][1])
    quietly_wrong = float(artefact["model"].predict_proba(exchanged)[0][1])
    if abs(quietly_wrong - right) > 0.1:
        break
reversed_keys = {f: natural[f] for f in reversed(FEATURES)}

print(f"1. the raw pickle, row prepared correctly            : {right:.4f}")
print(f"2. the raw pickle, two positions exchanged           : {quietly_wrong:.4f}"
      "   <- accepted without a word, status would be 200")
answer = champion.predict(pd.DataFrame([reversed_keys]))
print(f"3. the registered model, keys reversed               : "
      f"{float(np.asarray(answer)[0][1]):.4f}   <- reordered by name")
from mlflow.exceptions import MlflowException
renamed = {("rssi_c" if f == "rssiC" else f): v for f, v in natural.items()}
try:
    champion.predict(pd.DataFrame([renamed]))
    print("4. the registered model, rssiC renamed               : answered (unexpected)")
except MlflowException as refusal:
    print(f"4. the registered model, rssiC renamed               : refused — "
          f"{str(refusal).splitlines()[0][:52]}...")
print("\\nDiscipline protects the batch job written next year; enforcement protects you")
print("from your own discipline lapsing. Teach both.")'''),

(MARKDOWN, """### Honest speed

**Definition — percentile latency, nearest rank.** The p-th percentile of n
durations is the value at position ⌈p·n/100⌉ once they are sorted, so the number
reported is a duration that actually occurred.

    Q(p) = x_(⌈p · n / 100⌉), where x_(1) ≤ … ≤ x_(n) are the sorted durations

*(Hyndman & Fan, 1996, definition 1; Dean & Barroso, 2013)*

A duration is a property of the machine and the moment, so these numbers are
measured here, now, and they are not on any slide."""),

(CODE, '''def percentile_latency(durations, percentile=95.0):
    ordered = sorted(float(d) for d in durations)
    rank = max(1, math.ceil(percentile / 100 * len(ordered)))
    return ordered[min(rank, len(ordered)) - 1]

def durations_ms(version, requests=200):
    model = load_artefact(version)["model"]
    measured = []
    for index in range(requests):
        one = prepared[index % len(prepared)].reshape(1, -1)
        start = time.perf_counter()
        model.predict_proba(one)
        measured.append((time.perf_counter() - start) * 1000)
    return measured

print(f"{'version':8} {'mean ms':>9} {'p50 ms':>8} {'p95 ms':>8} {'p99 ms':>8}")
for version in ("v1", "v2"):
    measured = durations_ms(version)
    print(f"{version:8} {np.mean(measured):9.2f} {percentile_latency(measured, 50):8.2f} "
          f"{percentile_latency(measured, 95):8.2f} {percentile_latency(measured, 99):8.2f}")

textbook = [10.0] * 94 + [1000.0] * 6
print(f"\\nninety-four requests at 10 ms and six at 1000 ms: mean "
      f"{np.mean(textbook):.1f} ms describes nobody; the 95th percentile is "
      f"{percentile_latency(textbook, 95):.0f} ms, which an operations team can act on")
print("Latency is a feature. A model one point more accurate and seventeen times")
print("more work per request may be worse for the service it sits in.")'''),

(MARKDOWN, """## Practice

1. **Would a latency gate change the decision?** Add a second rule to
   `promote_if_better`: refuse a candidate whose 95th-percentile latency is more
   than twice the approved model's. Success criterion: v2 is refused, and your
   printed reason names both percentiles.
2. **How wrong can the costs be?** Sweep the cost ratio from 1:1 to 1:20, derive
   the threshold for each, and compare its realised cost against one half.
   Success criterion: you can name the ratios where the derivation loses, and say
   why (look at the reliability diagram again).
3. **Which column swap hurts most?** Swap each pair of the four features in turn
   and measure the change in accuracy. Success criterion: a table of six pairs
   ordered by damage, and one sentence on why the worst pair is the worst.

Answers in the Appendix."""),

(CODE, '''# Your workings here.
'''),

(MARKDOWN, """## Appendix

### Answers"""),

(CODE, '''# 1. A latency gate refuses v2 as well -- it does seventeen times the work per request.
p95 = {v: percentile_latency(durations_ms(v), 95) for v in ("v1", "v2")}
print(f"v1 p95 {p95['v1']:.2f} ms, v2 p95 {p95['v2']:.2f} ms -> "
      f"{'refused' if p95['v2'] > 2 * p95['v1'] else 'allowed'} by a two-times latency rule")

# 2. It beats one half at most ratios -- but NOT all, and that is worth seeing.
print("\\nratio  threshold   cost   cost at 0.50   derived wins?")
for ratio in (1, 2, 3, 4, 8, 20):
    t = 1.0 / (1.0 + ratio)
    def cost_at(th, r=ratio):
        said = probabilities >= th
        return int((said & (truth == 0)).sum()) + int(((~said) & (truth == 1)).sum()) * r
    verdict = "yes" if cost_at(t) < cost_at(0.5) else ("tie" if cost_at(t) == cost_at(0.5) else "NO")
    print(f"1:{ratio:<4} {t:9.3f} {cost_at(t):7,} {cost_at(0.5):14,} {verdict:>14}")
print("A tie is not a win: Lab 1's gate refuses a candidate that merely equals the")
print("model in service, and this sweep is held to the same rule.")

# 3. Swapping the two most differently-scaled columns hurts most.
print("\\nswapped pair            accuracy")
damage = []
for i in range(len(FEATURES)):
    for j in range(i + 1, len(FEATURES)):
        variant = prepared.copy(); variant[:, [i, j]] = variant[:, [j, i]]
        accuracy = float(((artefact["model"].predict_proba(variant)[:, 1] >= priced)
                          .astype(int) == truth).mean())
        damage.append((f"{FEATURES[i]} <-> {FEATURES[j]}", accuracy))
for pair, accuracy in sorted(damage, key=lambda item: item[1]):
    print(f"  {pair:22} {accuracy:.4f}")
print(f"  {'no swap':22} {(before == truth).mean():.4f}")'''),

(MARKDOWN, """### On the stand-in

Everything above ran against a model trained in seven seconds from Module 2's
generated table. That is deliberate, and it is the argument for release by
indirection: the service, the contract, the threshold and the skew demonstration
are all indifferent to which model sits behind the name.

When the modelling sessions hand over an artefact, it goes into
`service/artefacts/`, the registry pointer moves, the alias moves with it, and
every number in this notebook changes while not one line of it does."""),

(MARKDOWN, """## References

- Arrow, K. J., Harris, T. & Marschak, J. (1951). *Optimal Inventory Policy.* Econometrica 19(3), 250–272. https://doi.org/10.2307/1906813
- Breck, E., Polyzotis, N., Roy, S., Whang, S. E. & Zinkevich, M. (2019). *Data Validation for Machine Learning.* Proceedings of Machine Learning and Systems (MLSys) 1.
- Dean, J. & Barroso, L. A. (2013). *The Tail at Scale.* Communications of the ACM 56(2), 74–80. https://doi.org/10.1145/2408776.2408794
- DeGroot, M. H. & Fienberg, S. E. (1983). *The Comparison and Evaluation of Forecasters.* The Statistician 32(1/2), 12–22. https://doi.org/10.2307/2987588
- Elkan, C. (2001). *The Foundations of Cost-Sensitive Learning.* IJCAI, 973–978. https://cseweb.ucsd.edu/~elkan/rescale.pdf
- Fielding, R., Nottingham, M. & Reschke, J. (2022). *HTTP Semantics.* Request for Comments 9110, §15.5.21. https://www.rfc-editor.org/rfc/rfc9110
- Huyen, C. (2022). *Designing Machine Learning Systems*, ch. 7. O'Reilly.
- Hyndman, R. J. & Fan, Y. (1996). *Sample Quantiles in Statistical Packages.* The American Statistician 50(4), 361–365. https://doi.org/10.1080/00031305.1996.10473566
- Jensen, J. L. W. V. (1906). *Sur les fonctions convexes et les inégalités entre les valeurs moyennes.* Acta Mathematica 30, 175–193. https://doi.org/10.1007/BF02418571
- Niculescu-Mizil, A. & Caruana, R. (2005). *Predicting Good Probabilities with Supervised Learning.* ICML, 625–632. https://doi.org/10.1145/1102351.1102430
- Provost, F. & Fawcett, T. (2013). *Data Science for Business*, ch. 7–8. O'Reilly.
- Schelter, S., Biessmann, F., Januschowski, T., Salinas, D., Seufert, S. & Szarvas, G. (2018). *On Challenges in Machine Learning Model Management.* IEEE Data Engineering Bulletin 41(4), 5–15.
- Sculley, D. et al. (2015). *Hidden Technical Debt in Machine Learning Systems.* NeurIPS 28. https://papers.neurips.cc/paper/5656-hidden-technical-debt-in-machine-learning-systems.pdf
- Wilson, E. B. (1927). *Probable inference, the law of succession, and statistical inference.* JASA 22. https://www.tandfonline.com/doi/abs/10.1080/01621459.1927.10502953
- Zaharia, M. et al. (2018). *Accelerating the Machine Learning Lifecycle with MLflow.* IEEE Data Engineering Bulletin 41(4), 39–45.
- MLflow documentation — Model Registry workflow. https://mlflow.org/docs/latest/ml/model-registry/workflow/

*All output above is Author's own, computed from `Module 3/exercises/service/models.py`
— the stand-in model trained from Module 2's generated table — and from the local
MLflow store beside it. Nothing here reads the archive. Figures are written to
`Module 3/notebook/figures/`.*"""),
]


def main(*arguments):
    notebook = new_notebook(cells=[
        new_markdown_cell(text) if kind == MARKDOWN else new_code_cell(text)
        for kind, text in CELLS])
    notebook.metadata.update({
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"}})

    if "--no-run" not in arguments:
        # Executed where tools/check_notebook.py --run executes it, so that what
        # is committed is what ran.
        from nbclient import NotebookClient
        NotebookClient(notebook, timeout=1800, kernel_name="python3",
                       resources={"metadata": {"path": str(EXERCISES)}}).execute()

    OUTPUT.write_text(nbformat.writes(notebook))
    executed = sum(1 for c in notebook.cells
                   if c.cell_type == "code" and c.get("execution_count"))
    print(f"wrote {OUTPUT.name} — {len(CELLS)} cells, {executed} executed")


if __name__ == "__main__":
    main(*sys.argv[1:])
