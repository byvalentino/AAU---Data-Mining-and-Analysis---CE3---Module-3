"""Lab 1, solved — with the reasoning, not only the code.

Run it:  python3 solutions/lab_01.py        (or  python3 labs/01_the_gate.py  after apply.py)
It narrates the gate on the two measured models, then records both in a fresh
MLflow store under out/ and moves the alias, and draws the accuracy/size bars.
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
METRIC = "accuracy"
HIGHER_IS_BETTER = True


def promote_if_better(registry: dict, candidate: str, metrics: dict):
    """Promote only on the agreed metric, and say why when you do not.

    Definition graded by the check:
        promote(candidate) ⇔ m(candidate) > m(approved) + δ, with δ = 0 here (a tie is refused) and δ = 0.01 in Module 5
        (Zaharia et al., 2018). Slide: "Definition — the gate".

    The thing worth noticing is what this function refuses to look at. It does
    not know how large the candidate is, how new it is, how long it took to
    train, how much it cost, or who is keen on it. It compares one number.

    That narrowness is the whole value. Every one of those other properties is
    an argument somebody will make in a meeting, and the gate exists precisely
    so that the argument has already been settled — in the open, before there
    was a candidate to be enthusiastic about.

    The margin is a choice, and it is written down: nought here, so that a tie
    is refused; one point in Module 5. Module 4's Wilson interval on the number
    of labels is where a principled margin comes from — a difference smaller
    than the label noise is not a difference.
    """
    approved = registry["approved"]
    if candidate not in metrics:
        return False, f"candidate {candidate} has no measured metrics"
    if approved not in metrics:
        return False, f"approved model {approved} has no measured metrics"

    candidate_score = metrics[candidate][METRIC]
    approved_score = metrics[approved][METRIC]
    better = candidate_score > approved_score if HIGHER_IS_BETTER else candidate_score < approved_score

    if not better:
        return False, (
            f"candidate {candidate} scored {candidate_score:.4f} on {METRIC} against "
            f"approved {approved} at {approved_score:.4f}; not promoted")

    registry["approved"] = candidate
    # History is append-only, so that rollback is a step backwards rather than
    # an archaeology exercise.
    registry.setdefault("history", []).append(candidate)
    return True, (
        f"candidate {candidate} scored {candidate_score:.4f} on {METRIC} against "
        f"approved {approved} at {approved_score:.4f}; promoted")


def rollback(registry: dict) -> str:
    """Step the pointer back. The same move as promotion, in reverse.

    Definition graded by the check:
        serve(name) = artefact[approved]; promote: approved ← candidate, history ← history + [candidate]; rollback: history ← history[:−1], approved ← history[−1]
        (Zaharia et al., 2018; Schelter et al., 2018). Slide: "Definition — release by indirection".

    Worth saying out loud: rollback is not a special emergency procedure. It is
    the ordinary release mechanism run backwards, which is exactly why it can be
    trusted at three in the morning. A rollback path that is a different
    mechanism from the release path is a rollback path nobody has tested.
    """
    history = registry.get("history", [])
    if len(history) < 2:
        return registry["approved"]          # nothing to go back to
    history.pop()
    registry["approved"] = history[-1]
    return registry["approved"]


def log_training_run(name: str, model, X, y_pred, params: dict, metrics: dict) -> int:
    """One run, recorded as it happens; the model becomes a version of one name.

    Definition graded by the check:
        S = (name_i, type_i)_{i=1..k}, recorded with the model; accept(x) ⇔ ∀i: name_i ∈ columns(x) ∧ type(x[name_i]) = type_i; input = (x[name_1], …, x[name_k])
        (Zaharia et al., 2018). Slide: "Definition — model signature".

    Four things are written inside the run, and each answers a question that
    otherwise needs an investigation months later. The parameters: what was
    set. The metrics: what was measured, on the agreed metric. The environment:
    which library versions can unpickle this artefact — recorded from the
    process that trained it, not reconstructed from a file. And the model with
    its signature: the names, types and order of the columns it was trained on,
    which the platform enforces on every request from now on.
    """
    with mlflow.start_run(run_name=name):
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        info = mlflow.sklearn.log_model(
            model, artifact_path="model",
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
            signature=infer_signature(X, y_pred),
            input_example=X.head(2),
            registered_model_name=REGISTERED_MODEL,
            pyfunc_predict_fn="predict_proba",       # the service needs the probability
            pip_requirements=environment_pins())      # the environment, as it actually is
    return int(info.registered_model_version)


def promote_by_alias(client: MlflowClient, name: str, version, alias: str = CHAMPION) -> str:
    """Release is moving the alias. Rollback is moving it back.

    Definition graded by the check:
        alias: name@alias → version; promote: alias[champion] ← version; rollback: alias[champion] ← previous version; serve(models:/name@champion)
        (Zaharia et al., 2018). Slide: "Definition — release by alias".

    The service loads models:/aboard@champion. It never learns a version
    number, so a release changes nothing in the service and a rollback is this
    same call with the previous version — the fifty-line registry of part (a),
    kept by a platform.
    """
    client.set_registered_model_alias(name, alias, str(version))
    return str(client.get_model_version_by_alias(name, alias).version)


# --------------------------------------------------------------------------
# Part (c) — the verdict a person signs
# --------------------------------------------------------------------------
VERDICT_CALLS = ("promote", "hold", "roll back")

EVIDENCE_KEYS = (
    "candidate_accuracy", "approved_accuracy", "approved_accuracy_at_promotion",
    "candidate_realised_cost", "approved_realised_cost", "trivial_baseline_cost",
    "candidate_latency_p95", "latency_budget",
    "candidate_megabytes", "approved_megabytes",
)

REGRESSION_MARGIN = 0.02
LATENCY_PERCENTILE = 95.0


def promotion_verdict(evidence: dict) -> tuple[str, str]:
    """One of three words, and the argument for it.

    Definition graded by the check:
        verdict(evidence) = (call, reason), call ∈ {promote, hold, roll back}; roll back ⇔ m(approved, now) < m(approved, at promotion) − δ_r; promote ⇔ ¬roll back ∧ K(candidate) < K(approved) ∧ K(candidate) < K(trivial) ∧ p95(candidate) ≤ budget; otherwise hold
        (Provost & Fawcett, 2013, ch. 7-8; Sculley et al., 2015). Slide:
        "Definition — the promotion verdict".

    Three things about this function are worth more than the code in it.

    **The order of the tests.** The health of the model in service is asked
    first, and it does not depend on the candidate at all. A service that has
    fallen below what it promised is not repaired by releasing something new
    into it: the release path and the recovery path would then run at the same
    time, at the worst possible moment, and nobody would know afterwards which
    of the two changed the number. Put the previous version back, then argue
    about the candidate on a Tuesday.

    **The unit.** Nothing here compares accuracies to decide a release. Accuracy
    weights every mistake the same, and this operator does not: a passenger left
    standing was priced at four times a shuttle sent for nobody. The candidate
    that is more accurate and dearer at the priced threshold is held, and a
    dashboard showing accuracy would have released it.

    **The floor.** Being cheaper than the model in service is not enough. The
    honest question is whether the service earns its keep at all, and the
    comparison that answers it is against the best policy that never asks a
    model. Quote the saving against the habit alone and you have quoted the
    flattering half (Provost & Fawcett, 2013).
    """
    given = {key: float(evidence[key]) for key in EVIDENCE_KEYS if key in evidence}
    missing = [key for key in EVIDENCE_KEYS if key not in given]
    if missing:
        return "hold", (f"the evidence is incomplete: {', '.join(missing)} was not "
                        "measured, and a verdict on evidence nobody gathered is a guess")

    # 1. Is the model already in service still the model that was promoted?
    if given["approved_accuracy_at_promotion"] - given["approved_accuracy"] > REGRESSION_MARGIN:
        return "roll back", (
            f"the approved accuracy at promotion was "
            f"{given['approved_accuracy_at_promotion']:.4f} and the approved accuracy is "
            f"{given['approved_accuracy']:.4f} now, a fall past the {REGRESSION_MARGIN} "
            "regression margin, so the service in place has regressed; the previous version "
            "goes back before any candidate is discussed")

    # 2. Does the candidate cost less than what is running, in the operator's unit?
    if given["candidate_realised_cost"] >= given["approved_realised_cost"]:
        return "hold", (
            f"the candidate realised cost is {given['candidate_realised_cost']:.0f} against "
            f"the approved realised cost {given['approved_realised_cost']:.0f} at the priced "
            "threshold, so the release buys the operator nothing; the candidate accuracy of "
            f"{given['candidate_accuracy']:.4f} is measured with every mistake priced the "
            "same, which this operator does not do")

    # 3. Does it clear the floor — the best thing you could do with no model at all?
    if given["candidate_realised_cost"] >= given["trivial_baseline_cost"]:
        return "hold", (
            f"the candidate realised cost is {given['candidate_realised_cost']:.0f} and the "
            f"trivial baseline cost is {given['trivial_baseline_cost']:.0f} on the same rows, "
            "so the service is dearer than answering without a model at all; cheaper than "
            f"the approved realised cost {given['approved_realised_cost']:.0f} is the "
            "flattering half of that comparison")

    # 4. Is it inside the budget the service agreement promised its callers?
    if given["candidate_latency_p95"] > given["latency_budget"]:
        return "hold", (
            f"the candidate latency p95 is {given['candidate_latency_p95']:.0f} milliseconds "
            f"against a latency budget of {given['latency_budget']:.0f}, so one request in "
            "twenty breaks the agreement; the candidate realised cost of "
            f"{given['candidate_realised_cost']:.0f} is bought with a delay somebody has "
            "already promised not to impose")

    return "promote", (
        f"the candidate realised cost is {given['candidate_realised_cost']:.0f} against the "
        f"approved realised cost {given['approved_realised_cost']:.0f} and the trivial "
        f"baseline cost {given['trivial_baseline_cost']:.0f}, so it is cheaper than what is "
        "running and cheaper than using no model at all, and the candidate latency p95 of "
        f"{given['candidate_latency_p95']:.0f} milliseconds is inside the "
        f"{given['latency_budget']:.0f} budget")


if __name__ == "__main__":
    import contextlib
    import io
    import shutil
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from _narrate import narrator, show_table, save_figure

    say = narrator(LAB)
    say.info("Lab 1 — the gate refuses a bigger, newer, dearer candidate that is not better; "
             "then the same two models go into a real registry with their signature")

    metrics = load_metrics()
    say.info("loaded service/artefacts/metrics.json — %d versions, measured by "
             "service/models.py on Module 2's generated table (seed 20200122)", len(metrics))
    for version, facts in metrics.items():
        say.info("  %s: accuracy %.4f on the later 30 per cent of rows, %s bytes on disk, "
                 "trained on %d rows", version, facts["accuracy"], f"{facts['size_bytes']:,}",
                 facts["trained_on_rows"])

    # Part (a): the gate, on the measured candidate, then on a hypothetical better one.
    registry = json.loads(registry_path().read_text())
    say.info("registry.json says approved = %s, history = %s", registry["approved"],
             registry["history"])
    promoted, reason = promote_if_better(registry, "v2", metrics)
    say.info("promote v2? %s — %s", promoted, reason)
    say.info("the registry after the refusal is unchanged: approved = %s, because a refusal "
             "must leave the pointer exactly where it was", registry["approved"])
    better = {v: dict(f) for v, f in metrics.items()}
    better["v2"]["accuracy"] = round(metrics["v1"]["accuracy"] + 0.05, 4)
    promoted, reason = promote_if_better(registry, "v2", better)
    say.info("with a candidate five points better (hypothetical): promoted = %s; approved = %s, "
             "history = %s", promoted, registry["approved"], registry["history"])
    say.info("rollback → approved = %s; the same move reversed, not a separate procedure",
             rollback(registry))

    # Part (b): a fresh store under out/, so nothing here touches what the checks read.
    store = pathlib.Path(__file__).resolve().parent.parent / "out" / "lab_01_store"
    if store.exists():
        shutil.rmtree(store)
    client = open_store(store)
    say.info("opened a fresh MLflow store at out/lab_01_store (sqlite tracking + registry, "
             "artefacts beside it) — the labs' own store in exercises/ is untouched")
    table = load_table()
    say.info("loaded the table: %d rows, generated (seed 20200122); signature inferred from "
             "the rows where all four features are present", len(table))
    example = table[FEATURES].dropna().head(200).reset_index(drop=True)
    versions = {}
    for version in ("v1", "v2"):
        artefact = load_artefact(version)
        pipeline = as_pipeline(artefact)
        params = {"n_estimators": artefact["model"].n_estimators,
                  "max_depth": artefact["model"].max_depth,
                  "random_state": metrics[version]["seed"],
                  "features": ",".join(FEATURES),
                  "train_rows": metrics[version]["trained_on_rows"]}
        logged = {"accuracy": metrics[version]["accuracy"],
                  "size_bytes": float(metrics[version]["size_bytes"])}
        # MLflow narrates its own registry writes on standard error ("Created
        # version '1' of model 'aboard'"). True, and not this demonstration's
        # sentence: the line below says the same thing with the numbers that
        # matter, so the tool's copy goes to a sink.
        with contextlib.redirect_stderr(io.StringIO()):
            versions[version] = log_training_run(version, pipeline, example,
                                                 pipeline.predict_proba(example),
                                                 params, logged)
        say.info("logged run %s → registered version %d of '%s', with %d params, %d metrics, "
                 "the environment (%s) and the signature %s", version, versions[version],
                 REGISTERED_MODEL, len(params), len(logged), ", ".join(environment_pins()),
                 FEATURES)
    champion = max(metrics, key=lambda v: metrics[v]["accuracy"])
    now = promote_by_alias(client, REGISTERED_MODEL, versions[champion], CHAMPION)
    say.info("alias %s → version %s (%s), because it has the higher accuracy; the service "
             "loads models:/%s@%s and never a version number", CHAMPION, now, champion,
             REGISTERED_MODEL, CHAMPION)

    rows = []
    for version in ("v1", "v2"):
        model_version = client.get_model_version(REGISTERED_MODEL, str(versions[version]))
        run = client.get_run(model_version.run_id)
        rows.append({"run": version, "registered version": int(model_version.version),
                     "alias": ",".join(model_version.aliases) or "—",
                     "accuracy": run.data.metrics["accuracy"],
                     "size_bytes": int(run.data.metrics["size_bytes"]),
                     "n_estimators": run.data.params["n_estimators"],
                     "max_depth": run.data.params["max_depth"]})
    show_table(pd.DataFrame(rows), "the registry, read back through MlflowClient", logger=say)
    loaded = mlflow.pyfunc.load_model(f"models:/{REGISTERED_MODEL}@{CHAMPION}")
    say.info("the champion's signature, as the platform will enforce it: %s",
             loaded.metadata.get_input_schema().input_names())

    # The figure: what matters against what is persuasive.
    fig = make_subplots(rows=1, cols=2, subplot_titles=("What matters: accuracy on later data",
                                                        "What is persuasive: megabytes on disk"))
    names = ["v1 approved", "v2 candidate"]
    colours = ["#2A78D6", "#E07B39"]
    fig.add_trace(go.Bar(x=names, y=[metrics["v1"]["accuracy"], metrics["v2"]["accuracy"]],
                         marker_color=colours, text=[f"{metrics[v]['accuracy']:.4f}" for v in ("v1", "v2")],
                         textposition="outside", showlegend=False), row=1, col=1)
    fig.add_trace(go.Bar(x=names, y=[metrics["v1"]["size_bytes"] / 1e6, metrics["v2"]["size_bytes"] / 1e6],
                         marker_color=colours, text=[f"{metrics[v]['size_bytes'] / 1e6:.2f}" for v in ("v1", "v2")],
                         textposition="outside", showlegend=False), row=1, col=2)
    fig.update_yaxes(title_text="accuracy (share of rows)", range=[0, 1.0], row=1, col=1)
    fig.update_yaxes(title_text="size on disk (megabytes)", row=1, col=2)
    fig.update_layout(title="The gate compares one number and ignores the persuasive one")
    save_figure(fig, "gate_bars", LAB, logger=say)


    # Part (c): the verdict, on evidence this module actually measured. The two
    # accuracies come from metrics.json, the two bills and the no-model floor are
    # Lab 3's arithmetic at the priced threshold, and the percentile is Lab 4's,
    # measured here rather than quoted -- which is the point of the exercise.
    import numpy as np
    import time

    table = load_table()
    test = table.iloc[int(len(table) * 0.7):]
    truth = (test["label2"] == "IN").astype(int).to_numpy()
    priced, cost_fp, cost_fn = 0.2, 1.0, 4.0

    def bill(version):
        artefact = load_artefact(version)
        fitted = artefact["transform"]
        prepared = np.column_stack([
            ((test[f].astype(float).fillna(fitted["medians"][f]) - fitted["means"][f])
             / fitted["stds"][f]).to_numpy() for f in fitted["features"]])
        probabilities = artefact["model"].predict_proba(prepared)[:, 1]
        said = probabilities >= priced
        return (cost_fp * float(((said == 1) & (truth == 0)).sum())
                + cost_fn * float(((said == 0) & (truth == 1)).sum())), prepared, artefact

    approved_cost, prepared_v1, artefact_v1 = bill("v1")
    candidate_cost, _, _ = bill("v2")
    trivial = min(cost_fp * float((truth == 0).sum()), cost_fn * float((truth == 1).sum()))

    durations = []
    for index in range(200):
        row = prepared_v1[index % len(prepared_v1)].reshape(1, -1)
        started = time.perf_counter()
        artefact_v1["model"].predict_proba(row)
        durations.append((time.perf_counter() - started) * 1000)
    ordered = sorted(durations)
    rank = max(1, int(-(-LATENCY_PERCENTILE / 100 * len(ordered)) // 1))
    p95 = ordered[rank - 1]

    measured = {
        "candidate_accuracy": metrics["v2"]["accuracy"],
        "approved_accuracy": metrics["v1"]["accuracy"],
        "approved_accuracy_at_promotion": metrics["v1"]["accuracy"],
        "candidate_realised_cost": candidate_cost,
        "approved_realised_cost": approved_cost,
        "trivial_baseline_cost": trivial,
        "candidate_latency_p95": round(p95, 2),
        "latency_budget": 120.0,
        "candidate_megabytes": round(metrics["v2"]["size_bytes"] / 1e6, 2),
        "approved_megabytes": round(metrics["v1"]["size_bytes"] / 1e6, 2),
    }
    say.info("the evidence, measured on the %d test rows at the priced threshold %.2f "
             "(false positive 1, false negative 4): %s", len(test), priced,
             ", ".join(f"{k} {v}" for k, v in measured.items()))
    call, reason = promotion_verdict(measured)
    say.info("verdict: %s — %s", call.upper(), reason)
    say.info("and the case that decides the module: a candidate MORE accurate than what is "
             "running and dearer at the priced threshold")
    # Everything measured, one number moved: the candidate's accuracy raised above
    # the model in service. Its bill at the priced threshold is left exactly as it
    # was measured, which is what makes the case uncomfortable.
    dearer = {**measured, "candidate_accuracy": round(metrics["v1"]["accuracy"] + 0.05, 4)}
    call, reason = promotion_verdict(dearer)
    say.info("verdict: %s — %s", call.upper(), reason)
    say.info("the latency budget of %.0f milliseconds is an agreement, not a measurement; the "
             "%.0fth percentile beside it is measured on this machine today and belongs in no "
             "slide", measured["latency_budget"], LATENCY_PERCENTILE)

    say.info("what the check grades: v2 and a tied candidate refused with the registry unchanged "
             "and a reason quoting a score; a better candidate promoted; rollback returns v1; "
             "and in a fresh store two runs carry the module's metrics, champion sits on the "
             "more accurate version, and its signature names %s in order; and the verdict on six "
             "sets of evidence whose right calls differ, with a reason built from the numbers "
             "it was handed", FEATURES)
