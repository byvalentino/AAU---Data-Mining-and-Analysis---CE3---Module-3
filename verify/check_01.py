#!/usr/bin/env python3
"""Check 1 — the gate refuses a bigger, newer, slower, worse model; the store records both.

Part (a) grades the fifty-line registry: refusal, promotion, tie, rollback.
Part (b) grades the MLflow logging in a store of its own, under a temporary
directory, so it never depends on the student's earlier runs and never touches
the store the other labs read. Run identifiers are random, so nothing here
compares an identifier: the check compares content — run names, metrics,
params, which version the alias names, and the signature's column names.
"""
import contextlib, copy, io, pathlib, sys, tempfile
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _harness import run, close, not_ready, explain, grade_reason      # noqa: E402
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
try:
    from lab_support import (load_metrics, load_artefact, load_table, as_pipeline,   # noqa: E402
                             open_store, REGISTERED_MODEL, CHAMPION, FEATURES)
    import mlflow                                                       # noqa: E402
except ImportError as unready:
    not_ready(unready)


def part_a(lab, metrics):
    # The real test: a candidate that is bigger, newer and slower, and worse.
    registry = {"approved": "v1", "history": ["v1"]}
    promoted, reason = lab.promote_if_better(registry, "v2", metrics)
    assert promoted is False, explain(
        "m3:gate:refuse",
        "v2 was promoted",
        f"It is {metrics['v2']['size_bytes'] / metrics['v1']['size_bytes']:.0f} times the "
        f"size and scores {metrics['v2']['accuracy']:.4f} against "
        f"{metrics['v1']['accuracy']:.4f} on the agreed metric. Bigger and newer are not "
        "the metric, and the gate's value is that it cannot see either.")
    assert registry["approved"] == "v1", explain(
        "m3:gate:refusal-moved-pointer",
        "the candidate was refused and the registry changed anyway",
        "A refusal must leave the approved pointer exactly where it was: releasing IS "
        "writing that entry, so a refusal that writes it has released something.")
    assert isinstance(reason, str) and len(reason) > 15, (
        f"the reason was {reason!r}. A refusal a person cannot act on is a refusal "
        "they will re-submit unchanged — name both scores.")
    assert any(str(round(metrics[v]["accuracy"], 4)) in reason for v in ("v1", "v2")), (
        f"the reason {reason!r} does not contain either measured score. Say what was "
        "compared, not that a comparison happened.")

    # And it must promote when the candidate really is better.
    better = copy.deepcopy(metrics)
    better["v2"]["accuracy"] = metrics["v1"]["accuracy"] + 0.05
    registry = {"approved": "v1", "history": ["v1"]}
    promoted, reason = lab.promote_if_better(registry, "v2", better)
    assert promoted is True, "a genuinely better candidate was refused"
    assert registry["approved"] == "v2", (
        "promotion reported success but the approved pointer still names v1. "
        "Release *is* moving the pointer; nothing else counts as released.")
    assert registry["history"][-1] == "v2", (
        "history did not record the promotion, so rollback has nowhere to go")

    # A tie is not an improvement: the margin here is nought, and nought means
    # strictly greater, not greater or equal.
    tied = copy.deepcopy(metrics)
    tied["v2"]["accuracy"] = metrics["v1"]["accuracy"]
    registry = {"approved": "v1", "history": ["v1"]}
    promoted, _ = lab.promote_if_better(registry, "v2", tied)
    assert promoted is False, explain(
        "m3:gate:tie",
        "a candidate that scored exactly the same as the model in service was promoted",
        "The margin here is nought, and nought means strictly greater. Equal is not "
        "better: swapping a running model for no measured gain spends risk and buys "
        "nothing.")

    # Rollback is the same move, backwards.
    registry = {"approved": "v2", "history": ["v1", "v2"]}
    now = lab.rollback(registry)
    assert now == "v1" and registry["approved"] == "v1", (
        f"rollback returned {now!r} and left approved at {registry['approved']!r}; "
        "expected v1 for both")


def part_b(lab, metrics):
    table = load_table()
    # Raw rows with all four features present: a column absent from the example
    # rows would be inferred as optional, and the model's door requires all four.
    example = table[FEATURES].dropna().head(100).reset_index(drop=True)
    assert len(example) >= 10, "fixture error: too few rows carry all four features"

    with tempfile.TemporaryDirectory() as fresh:
        client = open_store(pathlib.Path(fresh))
        # v2 first, on purpose: the better model then becomes registered version 2,
        # so an alias hard-wired to version 1 names the wrong model.
        versions = {}
        for version in ("v2", "v1"):
            artefact = load_artefact(version)
            pipeline = as_pipeline(artefact)
            params = {"n_estimators": artefact["model"].n_estimators,
                      "max_depth": artefact["model"].max_depth,
                      "random_state": metrics[version]["seed"],
                      "train_rows": metrics[version]["trained_on_rows"]}
            logged = {"accuracy": float(metrics[version]["accuracy"]),
                      "size_bytes": float(metrics[version]["size_bytes"])}
            with contextlib.redirect_stderr(io.StringIO()):     # mlflow's "Created version" chatter
                returned = lab.log_training_run(version, pipeline, example,
                                                pipeline.predict_proba(example),
                                                params, logged)
            assert returned is not None and str(returned).isdigit(), (
                f"log_training_run returned {returned!r}; it should return the registered "
                "version number of the model it logged (info.registered_model_version)")
            versions[version] = str(returned)

        experiment = client.get_experiment_by_name("aboard-service")
        assert experiment is not None, "no experiment 'aboard-service' in the store"
        runs = {r.info.run_name: r for r in client.search_runs([experiment.experiment_id])}
        assert set(runs) >= {"v1", "v2"}, (
            f"the store holds runs named {sorted(runs)}; expected a run named v1 and one "
            "named v2 — start_run(run_name=name)")
        for version in ("v1", "v2"):
            recorded = runs[version].data.metrics
            assert "accuracy" in recorded, (
                f"run {version} carries no metric 'accuracy' — log_metrics(metrics)")
            close(recorded["accuracy"], metrics[version]["accuracy"], 1e-9,
                  f"run {version}'s logged accuracy is not the one the module measured")
            assert runs[version].data.params.get("n_estimators") is not None, (
                f"run {version} carries no parameters — log_params(params); months later "
                "'what was it trained with?' must be a lookup")

        registered = {str(mv.version) for mv in client.search_model_versions(
            f"name = '{REGISTERED_MODEL}'")}
        assert set(versions.values()) <= registered, (
            f"log_training_run returned versions {sorted(versions.values())} but the store "
            f"holds versions {sorted(registered)} of '{REGISTERED_MODEL}' — pass "
            "registered_model_name=REGISTERED_MODEL to log_model and return "
            "info.registered_model_version")

        best = max(metrics, key=lambda v: metrics[v]["accuracy"])
        worst = min(metrics, key=lambda v: metrics[v]["accuracy"])
        # Point the alias at the wrong model first, so that the move is a move.
        lab.promote_by_alias(client, REGISTERED_MODEL, versions[worst], CHAMPION)
        moved = lab.promote_by_alias(client, REGISTERED_MODEL, versions[best], CHAMPION)
        named = client.get_model_version_by_alias(REGISTERED_MODEL, CHAMPION)
        assert str(named.version) == versions[best], explain(
            "m3:alias:wrong-version",
            f"alias '{CHAMPION}' names version {named.version}",
            f"The more accurate model ({best}, accuracy {metrics[best]['accuracy']:.4f}) is "
            f"version {versions[best]}. Promotion is moving the alias onto the version that "
            "won the gate — and this check logs the better model first, so a version number "
            "written into your code names the wrong one.")
        assert moved is None or str(moved) == versions[best], (
            f"promote_by_alias returned {moved!r} but the alias names version {named.version}")

        with contextlib.redirect_stderr(io.StringIO()):
            champion = mlflow.pyfunc.load_model(f"models:/{REGISTERED_MODEL}@{CHAMPION}")
        schema = champion.metadata.get_input_schema()
        assert schema is not None, (
            "the champion was logged without a signature — infer_signature(X, y_pred) is what "
            "records the column names, types and order the platform will enforce")
        assert schema.input_names() == FEATURES, (
            f"the champion's signature names {schema.input_names()}; expected {FEATURES} in "
            "that order — infer it from the raw feature frame X, not from a numpy array")
        assert schema.required_input_names() == FEATURES, (
            f"the signature marks {sorted(set(FEATURES) - set(schema.required_input_names()))} "
            "optional; every feature is required at the model's door — infer the signature "
            "from rows where all four are present")



# --------------------------------------------------------------------------
# Part (c) — the verdict
# --------------------------------------------------------------------------
# Six sets of evidence whose right calls are not the same, and none of them can
# be answered by the quantity a dashboard shows. The numbers are this module's
# own measurements where it has one -- 0.818 and 0.7291 for the two accuracies,
# 1590 and 1745 for the two bills at the priced threshold, 1663 for the best
# policy that never asks a model, 94.92 megabytes against 0.37 -- so that a
# student recognises the case they are judging.
#
# The check owns the right call here, which Module 1's fitness verdict
# deliberately does not. The difference is that this module declares the rule on
# a slide and in the stub before there is a candidate: the release is priced in
# what the operator pays, the no-model floor has to be cleared as well as the
# model in service, a tie is not an improvement, and a service that has already
# regressed is repaired before it is changed.

BASE_EVIDENCE = {
    "candidate_accuracy": 0.7291,
    "approved_accuracy": 0.818,
    "approved_accuracy_at_promotion": 0.818,
    "candidate_realised_cost": 1590.0,
    "approved_realised_cost": 1590.0,
    "trivial_baseline_cost": 1663.0,
    "candidate_latency_p95": 41.0,
    "latency_budget": 120.0,
    "candidate_megabytes": 94.92,
    "approved_megabytes": 0.37,
}


def evidence(**changes):
    return {**BASE_EVIDENCE, **changes}


# name -> (evidence, the right call, the quantity that decided it)
VERDICTS = {
    # The module's whole argument: more accurate, and dearer where it is paid for.
    "more accurate and dearer": (
        evidence(candidate_accuracy=0.8624, candidate_realised_cost=1745.0),
        "hold", "candidate_realised_cost"),
    # Less accurate, and cheaper than both the model in service and the floor.
    "cheaper than both": (
        evidence(candidate_accuracy=0.7902, candidate_realised_cost=1421.0),
        "promote", "candidate_realised_cost"),
    # The model in service has fallen since the day the alias moved to it.
    "the service has regressed": (
        evidence(candidate_accuracy=0.7902, candidate_realised_cost=1421.0,
                 approved_accuracy=0.7314),
        "roll back", "approved_accuracy_at_promotion"),
    # Bigger, newer, more accurate -- and exactly level on the bill.
    "bigger and newer": (
        evidence(candidate_accuracy=0.8471, candidate_realised_cost=1590.0),
        "hold", "approved_realised_cost"),
    # Cheaper than what is running, dearer than answering without a model at all.
    "does not clear the floor": (
        evidence(candidate_realised_cost=1671.0, approved_realised_cost=1690.0),
        "hold", "trivial_baseline_cost"),
    # The cheapest of all, and outside the agreement its callers were given.
    "too slow to promise": (
        evidence(candidate_accuracy=0.8102, candidate_realised_cost=1402.0,
                 candidate_latency_p95=240.0),
        "hold", "candidate_latency_p95"),
}


def part_c(lab):
    calls = getattr(lab, "VERDICT_CALLS", None)
    assert isinstance(calls, (tuple, list)) and set(calls) == {"promote", "hold", "roll back"}, (
        f"VERDICT_CALLS is {calls!r}; the three calls are 'promote', 'hold' and 'roll back' "
        "and there is no fourth")
    for name in ("REGRESSION_MARGIN", "LATENCY_PERCENTILE"):
        value = getattr(lab, name, None)
        assert isinstance(value, (int, float)) and not isinstance(value, bool), (
            f"{name} is {value!r}; it is a number declared in the open, before there is a "
            "candidate anybody is fond of")
    declared = {"regression_margin": float(lab.REGRESSION_MARGIN),
                "latency_percentile": float(lab.LATENCY_PERCENTILE)}

    given, reasons = {}, {}
    for name, (facts, expected, deciding) in VERDICTS.items():
        handed = dict(facts)
        answer = lab.promotion_verdict(handed)
        assert isinstance(answer, tuple) and len(answer) == 2, (
            f"promotion_verdict() on {name!r} returned {answer!r}; it returns the pair "
            "(call, reason)")
        call, reason = answer
        assert call in calls, (
            f"promotion_verdict() called {name!r} {call!r}, which is not one of "
            f"{', '.join(repr(c) for c in calls)}")
        assert handed == facts, (
            f"promotion_verdict() edited the evidence it was handed, on {name!r}. A verdict "
            "reads its evidence; it does not change it.")
        again, _ = lab.promotion_verdict(dict(facts))
        assert again == call, (
            f"promotion_verdict() called {name!r} {call!r} and then {again!r} on the same "
            "evidence. A verdict that is not a function of its evidence cannot be defended.")
        # The call first, so that a wrong verdict is reported as a wrong verdict
        # rather than as a fault in the sentence arguing for it.
        assert call == expected, explain(
            f"m3:verdict:call:{name}",
            f"you called {name!r} {call!r} and the module's own rule calls it {expected!r}",
            "The rule is on the slide and in the stub, and it was agreed before there was "
            "a candidate: roll back if the model in service has fallen more than "
            "REGRESSION_MARGIN below what it scored at promotion; otherwise promote only if "
            "the candidate is cheaper than the model in service AND cheaper than the best "
            "policy that uses no model at all AND inside the latency budget; otherwise "
            "hold.")
        grade_reason(reason, {**facts, **declared}, key=f"m3:verdict:{name}", minimum_keys=2)
        spoken = str(reason).lower().replace("_", " ")
        assert deciding.replace("_", " ") in spoken, explain(
            f"m3:verdict:deciding:{name}",
            f"your reason for {name!r} never names {deciding}",
            f"On this evidence {deciding} is the quantity that settles the call — every "
            "other one points the other way or points nowhere. A reason that would fit any "
            "of these six is an argument about none of them.")
        given[name], reasons[name] = call, reason

    distinct = {str(reason).strip().lower() for reason in reasons.values()}
    assert len(distinct) == len(reasons), explain(
        "m3:verdict:same-reason",
        f"{len(reasons)} sets of evidence and {len(distinct)} distinct reason(s)",
        "One sentence cannot be the argument for a promotion and for a rollback. Build the "
        "reason out of the numbers you were handed, case by case.")

    # Four sets of evidence the student has never seen, each one number away from
    # a set they have. A table of remembered answers cannot survive them.
    moved = {
        # cheap enough, and now it clears both bars
        "the dear candidate made cheap":
            (evidence(candidate_accuracy=0.8624, candidate_realised_cost=1502.0),
             "promote"),
        # the cheap candidate made dearer than what is running
        "the cheap candidate made dear":
            (evidence(candidate_accuracy=0.7902, candidate_realised_cost=1601.0), "hold"),
        # a good candidate cannot rescue a service that has already fallen
        "a good candidate, a fallen service":
            (evidence(candidate_accuracy=0.7902, candidate_realised_cost=1421.0,
                      approved_accuracy=0.768), "roll back"),
        # exactly at the budget: the agreement says at most this slow, not slower
        "exactly at the budget":
            (evidence(candidate_accuracy=0.8102, candidate_realised_cost=1402.0,
                      candidate_latency_p95=120.0), "promote"),
    }
    for name, (facts, expected) in moved.items():
        call, _ = lab.promotion_verdict(dict(facts))
        assert call == expected, explain(
            f"m3:verdict:moved:{name}",
            f"one number was moved and you still called it {call!r}; the rule calls it "
            f"{expected!r} ({name})",
            "Each of these is one of the six sets of evidence with a single quantity "
            "changed. A verdict that answers the six and not these four is a table of "
            "remembered answers rather than a rule applied.")


def body(lab):
    metrics = load_metrics()
    assert metrics["v2"]["accuracy"] < metrics["v1"]["accuracy"], (
        "the fixture is wrong: v2 should measure worse than v1. Re-run "
        "`python3 service/models.py` and tell the instructor if it persists.")
    part_a(lab, metrics)
    part_b(lab, metrics)
    part_c(lab)


run(1, "01_the_gate", "promote_if_better", body)
