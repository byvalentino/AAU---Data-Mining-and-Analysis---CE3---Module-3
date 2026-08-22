#!/usr/bin/env python3
"""Train the two models this module serves, the transform that travels with them,
and record both in the module's local MLflow store.

    python3 service/models.py

**This is the stand-in.** The four modelling sessions that sit between Module 2
and Module 3 produce the model this course would rather serve. Until that
artefact is agreed and handed over, the labs need something to promote, refuse,
threshold and skew -- so this trains two small models from Module 2's own table,
deterministically, in a couple of seconds.

Nothing downstream knows the difference. `load_artefact()` returns the same shape
whatever produced it, so when a real artefact arrives it is dropped into
`service/artefacts/` and the labs are untouched. That is the whole point of
release by indirection, which is also what block one teaches.

Two models on purpose:

    champion    honest features only, and deliberately modest
    candidate   the same, larger and slower, and no better

Because the gate in Lab 1 has to refuse something, and refusing a *worse* model
is easy. Refusing a bigger, newer, more expensive model that is not actually
better is the decision people get wrong.

Two registries on purpose, too. The pickle registry (`registry.json`, a name
pointing at a version) is the fifty-line idea the student builds in Lab 1 part
(a). The MLflow store beside it (`mlruns.db` and `mlartifacts/`) is the same idea
as the industry keeps it: every training run recorded as it happens -- settings,
metrics, the environment, the data window and its checksum, the artefact -- with
a **model signature** (column names, types and order, inferred at logging and
enforced at prediction) and an alias, `champion`, that release moves. Lab 1 part
(b) writes the logging; Lab 4 part (b) shows what the signature refuses.
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import logging
import pathlib
import pickle
import shutil
import sys

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

HERE = pathlib.Path(__file__).resolve().parent
EXERCISES = HERE.parent
ARTEFACTS = HERE / "artefacts"
DATA = EXERCISES / "data"
sys.path.insert(0, str(DATA))

SEED = 20200122

# The declared input, in order. Order is part of the contract, not a detail --
# swap two columns and every request still succeeds and every answer is wrong.
#
# `stationary` is deliberately NOT here. In the generated table it is the target
# inverted, so a model given it scores 1.0000 and has learned nothing. Module 2's
# find_leaks() catches it alongside bus_id, which is the point of that lab.
FEATURES = ["speed", "rssi1", "rssi2", "rssiC"]
TARGET = "label2"
ABOARD = "IN"

# The service is trained on the labelled day. The second day carries no labels
# at all (Module 2 measured it), which is why Module 5 exists.
DAY = "2020-01-22"
PREPARED_TABLE = DATA / f"phones_{DAY}.parquet"

# The MLflow store: one sqlite file for tracking and registry, artefacts beside
# it. Both are gitignored -- setup.sh rebuilds them in seconds -- and both are
# rewritten from nothing by build(), so their content is deterministic apart from
# run identifiers and timestamps, which nothing here ever compares.
STORE = EXERCISES / "mlruns.db"
ARTEFACT_ROOT = EXERCISES / "mlartifacts"
EXPERIMENT = "aboard-service"
REGISTERED_MODEL = "aboard"
CHAMPION, CHALLENGER = "champion", "challenger"
METRIC = "accuracy"           # the agreed metric, agreed once, before any candidate


def build_table() -> pd.DataFrame:
    """Module 2's output: generated phone traces, one row per reading.

    Reads the Parquet that setup.sh prepared when it is there and falls back to
    the generator when it is not; the two are the same rows. The sort that
    follows is kept exactly as it was when the models were first measured: the
    train/test split is positional, so the row order is part of every number on
    the slides, and it must not move.
    """
    if PREPARED_TABLE.exists():
        phones = pd.read_parquet(PREPARED_TABLE)
    else:
        logging.getLogger("service").warning(
            "data/%s is missing -- generating the table instead; run  bash setup.sh",
            PREPARED_TABLE.name)
        from make_phones import generate
        phones = generate(day=DAY, with_truth=False)
    return phones.sort_values("timestamp_utc").reset_index(drop=True)


def fit_transform(train: pd.DataFrame) -> dict:
    """Module 2's fitted transform, stored so the service prepares input identically."""
    return {
        "features": list(FEATURES),
        "medians": {c: float(train[c].median()) for c in FEATURES},
        "means": {c: float(train[c].mean()) for c in FEATURES},
        "stds": {c: float(train[c].std(ddof=0)) or 1.0 for c in FEATURES},
    }


def apply_transform(frame: pd.DataFrame, fitted: dict) -> pd.DataFrame:
    """Fill and scale with the stored constants, in the stored order.

    The one implementation of the preparation. The single-request path, the
    batch path and the registered pipeline all call this; two implementations
    would be two things that must stay correct forever.
    """
    out = pd.DataFrame(index=frame.index)
    for column in fitted["features"]:
        values = pd.Series(frame[column], index=frame.index, dtype="float64")
        values = values.fillna(fitted["medians"][column])
        out[column] = (values - fitted["means"][column]) / fitted["stds"][column]
    return out[fitted["features"]]


class StoredTransform(BaseEstimator, TransformerMixin):
    """The stored transform as a scikit-learn step, so it travels with the model.

    A registered model is asked with a *named* frame. This step reads the
    columns by name, fills an absent value from the stored median, scales with
    the stored mean and standard deviation, and hands the forest the four
    columns in the stored order -- whatever order the caller sent them in. It
    fits nothing: the constants were fitted on the training rows by
    fit_transform() and are carried inside the object.
    """

    def __init__(self, fitted: dict):
        self.fitted = fitted

    def fit(self, X, y=None):
        return self                                    # already fitted, deliberately

    def transform(self, X):
        return apply_transform(pd.DataFrame(X), self.fitted).to_numpy()


def as_pipeline(artefact: dict) -> Pipeline:
    """The artefact as one object: preparation first, then the forest."""
    return Pipeline([("prepare", StoredTransform(artefact["transform"])),
                     ("forest", artefact["model"])])


def train_and_save() -> dict:
    """Train v1 and v2, pickle each with its transform, write metrics.json and registry.json."""
    ARTEFACTS.mkdir(parents=True, exist_ok=True)
    table = build_table()

    # Split by time, because Module 2 measured why a random split lies.
    cut = int(len(table) * 0.7)
    train, test = table.iloc[:cut], table.iloc[cut:]

    fitted = fit_transform(train)
    x_train, x_test = apply_transform(train, fitted), apply_transform(test, fitted)
    y_train = (train[TARGET] == ABOARD).astype(int)
    y_test = (test[TARGET] == ABOARD).astype(int)

    models = {
        # The champion: a shallow forest. Modest, small, fast.
        "v1": RandomForestClassifier(n_estimators=100, max_depth=6,
                                     random_state=SEED, n_jobs=1),
        # The candidate: five times the trees and no depth limit. More capacity,
        # more parameters, more disk, more latency -- and *worse* on data from a
        # later stretch of time, because the extra capacity went into memorising
        # the training period. This is the model the gate has to refuse, and
        # refusing it is harder than refusing something obviously broken.
        "v2": RandomForestClassifier(n_estimators=500, max_depth=None,
                                     random_state=SEED, n_jobs=1),
    }

    record = {}
    for version, model in models.items():
        # Fitted on plain arrays: a served request is a list of numbers, not a
        # named frame, and fitting on names makes sklearn warn on every call.
        model.fit(x_train.to_numpy(), y_train)
        accuracy = float((model.predict(x_test.to_numpy()) == y_test).mean())
        path = ARTEFACTS / f"model_{version}.pkl"
        path.write_bytes(pickle.dumps({"model": model, "transform": fitted}))
        record[version] = {
            "version": version,
            "kind": type(model).__name__,
            "accuracy": round(accuracy, 4),
            "size_bytes": path.stat().st_size,
            "trained_on_rows": int(len(train)),
            "features": list(FEATURES),
            "seed": SEED,
        }

    (ARTEFACTS / "metrics.json").write_text(json.dumps(record, indent=1))
    # The registry: a name pointing at a version. Release is moving the pointer.
    (ARTEFACTS / "registry.json").write_text(json.dumps(
        {"approved": "v1", "history": ["v1"]}, indent=1))
    return record


# --------------------------------------------------------------------------
# The MLflow store
# --------------------------------------------------------------------------

def open_store(root: pathlib.Path | None = None):
    """Point MLflow at a local store and return a client for it.

    Tracking and registry share one sqlite file, `<root>/mlruns.db`; artefacts
    go under `<root>/mlartifacts/`. No server, no network. The default root is
    the exercises directory -- the module's own store, written by build() -- and
    a check or a solution passes a temporary directory instead, so that nothing
    it logs can disturb what the labs read.

    Two quiet-downs, both deliberate. MLflow resets its own logger to INFO when
    it is imported, so the level is set after the import, not before. And the
    first touch of a new store runs a schema migration that narrates itself on
    stderr through a logging config file that also installs a handler on the
    root logger; that narration is not this module's story, so it goes to a sink
    and the handler it left behind is removed.
    """
    import mlflow
    from mlflow.tracking import MlflowClient

    root = EXERCISES if root is None else pathlib.Path(root)
    root.mkdir(parents=True, exist_ok=True)
    for noisy in ("mlflow", "alembic"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    uri = "sqlite:///" + str(root / "mlruns.db")
    root_logger = logging.getLogger()
    handlers_before = list(root_logger.handlers)
    with contextlib.redirect_stderr(io.StringIO()):
        mlflow.set_tracking_uri(uri)
        mlflow.set_registry_uri(uri)
        if mlflow.get_experiment_by_name(EXPERIMENT) is None:
            mlflow.create_experiment(
                EXPERIMENT, artifact_location=(root / "mlartifacts").as_uri())
    for handler in root_logger.handlers:
        if handler not in handlers_before:
            root_logger.removeHandler(handler)
    logging.getLogger("alembic").setLevel(logging.WARNING)
    mlflow.set_experiment(EXPERIMENT)
    return MlflowClient()


def environment_pins() -> list[str]:
    """The libraries the artefact needs, at the versions actually running.

    Recorded from the process that trained the model, not reconstructed from a
    requirements file that may or may not be what was installed. Months later,
    "which scikit-learn unpickles this forest?" is then a lookup.
    """
    import cloudpickle
    import sklearn
    return [f"scikit-learn=={sklearn.__version__}", f"pandas=={pd.__version__}",
            f"numpy=={np.__version__}", f"cloudpickle=={cloudpickle.__version__}"]


def data_checksum(frame: pd.DataFrame) -> str:
    """A hash of the values the model was fitted on, so the run names its data."""
    values = pd.util.hash_pandas_object(frame, index=False).to_numpy()
    return hashlib.sha256(values.tobytes()).hexdigest()


def signature_rows(frame: pd.DataFrame, rows: int = 200) -> pd.DataFrame:
    """Raw rows with every feature present, to infer the signature from.

    A column that is absent in the example rows is inferred as optional, and an
    optional column may be left out of a request without complaint. Every one of
    the four features is required at the model's door -- an absent *value* is a
    NaN inside a present column, and the stored median fills it -- so the
    signature is inferred from rows where all four are present.
    """
    return frame[FEATURES].dropna().head(rows).reset_index(drop=True)


def log_to_registry(record: dict) -> dict:
    """Log v1 and v2 as runs of the local store, register both, alias the better one.

    What a training run records, and why: parameters (so the settings are not
    reconstructed from a notebook that has moved on), the agreed metric (so the
    gate has a number), the environment (so the artefact can be unpickled), the
    data window and its checksum (so "trained on what?" is a lookup), and the
    artefact itself with its signature -- names, types, order -- which is the
    mechanism that stops columns being switched around at serving time.
    """
    import mlflow
    from mlflow.models import infer_signature

    client = open_store()
    table = build_table()
    cut = int(len(table) * 0.7)
    train, test = table.iloc[:cut], table.iloc[cut:]
    example = signature_rows(train)

    logged = {}
    for version, facts in record.items():
        artefact = load_artefact(version)
        pipeline = as_pipeline(artefact)
        # Written as the run happens, not reconstructed afterwards.
        params = {
            "n_estimators": artefact["model"].n_estimators,
            "max_depth": artefact["model"].max_depth,
            "random_state": SEED,
            "features": ",".join(FEATURES),
            "split": "first 70 per cent of rows by time",
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            "data_window": f"{DAY}, generated phone traces, seed {SEED}",
            "data_checksum": data_checksum(train),
            "metric": METRIC,
        }
        metrics = {
            METRIC: float(facts["accuracy"]),
            "size_bytes": float(facts["size_bytes"]),
            "trees": float(len(artefact["model"].estimators_)),
        }
        with contextlib.redirect_stderr(io.StringIO()), mlflow.start_run(run_name=version):
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            info = mlflow.sklearn.log_model(
                pipeline, artifact_path="model",
                signature=infer_signature(example, pipeline.predict_proba(example)),
                input_example=example.head(2),
                registered_model_name=REGISTERED_MODEL,
                # The service needs the probability, not the label.
                pyfunc_predict_fn="predict_proba",
                pip_requirements=environment_pins())
        logged[version] = {"registered_version": int(info.registered_model_version),
                           "accuracy": float(facts["accuracy"])}

    # The gate, applied to the store: the champion is the better of the two on
    # the agreed metric, and a tie keeps what came first. The candidate is aliased
    # too, so that "challenger" is a name and not a file.
    versions = list(logged)
    champion = versions[0]
    for version in versions[1:]:
        if logged[version]["accuracy"] > logged[champion]["accuracy"]:
            champion = version
    challenger = next(v for v in versions if v != champion)
    client.set_registered_model_alias(REGISTERED_MODEL, CHAMPION,
                                      logged[champion]["registered_version"])
    client.set_registered_model_alias(REGISTERED_MODEL, CHALLENGER,
                                      logged[challenger]["registered_version"])
    return logged


def reset_store() -> None:
    """Start the store from nothing, so that its content is the same on every build."""
    if STORE.exists():
        STORE.unlink()
    if ARTEFACT_ROOT.exists():
        shutil.rmtree(ARTEFACT_ROOT)


def build() -> dict:
    """Everything setup.sh needs from this file: pickles, metrics, both registries.

    It ends by loading both aliases once. That is a smoke test -- if the champion
    cannot be read back from a fresh process, setup should fail here rather than
    in a student's check -- and it also settles the store's contents: MLflow
    writes a small `registered_model_meta` marker into a version's artefact
    directory the first time that version is loaded through `models:/name@alias`.
    Counting the files before that first load records a number no lab will ever
    see again, and check 0 would then report the store as tampered with the first
    time Lab 4 asks the registry a question.
    """
    record = train_and_save()
    reset_store()
    logged = log_to_registry(record)
    for version in record:
        record[version]["mlflow_version"] = logged[version]["registered_version"]
    for alias in (CHAMPION, CHALLENGER):
        load_registered(alias)
    return record


def load_artefact(version: str) -> dict:
    """{"model": ..., "transform": ...} — the pair, never one without the other."""
    return pickle.loads((ARTEFACTS / f"model_{version}.pkl").read_bytes())


def load_metrics() -> dict:
    return json.loads((ARTEFACTS / "metrics.json").read_text())


def load_registered(alias: str = CHAMPION):
    """The model behind an alias, as the platform serves it: schema enforced."""
    import mlflow
    open_store()
    with contextlib.redirect_stderr(io.StringIO()):
        return mlflow.pyfunc.load_model(f"models:/{REGISTERED_MODEL}@{alias}")


if __name__ == "__main__":
    for version, facts in build().items():
        print(f"{version}  {facts['kind']:24} accuracy {facts['accuracy']:.4f}  "
              f"{facts['size_bytes']:>9,} bytes  registered as version "
              f"{facts['mlflow_version']}")
