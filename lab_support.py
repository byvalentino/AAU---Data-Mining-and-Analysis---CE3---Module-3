"""What every Module 3 lab needs: the unsolved marker and the service's parts.

Module 3 serves what Module 2 built. The table is generated phone traces
calibrated from the archive; the vehicle telemetry slice is real. Neither
contains personal data, which is why this repository can be public.

The model is a **stand-in**, trained by setup.sh from Module 2's table. The four
modelling sessions between Module 2 and Module 3 produce the model this course
would rather serve; when that artefact is agreed it drops into
service/artefacts/ and nothing here changes. That is release by indirection,
which is also what block one teaches.

Two registries sit side by side, on purpose. `registry.json` and the pickles are
the fifty-line registry Lab 1 part (a) builds; `mlruns.db` and `mlartifacts/`
are the same idea kept the way industry keeps it -- the local MLflow store that
Lab 1 part (b) writes to and Lab 4 part (b) reads from. The helpers at the
bottom of this file open that store, load the model behind an alias, and wrap an
artefact as the pipeline the store registers.
"""
from __future__ import annotations

import logging
import pathlib
import subprocess
import sys

import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "data"))

# The store's names, so a lab, a check and a solution all say the same word.
REGISTERED_MODEL = "aboard"          # the registered model, one name for every version
CHAMPION = "champion"                # the alias the service loads; release moves it
CHALLENGER = "challenger"            # the alias of the candidate waiting at the gate
FEATURES = ["speed", "rssi1", "rssi2", "rssiC"]

# --------------------------------------------------------------------------
# The shape of the table Module 2 hands over
# --------------------------------------------------------------------------
# Module 2's Lab 4 ends by writing one table and one manifest that describes it:
# one row per phone per window, the stored transform's columns filled and
# scaled, a mask beside every column something was filled in and beside no
# other, the split point recorded as an INSTANT rather than a row number, and
# the fitted transform and the conservation ledger stored inside the manifest.
# Its schema version is 1.0 and its own copy lands in
# `Module 2/exercises/out/handoff/`, which is generated and never committed.
#
# `data/prepare.py` writes this module's copy under `data/handoff/`, so that
# this repository clones and runs on its own; the comment at the head of that
# function says what is faithfully Module 2's and what is this module's
# stand-in. Lab 2's contract is graded against `feature_columns` here, so a
# service whose door accepts a different set of fields from the table upstream
# promises fails check 2 rather than being discovered by somebody retraining.
HANDOFF_SCHEMA = "1.0"
HANDOFF_KEYS = ("schema_version", "rows", "columns", "key", "feature_columns",
                "mask_columns", "target", "split_point", "train_rows", "test_rows",
                "transform", "ledger")


class NotSolved(Exception):
    """A lab stub raises this. The check turns it into exit code 2.

    It is not an error. It means "you have not written this yet", which is a
    different state from "you wrote it and it is wrong", and the checks say so.
    """


class EnvironmentNotReady(Exception):
    """The tools or the data this module needs are missing. The checks exit 3.

    A third state, separate from the other two, because "this machine is not set
    up" is not "your code is wrong". A student told the second while the first is
    true goes hunting for a bug that was never there. Every check imports it from
    here, so the modules share one class rather than several with the same name.
    """


def _trained(artefact: str) -> None:
    """Build the stand-in models if setup.sh has never been run in this clone.

    Without this, a fresh clone reports all four labs as *wrong*: the artefacts
    are absent, loading one raises FileNotFoundError, and the harness prints
    exit 1 — its code for "you wrote it and it is not right yet". The fault is
    the missing setup, so say that instead.

    The build runs as a subprocess, exactly as setup.sh runs it, so that the
    MLflow store it writes is the same store setup.sh would have written -- the
    registered pipeline is pickled by value when service/models.py is the main
    script, and by reference to this directory when it is imported, and only the
    first can be read from anywhere.
    """
    from service import models

    if not (models.ARTEFACTS / artefact).exists() or not models.STORE.exists():
        logging.getLogger("lab_support").warning(
            "service/artefacts/%s or the MLflow store is missing -- building the "
            "stand-in models now; run  bash setup.sh  to do this once, properly", artefact)
        subprocess.run([sys.executable, str(models.HERE / "models.py")], check=False)
    if not (models.ARTEFACTS / artefact).exists():
        raise EnvironmentNotReady(
            f"service/artefacts/{artefact} is still missing after training the "
            "stand-in models")


def load_table() -> pd.DataFrame:
    """Module 2's output: one row per phone reading, split by time downstream.

    Reads data/phones_2020-01-22.parquet, which setup.sh prepared, and falls back
    to the generator with a warning when it is absent. Same rows either way.
    """
    from service.models import build_table
    return build_table()


def load_artefact(version: str = "v1") -> dict:
    """{"model": ..., "transform": ...} — the pair, never one without the other."""
    _trained(f"model_{version}.pkl")
    from service.models import load_artefact as _load
    return _load(version)


def load_metrics() -> dict:
    """What each version measured when it was trained. The gate reads this."""
    _trained("metrics.json")
    from service.models import load_metrics as _load
    return _load()


def load_handoff():
    """(table, manifest) — the table Module 2 hands over and the manifest describing it.

    Raises EnvironmentNotReady rather than FileNotFoundError when setup.sh has
    never run, because "your data is not there" and "your code is wrong" are
    different states and the checks keep them apart.
    """
    import json

    folder = HERE / "data" / "handoff"
    if not (folder / "manifest.json").exists():
        logging.getLogger("lab_support").warning(
            "data/handoff/ is missing -- preparing the datasets now; run  bash setup.sh  "
            "to do this once, properly")
        subprocess.run([sys.executable, str(HERE / "data" / "prepare.py")], check=False)
    if not (folder / "manifest.json").exists():
        raise EnvironmentNotReady(
            "data/handoff/manifest.json is still missing after running data/prepare.py")
    manifest = json.loads((folder / "manifest.json").read_text())
    missing = [key for key in HANDOFF_KEYS if key not in manifest]
    if missing:
        raise EnvironmentNotReady(
            f"data/handoff/manifest.json is not schema {HANDOFF_SCHEMA}: it has no "
            f"{', '.join(missing)}")
    return pd.read_parquet(folder / "table.parquet"), manifest


def registry_path() -> pathlib.Path:
    _trained("registry.json")
    return HERE / "service" / "artefacts" / "registry.json"


# --------------------------------------------------------------------------
# The MLflow store
# --------------------------------------------------------------------------

def open_store(root: pathlib.Path | None = None):
    """Point MLflow at a local store and return an MlflowClient for it.

    With no argument: the module's own store, exercises/mlruns.db, which
    setup.sh wrote and Lab 4 reads. With a directory: a fresh store there --
    what a check does, and what a solution does under out/, so that nothing
    logged for demonstration can disturb what the labs read.
    """
    from service.models import open_store as _open
    return _open(root)


def load_registered(alias: str = CHAMPION):
    """The model behind an alias in the module's store, as the platform serves it.

    Returns an MLflow pyfunc model. Ask it with a one-row pandas frame whose
    columns are the raw request fields, in any order: the signature restores
    the order, and refuses a column it was not trained with.
    """
    _trained("model_v1.pkl")
    from service.models import load_registered as _load
    return _load(alias)


def as_pipeline(artefact: dict):
    """An artefact as one scikit-learn object: the stored transform, then the forest.

    This is what the store registers -- the preparation travels with the model,
    so a named request is filled, scaled and ordered by the artefact itself.
    """
    from service.models import as_pipeline as _wrap
    return _wrap(artefact)


def environment_pins() -> list[str]:
    """The libraries the artefact needs, at the versions actually running."""
    from service.models import environment_pins as _pins
    return _pins()
