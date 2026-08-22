#!/usr/bin/env python3
"""Prepare every dataset and artefact the labs read, and write data/MANIFEST.json.

    python3 data/prepare.py          called by setup.sh; safe to re-run

Generated data is generated here, once, deterministically (seed 20200122), and
written to data/ as Parquet, so that (a) a lab never depends on a generator
running at import time, (b) `make check` can verify the data is present before
grading anything (verify/check_00_data.py), and (c) a teacher can open the file
a student worked on. Shipped data (the archive slice) is verified, not rewritten.

The manifest records, per file, the row count, the column count and a hash of
the values -- not of the bytes, because Parquet metadata carries a writer string.
"""
from __future__ import annotations

import json
import pathlib
import subprocess
import sys

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent          # exercises/data
EXERCISES = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(EXERCISES))

MANIFEST = HERE / "MANIFEST.json"
SLICE = HERE / "bus_slice.csv.gz"


def content_hash(frame: pd.DataFrame) -> str:
    return f"{int(pd.util.hash_pandas_object(frame, index=False).sum()):x}"


def table_entry(path: pathlib.Path, frame: pd.DataFrame, source: str, note: str) -> dict:
    return {"path": str(path.relative_to(EXERCISES)), "kind": "table", "source": source,
            "rows": int(len(frame)), "columns": int(frame.shape[1]),
            "content_hash": content_hash(frame), "note": note}



# --------------------------------------------------------------------------
# The hand-off from Module 2
# --------------------------------------------------------------------------
# Module 2's Lab 4 ends by writing one table and one manifest describing it --
# `handoff_table(aligned, fitted, ledger, target, train_fraction)`, schema
# version 1.0, landing in `Module 2/exercises/out/handoff/`. That file is the
# only thing Module 2 produces which another person ever opens, and this module
# is the first person to open it.
#
# It is rebuilt here rather than copied for two reasons. A student clones this
# repository without Module 2's `out/` directory, which is generated and
# gitignored; and the shape has to be verifiable by check 0 in this module's own
# terms. So this writes Module 2's shape over this module's own day: one row per
# phone per window, the stored transform's columns filled and scaled, a mask
# beside every column something was filled in and beside no other, the split
# point recorded as an instant, and the transform and the ledger inside the
# manifest.
#
# The window here is the phone's own reporting instant, which is what makes
# (phone_id, window) unique: these phones report every 0.993 seconds, so a
# one-second floor would occasionally put two readings of one phone in one
# window. The ledger records that choice, as Module 2's does.
#
# WHAT IS HONESTLY DIFFERENT, and it is the open handover this module's DONE.md
# names: Module 2's own demonstration aggregates to a five-second grain and
# names its feature columns phone_speed, rssi1, rssi2 and bus_speed, with a
# target that is the SHARE of a window's readings that were aboard. This
# module's stand-in service is fitted on the per-reading columns speed, rssi1,
# rssi2 and rssiC with a binary target, because the four modelling sessions that
# sit between Module 2 and Module 3 are what decide the final feature set. When
# that decision is made, this file is replaced by Module 2's own, and Lab 2's
# contract and check 2 follow its `feature_columns` without a line of the
# service changing. That is release by indirection applied to a schema, and it
# is why the check compares the two rather than trusting either.
HANDOFF_SCHEMA = "1.0"
HANDOFF = HERE / "handoff"


def build_handoff() -> pd.DataFrame:
    """Module 2's schema, over this module's day. Returns the table it wrote."""
    import json as _json
    from service.models import (FEATURES, TARGET, ABOARD, build_table, fit_transform,
                                DAY, SEED)

    table = build_table()
    cut = int(len(table) * 0.7)
    fitted = fit_transform(table.iloc[:cut])

    out = pd.DataFrame({"phone_id": table["phone_id"].to_numpy(),
                        "window": table["timestamp_utc"].to_numpy()})
    mask_columns = []
    for column in fitted["features"]:
        values = pd.Series(table[column].to_numpy(), dtype="float64")
        absent = values.isna()
        filled = values.fillna(fitted["medians"][column])
        out[column] = ((filled - fitted["means"][column]) / fitted["stds"][column]).to_numpy()
        # The mask goes beside the value, and only where the fill did something.
        # A mask of all noughts on a complete column is a column of noise.
        if bool(absent.any()):
            name = f"{column}_missing"
            out[name] = absent.astype(int).to_numpy()
            mask_columns.append(name)
    out[TARGET] = (table[TARGET].to_numpy() == ABOARD).astype(int)

    # The split point is an instant, not a row number: rows are reordered,
    # filtered and appended, and instants are not. Every row sharing the cut
    # instant lands on the same side, which is why the cut is moved to the start
    # of the window the row at the fraction sits in.
    split_point = out["window"].iloc[min(cut, len(out) - 1)]
    trains = out["window"] < split_point
    out["split"] = np.where(trains, "train", "test")

    HANDOFF.mkdir(parents=True, exist_ok=True)
    out.to_parquet(HANDOFF / "table.parquet", index=False)
    manifest = {
        "schema_version": HANDOFF_SCHEMA,
        "source": "Module 2, Lab 4, handoff_table(); rebuilt here by data/prepare.py "
                  "over Module 3's own day so that this repository clones on its own",
        "rows": int(len(out)),
        "columns": int(out.shape[1]),
        "key": ["phone_id", "window"],
        "feature_columns": list(fitted["features"]),
        "mask_columns": mask_columns,
        "target": TARGET,
        "split_point": str(split_point),
        "train_fraction": 0.7,
        "train_rows": int(trains.sum()),
        "test_rows": int((~trains).sum()),
        "transform": fitted,
        "ledger": {"grain": "the phone's own reporting instant, about 0.993 seconds apart",
                   "phone_rows_in": int(len(table)),
                   "rows_out": int(len(out)),
                   "rows_dropped": 0,
                   "drop_reasons": {},
                   "day": DAY,
                   "seed": SEED,
                   # Said out loud rather than smoothed over. service/models.py
                   # cuts the training set POSITIONALLY at 70 per cent of the
                   # rows, and that row number falls inside an instant: twelve
                   # phones report together, so eleven of that instant's rows
                   # train and one tests. This table applies Module 2's rule
                   # instead -- the split point is the instant itself, and every
                   # row sharing it lands on the test side -- which is why the
                   # two differ by eleven rows. The service's models were
                   # measured before this table existed and are not retrained to
                   # match; the difference is 0.1 per cent of the rows, and it is
                   # named in DONE.md rather than left for somebody to find.
                   "split_rule": "the split point is an instant; no window straddles it",
                   "service_positional_cut_rows": int(cut)},
    }
    (HANDOFF / "manifest.json").write_text(_json.dumps(manifest, indent=1, default=str))
    return out


def main() -> int:
    files = []

    # 1. The archive slice ships with every module; verify it, never rewrite it.
    if not SLICE.exists():
        print(f"prepare failed: {SLICE.relative_to(EXERCISES)} is missing from the checkout")
        return 1
    bus = pd.read_csv(SLICE, low_memory=False)
    files.append(table_entry(SLICE, bus, "archive",
                             "vehicle VJRD1A10224000055, 22-23 January 2020, as shipped"))
    print(f"verified  {SLICE.name}: {len(bus)} rows x {bus.shape[1]} columns")

    # 2. Generated phone traces, where the module has the generator.
    if (HERE / "make_phones.py").exists():
        from make_phones import generate, CALIBRATION
        for day in CALIBRATION["phones_per_day"]:
            frame = generate(day=day, with_truth=False)
            out = HERE / f"phones_{day}.parquet"
            frame.to_parquet(out, index=False)
            files.append(table_entry(out, frame, "generated",
                                     f"make_phones.generate(day={day!r}), seed 20200122"))
            print(f"generated {out.name}: {len(frame)} rows x {frame.shape[1]} columns")

    # 3. The generated stream, where the module has a world.
    if (EXERCISES / "service" / "world.py").exists():
        from service import world
        frame = world.stream()
        out = HERE / "stream.parquet"
        frame.to_parquet(out, index=False)
        files.append(table_entry(out, frame, "generated",
                                 "service.world.stream(), 28 days, seed 20200122"))
        print(f"generated {out.name}: {len(frame)} rows x {frame.shape[1]} columns")

    # 4. Trained artefacts, where the module serves a model. service/models.py
    #    trains the two forests, pickles each with its transform, and records
    #    both as runs of the local MLflow store (mlruns.db + mlartifacts/), with
    #    the alias `champion` on the better one. It runs as a script on purpose:
    #    the pipeline class it registers is then pickled by value, so the store
    #    can be read by any process without this directory on its import path.
    if (EXERCISES / "service" / "models.py").exists():
        result = subprocess.run([sys.executable, str(EXERCISES / "service" / "models.py")],
                                capture_output=True, text=True)
        if result.returncode != 0:
            print("prepare failed: service/models.py did not train --")
            print(result.stdout[-1500:], result.stderr[-1500:])
            return 1
        artefacts = EXERCISES / "service" / "artefacts"
        for path in sorted(artefacts.glob("*")):
            if path.is_file():
                files.append({"path": str(path.relative_to(EXERCISES)), "kind": "artefact",
                              "source": "generated", "bytes": path.stat().st_size,
                              "note": "written by service/models.py"})
        print(f"trained   {len([f for f in files if f['kind'] == 'artefact'])} artefact(s) "
              f"in service/artefacts/")
        store, artefact_root = EXERCISES / "mlruns.db", EXERCISES / "mlartifacts"
        if not store.exists() or not any(artefact_root.rglob("MLmodel")):
            print("prepare failed: service/models.py did not write the MLflow store "
                  "(mlruns.db and mlartifacts/)")
            return 1
        files.append({"path": "mlruns.db", "kind": "artefact", "source": "generated",
                      "bytes": store.stat().st_size,
                      "note": "MLflow tracking and registry store, sqlite, written by "
                              "service/models.py -- two runs (v1, v2), registered model "
                              "'aboard', alias champion on the better version"})
        files.append({"path": "mlartifacts", "kind": "directory", "source": "generated",
                      "files": sum(1 for p in artefact_root.rglob("*") if p.is_file()),
                      "note": "MLflow artefacts: both registered models with signature, "
                              "input example, pinned environment, and the "
                              "registered_model_meta marker MLflow writes when a version "
                              "is first loaded through models:/aboard@alias -- build() "
                              "loads both aliases, so this count is the one a lab sees"})
        print("recorded  mlruns.db and mlartifacts/ (the MLflow store)")

        # The hand-off from Module 2, in Module 2's own schema. Written after the
        # transform exists, because the manifest carries it.
        handoff = build_handoff()
        files.append(table_entry(HANDOFF / "table.parquet", handoff, "generated",
                                 "the table Module 2 hands over, schema " + HANDOFF_SCHEMA
                                 + " -- one row per phone per window, a mask beside every "
                                   "filled value, the split point in the manifest"))
        files.append({"path": "data/handoff/manifest.json", "kind": "artefact",
                      "source": "generated",
                      "bytes": (HANDOFF / "manifest.json").stat().st_size,
                      "note": "the hand-off manifest: schema version, key, feature columns, "
                              "mask columns, target, split point, the stored transform and "
                              "the ledger -- Lab 2's contract is graded against its "
                              "feature_columns"})
        print(f"built     data/handoff/ -- {len(handoff)} rows x {handoff.shape[1]} columns, "
              f"schema {HANDOFF_SCHEMA}")

    MANIFEST.write_text(json.dumps({"seed": 20200122, "files": files}, indent=1))
    print(f"wrote     data/MANIFEST.json -- {len(files)} entries")
    return 0


if __name__ == "__main__":
    sys.exit(main())
