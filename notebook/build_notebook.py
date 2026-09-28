#!/usr/bin/env python3
"""Build Module 3's demonstration notebook, and execute it.

    python "Module 3/notebook/build_notebook.py"            build and run
    python "Module 3/notebook/build_notebook.py" --no-run   build only

Rewritten 28 September 2026 to the specification Module 4's notebook set. Module 3
is one notebook in three parts, as it is three decks: 3.0 (beer and diapers),
3.1 (deploying a model as a service I) and 3.2 (II). It holds itself to three
things:

  every image on a shown slide is drawn here by Python -- the 3.0 figures by the
      deck's own script on the deck's own simulation, the 3.1 and 3.2 figures by
      slides/make_figs.py's drawing functions fed the service trained here, the
      rest recomputed on the lab data or redrawn and labelled "illustrative"
      (notebook/figure_map.json says which, image by image);
  every lab exercise is stated as its stub states it, and its solution runs with
      every function's code visible -- copied verbatim from exercises/, the
      shipped machinery included, and checked against the files by
      tools/check_notebook_sources.py;
  every number the slides print is recomputed, and agrees() stops the run if
      the deck and the code disagree; beside() prints the few that differ, with
      the reason.

Data: the 3.0 receipts are simulated by slides/simulate_receipts.py (seed
20260927); 3.1 and 3.2 run on the phone traces data/make_phones.py generates
(seed 20200122), the two forests service/models.py trains on them, the MLflow
store it writes, and the vehicle slice data/bus_slice.csv.gz. The notebook copies
exercises/ to a temporary directory and works there, so a student's registry,
store and out/ folder are never touched. Executed from `Module 3/exercises`.
Needs the lab requirements plus notebook/requirements.txt.

One helper here goes beyond tools/notebook_kit.py: `lines()`. Two of the deck's
scripts, slides/simulate_receipts.py and slides/make_figs_3.0.py, are written as
top-level statements rather than functions, which the kit's extractor cannot
address by name. `lines()` copies a run of their lines verbatim at build time
and tags the cell `lines` with the file and the line numbers. Check 3i does not
re-verify that kind; rebuilding the notebook does.
"""
from __future__ import annotations

import sys
from pathlib import Path

from nbformat.v4 import new_code_cell

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(HERE))

from notebook_kit import Notebook, execute  # noqa: E402

OUTPUT = HERE / "Module3_demonstration.ipynb"
EXERCISES = HERE.parent / "exercises"

nb = Notebook(3, HERE / "references.json")


# The shape of every explanation cell above a code cell.
def explain(goal: str, why: str, what: str, so_what: str, extra: str = "") -> None:
    text = (f"**Goal.** {goal}\n\n**Why.** {why}\n\n**What the code does.** {what}\n\n"
            f"**So what.** {so_what}")
    nb.md(text + (f"\n\n{extra}" if extra else ""))


def _span(relpath: str, first: str, last: str) -> tuple[int, int, str]:
    """The lines of `relpath` from the first one starting with `first` to the first
    one after it containing `last`, verbatim (1-based line numbers)."""
    text = (ROOT / relpath).read_text(encoding="utf-8").splitlines()
    start = next(i for i, line in enumerate(text) if line.startswith(first))
    end = next(i for i in range(start, len(text)) if last in text[i])
    return start + 1, end + 1, "\n".join(text[start:end + 1])


def lines(relpath: str, first: str, last: str, cite: str = "") -> None:
    """A code cell holding a run of a script's lines, verbatim (see the module docstring)."""
    a, b, body = _span(relpath, first, last)
    header = f"# Source: {relpath} — lines {a}–{b}, verbatim"
    if cite:
        header += "\n# References: " + nb.references.render(cite)
    cell = new_code_cell(header + "\n\n" + body)
    cell.metadata["course"] = {"kind": "lines", "file": relpath, "lines": [a, b]}
    nb.cells.append(cell)


def figure_lines(names, relpath: str, first: str, last: str, show: str,
                 slides: list[str], treatment: str) -> None:
    """A figure cell: a run of a figure script's lines, verbatim, then one line that
    shows the picture those lines saved."""
    a, b, body = _span(relpath, first, last)
    text = (f"# Source: {relpath} — lines {a}–{b}, verbatim; the last line shows the "
            f"picture they saved\n\n{body}\n\n{show}")
    nb.figure(names, text, slides=slides, treatment=treatment)


# =============================================================================
# Front matter and set-up
# =============================================================================

def front_matter() -> None:
    nb.md("""
    # Module 3 — From a pattern to a decision, and a model as a service

    **Data Mining and Analysis (course code CE3) · Aalborg University, Copenhagen**

    Module 3 is three decks and this is one notebook in three parts, in the decks'
    order.

    | Part | Deck | The question | Laboratory |
    |---|---|---|---|
    | 3.0 | `Module3.0.pptx` — Beer and diapers | What does a rise in sales need before a cause can be named? | none; the receipts are simulated |
    | 3.1 | `Module3.1.pptx` — Deploying a model as a service I | Which version answered, is the request possible, and what does a mistake cost? | Lab 1 — the gate; Lab 2 — the contract; Lab 3 — the threshold |
    | 3.2 | `Module3.2.pptx` — Deploying a model as a service II | Does the served model give the trained answer, and in time? | Lab 4 — skew and honest speed |

    **How to read it.** Every code cell has a short note above it: the *goal*, *why*
    it is done, *what the code does*, and *so what* — what the result lets you say.
    Cells that begin `# Source: … verbatim` are copied from the repository's files
    — the lab solutions, the machinery they run on, and the deck's own figure
    scripts — so the code you read is the code that runs. Cells that draw a figure
    name the slide they reproduce. Formulas are written in Python with `sympy` and
    displayed as LaTeX. A line `deck … computed …` is a number a slide prints,
    recomputed; if the two ever disagree the notebook stops.

    **How to run it.** From `Module 3/exercises`, after `bash setup.sh` and
    `pip install -r ../notebook/requirements.txt`. It runs in a few minutes; most of
    that is MLflow logging a 95-megabyte candidate model.

    **Data.** Part 3.0 has no real dataset: its receipts are simulated by
    `slides/simulate_receipts.py` (seed 20260927), and every number on its slides
    comes from that simulation. Parts 3.1 and 3.2 run on what the labs run on:
    phone traces generated by `exercises/data/make_phones.py` from the archive's
    measured rates (seed 20200122, 10,800 rows for 22 January), the two forests
    `exercises/service/models.py` trains on them, the MLflow store it writes, and
    `exercises/data/bus_slice.csv.gz` — shuttle VJRD1A10224000055 on 22 and 23
    January 2020, 48,290 readings. No row describes a person.
    """)


def setup() -> None:
    nb.md("## Set-up")
    explain(
        "Load the libraries, and define the small tools every later cell uses.",
        "A notebook that claims to reproduce three decks needs a way to show a figure, "
        "a formula and a number side by side with what the slide printed.",
        "`show()` renders a plotly figure to a static PNG embedded in the notebook, so it "
        "displays anywhere, including offline; `show_png()` embeds a picture a script "
        "saved. `formula()` displays sympy expressions as LaTeX. `agrees()` prints a "
        "number the deck states beside the same number computed here and stops the run "
        "when they differ; `beside()` prints two numbers that are expected to differ, "
        "with the reason. MLflow's download progress bars are switched off, because they "
        "print timings that change on every run.",
        "If a deck and the code ever disagree, this notebook fails to run rather than "
        "quietly showing a different number.")
    nb.code('''
    import json
    import math
    import os
    import shutil
    import sys
    import tempfile
    import types
    import warnings
    from pathlib import Path

    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import sympy as sp
    from scipy import stats
    from IPython.display import Image, Math, display

    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning, module="mlflow")
    os.environ["MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR"] = "false"
    pd.set_option("display.width", 120)

    NAVY, GREEN_OK = "#1F2A5A", "#2E8B57"


    def show(fig, name, width=1000, height=560):
        """Draw a plotly figure as a static image inside the notebook."""
        fig.update_layout(template="plotly_white", width=width, height=height)
        if fig.layout.margin.t is None:          # unless the figure asked for its own room
            fig.update_layout(margin=dict(l=70, r=30, t=70, b=60))
        display(Image(fig.to_image(format="png", width=width, height=height, scale=1)))


    def show_png(path):
        """Embed a picture a script saved to disk."""
        display(Image(filename=str(path)))


    def formula(*parts):
        """Display sympy expressions (or LaTeX strings) side by side."""
        pieces = [p if isinstance(p, str) else sp.latex(p, order="none") for p in parts]
        display(Math(r"\\qquad ".join(pieces)))


    def agrees(what, computed, stated, places):
        """A number the deck states, recomputed here. Stops the run on a mismatch."""
        mine = round(float(computed), places)
        if abs(mine - float(stated)) > 0.5 * 10 ** -places + 1e-12:
            raise AssertionError(f"{what}: the deck says {stated}, the code gives {mine}")
        print(f"  {what:<66} deck {stated:<10} computed {int(mine) if places == 0 else mine}")


    DIFFERENCES = []   # every beside() call, gathered for the closing table


    def beside(what, computed, stated, why):
        """Two numbers that are expected to differ, printed with the reason."""
        DIFFERENCES.append((what, str(computed), str(stated), why))
        print(f"  {what}: computed here {computed}, stated {stated} — {why}")
    ''')

    nb.md("""
    ### A working copy of the exercises

    The labs write: a registry file, a local MLflow store, figures under `out/`.
    A notebook that ran the solutions inside `exercises/` would overwrite whatever a
    student has there. So the first thing it does is copy the exercises folder —
    the sources only — to a temporary directory and move into it.
    """)
    explain(
        "Work in a copy of `exercises/`, never in the student's folder.",
        "Lab 1 moves a registry pointer and logs runs to MLflow; Lab 4 reads the store; "
        "the solutions write under `out/`. None of that may touch a student's own files.",
        "Copies the folder without anything a run generates (the Parquet tables, the "
        "hand-off, the trained artefacts, the MLflow store, `out/`), then changes the "
        "working directory to the copy. Every path the labs use is relative to "
        "`exercises/`, so the verbatim code below works unchanged; the generated files are "
        "rebuilt from nothing further down, exactly as `setup.sh` builds them.",
        "Every number below comes from a store this notebook built itself, the same on "
        "every run, and your `exercises/` is exactly as you left it.")
    nb.code('''
    SOURCE = Path.cwd()
    assert (SOURCE / "lab_support.py").exists(), "run this notebook from Module 3/exercises"
    SLIDES = SOURCE.parent / "slides"                 # read only: the decks' recorded numbers
    COURSE = SOURCE.parent.parent                     # read only: Modules 1 and 2's measured.json
    GENERATED = shutil.ignore_patterns(
        "out", "landing", "mlruns.db", "mlartifacts", "__pycache__", "*.parquet", "handoff",
        "MANIFEST.json", "artefacts", "timing.json", "DATA_PROFILE.md", ".your_attempt")
    WORK = Path(tempfile.mkdtemp(prefix="module3_notebook_")) / "exercises"
    shutil.copytree(SOURCE, WORK, ignore=GENERATED)
    os.chdir(WORK)
    print("working in a temporary copy of exercises/, which holds:",
          ", ".join(sorted(p.name for p in WORK.iterdir() if not p.name.startswith("."))))
    ''')
    explain(
        "Let verbatim code that imports the course's own files find this notebook's "
        "definitions instead.",
        "The lab files import each other (`from service.models import …` inside "
        "`lab_support.py`, `from _narrate import …` inside each solution's demonstration). "
        "This notebook imports none of them: it defines them in its own cells.",
        "`as_module()` wraps names already defined here as a module and registers it, so "
        "that an `import` inside verbatim code returns the notebook's objects.",
        "The code you read in a cell is the code that runs, even when a file imports "
        "another file.")
    nb.code('''
    def as_module(name, names):
        """Register the notebook's own definitions of `names` as the module `name`."""
        module = types.ModuleType(name)
        for attribute in names:
            setattr(module, attribute, globals()[attribute])
        sys.modules[name] = module
        if "." in name:
            parent, _, child = name.rpartition(".")
            package = sys.modules.get(parent) or types.ModuleType(parent)
            package.__path__ = []
            setattr(package, child, module)
            sys.modules[parent] = package
        return module
    ''')


# =============================================================================

def main() -> int:
    front_matter()
    setup()
    from sections import part30, service, block1, block2, block3, part32, closing
    part30.build(nb, explain, lines, figure_lines)
    service.build(nb, explain)
    block1.build(nb, explain)
    block2.build(nb, explain)
    block3.build(nb, explain)
    part32.build(nb, explain)
    closing.build(nb, explain)
    nb.md("""
    ---
    ## Where this notebook and the slides differ, and why

    The slides are the master and are not changed. Where a number computed here does
    not match the number a slide prints, the notebook printed both at that point, with
    the reason. The table gathers every one of them, as they were printed in this run.

    Differences in wording and labels, which the table cannot hold:

    - **3.0, slides 13 and 15**: slide 13 reads "10,000 receipts a day", slide 15
      "N = 10,000 receipts per week". The notebook takes both examples as the slides give
      them: slide 13's stock as a count per 10,000 receipts, whatever the period, and slide
      15's margin for 10,000 receipts per week. The simulation behind them averages about
      1,726 receipts a day; its own layout example, at that volume, is printed beside
      slide 15's.
    - **3.1, slides 20–24** (and the Lab 1 and Lab 4 docstrings): the registry, aliases
      and signatures are credited to Zaharia et al. (2018), a paper that describes
      MLflow's tracking, projects and models components only. The notebook cites it for
      the run record alone.
    - **3.1, slide 34**: the drawn caption says "candidate ≥ in service + δ"; the gate's
      definition, the lab and the check use ">", so a tie is refused.
    - **3.1, slide 41**: the text says "z for 1 − 0.05/C" but prints 2.58 for C = 5, which
      is the two-sided quantile for 1 − 0.05/(2C); the one-sided value is 2.33.
    - **3.1, slide 62**: the derivation ends "p > C_fp/(C_fp + C_fn) = t", while the
      definition card, Lab 3, its check and the figures decide aboard when p ≥ t; at
      equality both answers cost the same.
    - **3.2, slide 12**: "Slide 74 shows this spread" points to a slide number of an
      earlier deck; the spread is drawn on this deck's skew figure (slides 7–8).
    - **3.2, slides 19 and 21**: the two constructed bells of covariate shift are replaced
      by the size of the vehicle's speed on the two days, a stand-in for the model's
      feature: in the generated phone traces the model's own `speed` does not shift.
    - **3.2, slide 21**: the names "label shift" and "concept shift" follow Moreno-Torres et
      al. (2012); Storkey's chapter, cited beside them, calls the first prior probability
      shift and has no name for the second.
    - **3.2, slide 22**: "~1,450 req/s" divides 100 by the mean time of one request;
      served one after another, the 100 requests run at the rate the latency cell prints.
    - Several sources named in the slide text (Moreno-Torres et al., 2012; Savage, 2009;
      Goodman, 1965; Student, 1908) are missing from the decks' reference slides.
    """)
    explain(
        "Gather every number this notebook printed beside a slide's number.",
        "A reader should find every difference between the notebook and the slides in one "
        "place, with its reason, without searching the notebook.",
        "Tabulates what each `beside()` call recorded during this run: the quantity, the value "
        "computed here, the value on the slide or in the archive, and why they differ.",
        "Every row is explained where it first appears; none of them is a change to the slides.")
    nb.code('''
    with pd.option_context("display.max_colwidth", None):     # the reasons, in full
        display(pd.DataFrame(DIFFERENCES, columns=["quantity", "computed here",
                                                   "the slide or the archive", "why they differ"]))
    ''')
    nb.write(OUTPUT)
    print(f"wrote {OUTPUT.relative_to(ROOT)}")
    if "--no-run" not in sys.argv:
        execute(OUTPUT, EXERCISES)
        print(f"executed {OUTPUT.relative_to(ROOT)} in {EXERCISES.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
