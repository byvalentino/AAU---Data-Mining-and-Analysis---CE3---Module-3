"""Part 3.1 opening: the case, the question, and the service's machinery built from
nothing, as setup.sh builds it (Module3.1.pptx slides 1-11)."""

EX = "Module 3/exercises"


def build(nb, explain) -> None:
    nb.md("""
    ---
    # Part 3.1 — Deploying a model as a service, I

    *Deck: `Module3.1.pptx`. A model that other programs can ask, and that can prove which
    version answered.*

    **The hook** (*slide: "A good model with a careless cutoff loses to no model at
    all"*). A bus operator asks a model: is this person aboard? The dumbest rule —
    always answer aboard — costs the operator 1,663 vehicle-hours on the 3,241 test
    rows. The trained model with the 0.5 cutoff every library uses costs 2,054, about
    24 per cent worse than no model. The same model with a cutoff computed from one
    line of arithmetic costs 1,590, and nothing was retrained. These numbers are
    recomputed in Block three below. The module is about everything after the model:
    the service around it, the cutoff, the check at the door, and the proof of which
    version answered.

    **Three beliefs to drop** (*slide: "What you need already, and three beliefs to
    drop"*): using a model means proving which copy answered, not copying it
    somewhere; a 0.5 threshold is not neutral — it assumes a false positive and a false
    negative cost the same; and no error does not mean working.

    **Six words** (*slide: "Six words are used all day"*): a *model* applies a rule
    learnt from examples; a *service* waits for questions from other programs; a
    *request* is one question; a *version* is one saved copy of a model with a name
    that never changes; a *data contract* declares what the service accepts, checked
    before the model is asked; *aboard* means the phone's owner was on the shuttle.

    **Four objectives** (*slide: "Four objectives state their reason before their
    task"*): accept a better model without editing the service (Block one); refuse
    impossible values before the model is asked (Block two); compute the cutoff from
    the two prices (Block three); cause training–serving skew on purpose, then prevent
    it (Block four, in Part 3.2).

    **Every symbol** (*slide: "Every symbol used today"*): p the model's probability
    that the person was aboard; t the cutoff, aboard when p ≥ t; y the true answer, 1
    aboard; C_FP = 1 vehicle-hour, a bus sent for nobody; C_FN = 4 vehicle-hours, a
    passenger left waiting; K the total cost; speed in metres per second; rssi1, rssi2,
    rssiC the signal strengths heard from the beacons, in decibel-milliwatts.
    """)

    # --- the machinery ------------------------------------------------------------------
    nb.md("""
    ## The service, built from nothing

    Before any lab runs, `bash setup.sh` prepares everything the labs read: the phone
    traces, the two trained models, the fifty-line registry and the MLflow store. This
    section does the same in the working copy, with all the code on the page: the
    generator, the training script, the hand-off table and the support file the four
    labs share.

    **The model is a stand-in.** The four modelling sessions between Module 2 and
    Module 3 produce the model this course would rather serve. Until that artefact is
    agreed, the labs need something to promote, refuse, threshold and skew, so
    `service/models.py` trains two small forests from Module 2's generated table. When
    the real artefact arrives it drops into `service/artefacts/` and nothing downstream
    changes — which is release by indirection, the subject of Block one.
    """)
    explain(
        "Let the generator's path constants resolve inside a notebook.",
        "`make_phones.py` finds its calibration file relative to its own location "
        "(`__file__`), which a notebook does not have.",
        "Points `__file__` at the generator in the working copy.",
        "The next cell can be copied from the file unchanged.")
    nb.code('__file__ = str(WORK / "data" / "make_phones.py")')
    explain(
        "Define the phone-trace generator.",
        "The real `passengers.csv` traces sixteen identifiable volunteers and is personal "
        "data, so it is not in the repository. What ships is a generator whose rates — how "
        "often each beacon is heard, how often the target is aboard, the row counts — were "
        "measured from the real file and written into `calibration.json`.",
        "Copies `data/make_phones.py` verbatim: twelve phones on 22 January, fifteen minutes "
        "each at 0.993 seconds, a ride that makes 56.3 per cent of rows aboard, beacons "
        "heard at the archive's measured rates and absent when weak, and a `bus_id` present "
        "exactly when aboard — the leak Module 2 hunts.",
        "The table every lab of 3.1 and 3.2 reads is generated here, deterministically.")
    nb.source(f"{EX}/data/make_phones.py", "HERE", "CALIBRATION", "SEED", "BEACONS", "PROXIMITY",
              "RANGE_METRES", "STRENGTH_AT_ONE_METRE", "PATH_LOSS", "generate")
    explain(
        "Make the generator importable under its file's name.",
        "`service/models.py` and `data/prepare.py` both run `from make_phones import "
        "generate`; without this they would look for a file the notebook never imports.",
        "Registers the five names just defined as the module `make_phones`.",
        "The training script below finds this generator, not a second copy of it.")
    nb.code('make_phones = as_module("make_phones", ["CALIBRATION", "SEED", "BEACONS", "PROXIMITY", "generate"])')
    explain(
        "Run the generator's own demonstration.",
        "It checks the generated table against the archive's measured absence shares.",
        "The lines below are `make_phones.py`'s `__main__` block, verbatim.",
        "The shares land near the archive's, and the absence is encoded twice — an empty "
        "signal strength and a proximity of −1 — on exactly the same rows.")
    nb.step(f"{EX}/data/make_phones.py", 0)
    explain(
        "Write the two days of phone traces where the labs read them.",
        "`setup.sh` runs `data/prepare.py`, whose step 2 writes one Parquet file per day, so "
        "that no lab depends on a generator running at import time.",
        "Generates each day in `calibration.json` and writes `data/phones_<day>.parquet` "
        "(this is prepare.py's step 2, written out; the rest of prepare.py is the "
        "subprocess that trains the models, which the next cells do in place).",
        "10,800 rows for the training day, the number the labs quote.")
    nb.code('''
    for day in CALIBRATION["phones_per_day"]:
        traces = generate(day=day, with_truth=False)
        traces.to_parquet(WORK / "data" / f"phones_{day}.parquet", index=False)
        print(f"generated phones_{day}.parquet: {len(traces):,} rows x {traces.shape[1]} columns")
    agrees("rows of the training day, 22 January", len(pd.read_parquet("data/phones_2020-01-22.parquet")), 10800, 0)
    ''')

    explain(
        "Let the training script's path constants resolve.",
        "`service/models.py` also finds its folders from `__file__`.",
        "Points `__file__` at it in the working copy.",
        "The artefacts and the MLflow store land inside the copy.")
    nb.code('__file__ = str(WORK / "service" / "models.py")')
    explain(
        "Define the table, the stored transform and the pipeline that carries it.",
        "Order is part of the contract: swap two columns and every request still succeeds "
        "and every answer is wrong. So the transform is fitted once, on the training rows, "
        "and stored with the model — the one implementation of the preparation, used by the "
        "request path, the batch path and the registered pipeline alike "
        "[@breck2019; @huyen2022].",
        "Copies the constants, `build_table`, `fit_transform`, `apply_transform`, the "
        "`StoredTransform` step and `as_pipeline` from `service/models.py`, verbatim. The "
        "`stationary` column is deliberately left out: in the generated table it is the "
        "target inverted.",
        "One preparation: median fill, then z-score, in the stored order.")
    nb.source(f"{EX}/service/models.py", "HERE", "EXERCISES", "ARTEFACTS", "DATA", "SEED",
              "FEATURES", "TARGET", "ABOARD", "DAY", "PREPARED_TABLE", "STORE", "ARTEFACT_ROOT",
              "EXPERIMENT", "REGISTERED_MODEL", "CHAMPION", "METRIC", "build_table",
              "fit_transform", "apply_transform", "StoredTransform", "as_pipeline",
              cite="[@breck2019]")
    explain(
        "Define the training of the two models, and the fifty-line registry.",
        "The gate in Lab 1 has to refuse something, and refusing a *worse* model is easy. "
        "Refusing a bigger, newer, slower model that is not better is the decision people get "
        "wrong, so the candidate is built to be exactly that.",
        "Copies `train_and_save` verbatim: split by time at 70 per cent of the rows, fit the "
        "transform on the first part, train v1 (100 trees, depth 6) and v2 (500 trees, no "
        "depth limit), measure accuracy on the later rows, pickle each with its transform, and "
        "write `metrics.json` and `registry.json` — the pointer `approved` and its history.",
        "Two artefacts and a registry that names v1.")
    nb.source(f"{EX}/service/models.py", "train_and_save")
    explain(
        "Define the MLflow store: open it, record the environment and the data, log both "
        "runs, alias the better one.",
        "Months later, \"which model answered, trained on what, with which settings?\" has to "
        "be a lookup rather than an investigation [@schelter2018, § 2; @zaharia2018, § 3].",
        "Copies the store functions verbatim. Each training run records its parameters, the "
        "agreed metric, the environment actually running, the data window and a checksum of "
        "it, and the pipeline with a signature inferred from rows where all four features "
        "are present; the alias `champion` goes on the more accurate version. (Zaharia et al. "
        "describe the run record; the registry, the alias and the signature came to MLflow "
        "after that paper.)",
        "The industrial version of the fifty-line registry, written as it happens.")
    nb.source(f"{EX}/service/models.py", "open_store", "environment_pins", "data_checksum",
              "signature_rows", "log_to_registry", "reset_store", "build", "load_artefact",
              "load_metrics", "load_registered",
              cite="[@zaharia2018, § 3, the run record; @schelter2018, § 2]")
    explain(
        "Make the training script importable under its package name.",
        "`lab_support.py` and `data/prepare.py` import from `service.models`; each import "
        "must return the definitions above.",
        "Registers every name defined from `service/models.py` as the module `service.models`, "
        "inside a package `service`.",
        "One copy of the training code in the whole notebook.")
    nb.code('''
    service_models = as_module("service.models", [
        "HERE", "EXERCISES", "ARTEFACTS", "DATA", "SEED", "FEATURES", "TARGET", "ABOARD", "DAY",
        "PREPARED_TABLE", "STORE", "ARTEFACT_ROOT", "EXPERIMENT", "REGISTERED_MODEL", "CHAMPION",
        "CHALLENGER", "METRIC", "build_table", "fit_transform", "apply_transform", "StoredTransform",
        "as_pipeline", "train_and_save", "open_store", "environment_pins", "data_checksum",
        "signature_rows", "log_to_registry", "reset_store", "build", "load_artefact", "load_metrics",
        "load_registered"])
    ''')
    explain(
        "Train both models and write both registries — `setup.sh`'s training step.",
        "Everything in Blocks one to four reads what this step writes.",
        "Runs the `__main__` block of `service/models.py`, verbatim: `build()` trains, "
        "pickles, writes the registry, starts the MLflow store from nothing, logs both runs "
        "and loads both aliases once as a smoke test.",
        "v1 at 0.8180 and v2 at 0.7291 on the later rows: the candidate is 8.9 points "
        "*less* accurate, and 258 times larger.")
    nb.step(f"{EX}/service/models.py", 0)

    explain(
        "Let the hand-off builder's path constants resolve.",
        "`data/prepare.py` finds the exercises folder from `__file__`, like the two scripts "
        "above.",
        "Points `__file__` at `data/prepare.py` in the working copy.",
        "The hand-off lands in the copy's `data/handoff/`.")
    nb.code('__file__ = str(WORK / "data" / "prepare.py")')
    explain(
        "Define the table Module 2 hands over, in Module 2's schema.",
        "Lab 2's contract is graded against the hand-off's `feature_columns`: a service "
        "whose door accepts different fields from the table upstream promises is documenting "
        "something else.",
        "Copies the hand-off's constants and `build_handoff` from `data/prepare.py`, verbatim: "
        "one row per phone per instant, the stored transform's columns, a mask beside every "
        "filled column, the split point recorded as an instant.",
        "The hand-off is built by the same preparation the service stores.")
    nb.source(f"{EX}/data/prepare.py", "HERE", "EXERCISES", "HANDOFF_SCHEMA", "HANDOFF",
              "build_handoff")
    explain(
        "Build the hand-off and read its manifest.",
        "The manifest is what Lab 2 reads to know which fields the door must accept.",
        "Runs `build_handoff()` and prints the table's shape, the schema version, the feature "
        "columns and the mask columns the manifest records.",
        "The manifest's feature columns are the four the service is trained on.")
    nb.code('''
    handoff = build_handoff()
    manifest = json.loads((HANDOFF / "manifest.json").read_text())
    print(f"data/handoff/: {len(handoff):,} rows x {handoff.shape[1]} columns, schema "
          f"{manifest['schema_version']}, feature columns {manifest['feature_columns']}, "
          f"masks {manifest['mask_columns']}")
    ''')

    nb.md("""
    ### The support file the four labs share

    `exercises/lab_support.py` holds the unsolved marker and the service's parts, so
    that a lab, a check and a solution say the same word for the same thing. Nothing
    here is imported from it: the next cells *are* it.
    """)
    explain(
        "Let `lab_support.py`'s path constant resolve.",
        "It finds the exercises folder from `__file__`.",
        "Points `__file__` at it in the working copy.",
        "The next cell is the file, unchanged.")
    nb.code('__file__ = str(WORK / "lab_support.py")')
    explain(
        "Define the names, the three states of a check, and the loaders every lab calls.",
        "A check says \"not written yet\" (exit 2), \"written, not right\" (exit 1) and "
        "\"this machine is not set up\" (exit 3) as three different things, because a "
        "student told the second while the third is true hunts for a bug that is not there.",
        "Copies `lab_support.py` verbatim. Its loaders import from `service.models`, which "
        "resolves to the definitions above.",
        "`load_artefact`, `load_metrics`, `registry_path`, `open_store`, `load_registered` "
        "and `as_pipeline` are the whole interface the labs see.")
    nb.source(f"{EX}/lab_support.py", "HERE", "REGISTERED_MODEL", "CHAMPION", "CHALLENGER",
              "FEATURES", "HANDOFF_SCHEMA", "HANDOFF_KEYS", "NotSolved", "EnvironmentNotReady",
              "_trained", "load_table", "load_artefact", "load_metrics", "load_handoff",
              "registry_path", "open_store", "load_registered", "as_pipeline",
              "environment_pins")
    explain(
        "Give the solutions the narration helpers they print with.",
        "Each solution's demonstration tells its story through `narrator`, `show_table` "
        "and `save_figure`.",
        "Copies `narrator` and `show_table` verbatim from `exercises/_narrate.py`.",
        "The demonstrations further down print through these. Their lines start with the "
        "seconds since the notebook began, which differ from run to run.")
    nb.source(f"{EX}/_narrate.py", "_START", "_Elapsed", "narrator", "show_table")
    explain(
        "Show the solutions' figures in the notebook instead of writing them to disk.",
        "In the terminal `save_figure` writes `out/lab_0K_<name>.html`, which a notebook "
        "reader would have to open by hand.",
        "Defines `save_figure` with the file's layout, drawing the figure in place — the one "
        "function of `_narrate.py` that is not copied — and registers the three helpers as "
        "the module `_narrate`, which the demonstrations import.",
        "The demonstrations run unchanged, and their figures appear under them.")
    nb.code('''
    def save_figure(fig, name, lab, logger=None, width=1000, height=560):
        """The notebook's save_figure: the layout of exercises/_narrate.py, shown in place
        instead of written to out/lab_0K_<name>.html."""
        fig.update_layout(template="plotly_white", width=width, height=height,
                          margin=dict(l=60, r=30, t=60, b=60))
        show(fig, f"lab_{lab:02d}_{name}", width=width, height=height)


    _narrate = as_module("_narrate", ["narrator", "show_table", "save_figure"])
    ''')
    explain(
        "Read what the two versions measured when they were trained.",
        "The gate, the verdict and the four drivers all start from this file.",
        "Loads `service/artefacts/metrics.json` through `lab_support.load_metrics()`.",
        "The numbers Block one argues about.")
    nb.code('''
    METRICS = load_metrics()
    display(pd.DataFrame(METRICS).T[["kind", "accuracy", "size_bytes", "trained_on_rows", "seed"]])
    agrees("accuracy of v1, the model in service", METRICS["v1"]["accuracy"], 0.818, 3)
    agrees("accuracy of v2, the candidate", METRICS["v2"]["accuracy"], 0.7291, 4)
    agrees("the candidate is less accurate by, points", 100 * (METRICS["v1"]["accuracy"] - METRICS["v2"]["accuracy"]), 8.9, 1)
    ''')

    # --- the case ---------------------------------------------------------------------
    nb.md("""
    ## The case

    *Slides: "Where this sits — five modules, one case, one thread" and "The case — two
    shuttles, sixteen phones, five beacons".* Two automated shuttles ran a fixed loop in
    Copenhagen on 22 and 23 January 2020, reporting position, speed and payload about
    twice a second; sixteen volunteers carried instrumented phones; beacons sat on the
    two vehicles and at three stops; researchers recorded by hand where each person
    was. (The first slide's picture is a frame of the trial's camera video and is not
    reproduced; the medallion-architecture picture and the data-preparation poster that
    follow are not reproduced either — see `figure_map.json`.)
    """)
    explain(
        "Draw the loop from the vehicle's own positions.",
        "The slide shows a screenshot of the route with its three stops. The shipped slice "
        "holds the vehicle's own GPS fixes, which draw the same loop, and its door log, which "
        "marks where it stopped to let people on and off.",
        "Converts latitude and longitude to metres east and north of the loop's corner "
        "(one degree of latitude is 111.32 km; a degree of longitude is that times the cosine "
        "of the latitude), draws every fix by day, and marks the fixes where a door was open. "
        "Then it measures, for each day, how far the fixes reach and how many leave the "
        "southern edge of the box.",
        "Only the 23 January fixes draw the loop. On 22 January all but about 2 per cent of "
        "the fixes stay on a straight stretch about 52 metres long and 6 wide along the "
        "southern edge, and about half of that day's moving speed readings are negative — "
        "the vehicle went back and forth rather than round. The whole route fits in a box of "
        "about 67 by 86 metres; a loop that short is why being near a stop beacon and being "
        "on the vehicle cannot be told apart by radio.")
    nb.figure("route_map", '''
    bus = pd.read_csv("data/bus_slice.csv.gz", low_memory=False)
    latitude0 = bus["lat"].mean()
    east = (bus["lon"] - bus["lon"].min()) * 111320 * math.cos(math.radians(latitude0))
    north = (bus["lat"] - bus["lat"].min()) * 111320
    day = pd.to_datetime(bus["utc_time"], utc=True).dt.date.astype(str)
    fig = go.Figure()
    for which, colour in (("2020-01-22", "#2A78D6"), ("2020-01-23", "#E07B39")):
        here = day == which
        fig.add_scatter(x=east[here], y=north[here], mode="markers", marker=dict(size=3, color=colour),
                        name=f"GPS fixes, {which}")
    doors = bus["door_state"].eq("opened")
    fig.add_scatter(x=east[doors], y=north[doors], mode="markers", name="door open: a stop",
                    marker=dict(size=7, color="#C0392B", symbol="x"))
    fig.update_layout(title=f"The loop, drawn from vehicle {bus['vehicle_id'].iloc[0]}'s own positions",
                      xaxis_title="metres east", yaxis_title="metres north",
                      yaxis=dict(scaleanchor="x", scaleratio=1), legend=dict(orientation="h", y=-0.15))
    show(fig, "route_map", width=760, height=720)
    agrees("north-south extent of the route, metres", north.max(), 67, 0)
    agrees("east-west extent of the route, metres", east.max(), 86, 0)
    for which in ("2020-01-22", "2020-01-23"):
        here, moving = day == which, (day == which) & bus["speed"].ne(0)
        print(f"{which}: 2nd-98th percentile of the fixes {east[here].quantile(0.02):.0f}-{east[here].quantile(0.98):.0f} m east, "
              f"{north[here].quantile(0.02):.0f}-{north[here].quantile(0.98):.0f} m north; "
              f"fixes more than 10 m north {100 * (north[here] > 10).mean():.1f} %; "
              f"moving readings with a negative speed {100 * (bus.loc[moving, 'speed'] < 0).mean():.1f} %")
    beside("rows in the vehicle file", f"{len(bus):,} rows, {bus.shape[1]} columns, "
           f"{bus['vehicle_id'].nunique()} vehicle", "53,155 rows and 22 columns from 2 vehicles",
           "the slide counts the archive's bus file, which is not shipped; the labs ship the slice "
           "of one vehicle, and the route box is the same")
    ''', slides=["3.1 #3", "3.2 #4"], treatment="lab data: the route screenshot replaced by the "
                                                "slice's own GPS fixes and door log")
    explain(
        "Put the module's position on the data it will serve.",
        "The slide says labels cover 100 per cent of the first day and none of the second: "
        "next day no one is writing labels, which is why Module 5 exists.",
        "Reads the archive figure from Module 2's recorded `measured.json`, and measures the "
        "same share on the generated phone tables the labs use.",
        "The generated traces label both days, so the lab data cannot show the gap; the "
        "archive number is the slide's, and it is printed beside the lab data's.")
    nb.code('''
    module2 = json.loads((COURSE / "Module 2" / "slides" / "measured.json").read_text())
    archive = module2["label_coverage_per_day"]["value"]
    for which in ("2020-01-22", "2020-01-23"):
        generated = 100 * pd.read_parquet(f"data/phones_{which}.parquet")["label2"].notna().mean()
        beside(f"rows carrying a label on {which}, per cent", f"{generated:.1f} (generated phones)",
               f"{archive[which]:.1f} (archive, Module 2's measured.json)",
               "make_phones.py labels every row it generates; only the archive lacks the second day's labels")
    ''')
