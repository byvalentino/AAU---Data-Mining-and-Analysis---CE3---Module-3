"""Block one — keeping a record, and releasing a model; Laboratory 1 (Module3.1.pptx
slides 12-43)."""

S = "Module 3/exercises/solutions"
LABS = "Module 3/exercises/labs"
FIGS = "Module 3/slides/make_figs.py"


def solution_paragraphs(nb, explain, relpath: str, notes: list) -> None:
    """One cell per paragraph of a solution's `__main__` block, each under its own
    explanation (goal, why, what the code does, so what)."""
    from notebook_kit import main_paragraphs
    count = len(main_paragraphs(relpath))
    assert len(notes) == count, f"{relpath}: {count} paragraphs, {len(notes)} explanations"
    for number, note in enumerate(notes, start=1):
        explain(*note)
        nb.paragraph(relpath, number)


def build(nb, explain) -> None:
    nb.md("""
    ---
    ## Block one — Keeping a record, and releasing a model

    *Slides: "A model measured once in a notebook cannot answer for itself later",
    "Three words for this block are the record, the artefact and the metric" and "A
    training run writes five facts down while it is happening".*

    A notebook measures a model once, on questions its author already holds. A service
    answers questions it has never seen, sent by another program, months later — and
    must name the model behind any past answer and say what it was trained on. That
    record cannot be reconstructed afterwards, so a training run writes five facts
    while it happens: the settings, the metric the gate will use (agreed before any
    candidate existed), the environment with every version fixed, the window of data
    and a checksum of it, and the artefact — the model and its preparation saved as one
    file [@schelter2018]. `log_to_registry()` above wrote exactly these.
    """)

    # --- slides 16-18: registry, release, rollback -------------------------------------
    nb.md("### The model registry, release and rollback\n\n"
          "*Slides: \"Definition — the model registry\", \"A release is done in five steps, "
          "and the service is not one of them\" and \"A rollback is the release done "
          "backwards, in three steps\".*")
    explain(
        "Define the gate and the rollback as the Lab 1 solution writes them.",
        "A registry is a list of saved versions with one entry naming the version in "
        "service. With each version's metadata and lineage recorded — who trained it, with "
        "which settings, on which data — \"which model, trained on what?\" becomes a query "
        "rather than an investigation [@schelter2018, § 2].",
        "Copies the lab's agreed metric and `promote_if_better` and `rollback` from "
        "`solutions/lab_01.py`, with their docstrings. The docstrings credit these definitions "
        "to Zaharia et al. (2018), as the deck does; that paper describes MLflow's run record, "
        "not this registry (see the closing notes).",
        "Promotion is one write to one entry; rollback is the same write backwards, which "
        "is why it can be trusted at three in the morning.")
    nb.source(f"{S}/lab_01.py", "LAB", "METRIC", "HIGHER_IS_BETTER", "promote_if_better",
              "rollback", cite="[@schelter2018, § 2]")
    explain(
        "Ask the fifty-line registry to release the candidate.",
        "The candidate is bigger, newer, slower — and worse on the agreed metric.",
        "Reads `registry.json`, asks `promote_if_better` about v2 on a copy, and prints the "
        "registry after the refusal.",
        "Refused, with a reason quoting both scores, and the pointer exactly where it was.")
    nb.code('''
    registry_now = json.loads(registry_path().read_text())
    print("registry.json:", registry_now)
    trial = json.loads(json.dumps(registry_now))
    print("promote v2?", promote_if_better(trial, "v2", METRICS))
    print("after the refusal:", trial, "- unchanged:", trial == registry_now)
    ''')

    nb.md("### Registries in practice, and release by indirection\n\n"
          "*Slides: \"Commercial model registries in practice\" and \"Definition — release by "
          "indirection\".* The first slide shows vendors' logos and screenshots (MLflow, "
          "SageMaker, Azure ML, Vertex AI, Databricks Unity Catalog, Weights & Biases); they "
          "are not reproduced. In the MLOps architecture derived by {@kreuzberger2023}, the "
          "model registry is component C6, which stores the trained models with their "
          "metadata; MLflow, SageMaker and Azure ML are among their examples.")
    explain(
        "Write the definition of release by indirection.",
        "The service loads whichever artefact the entry marks approved, so a release never "
        "touches the service's code. MLOps tools call the pattern model aliasing — MLflow's "
        "`@champion`; general software calls it the pointer or symlink pattern.",
        "Writes the three statements of the definition card in sympy — a promotion or a "
        "rollback is written as the value of the entry after the move, approved′ — then "
        "checks them on the registry: promote a hypothetical better candidate, then roll back.",
        "The entry moves; the service's code does not. Indirection must not be relied on "
        "when the new model expects different input columns — that is what the signature is "
        "for.")
    nb.equation("indirection", '''
    entry_name, entry = sp.Symbol(r"\\mathrm{name}"), sp.Symbol(r"\\mathrm{approved}")
    entry_after = sp.Symbol(r"\\mathrm{approved}'")          # the entry after the move
    entry_history = sp.IndexedBase(r"\\mathrm{history}")
    formula(sp.Eq(sp.Function(r"\\mathrm{serve}")(entry_name), sp.Function(r"\\mathrm{artefact}")(entry)))
    formula(sp.Implies(sp.Symbol(r"\\mathrm{promote}"), sp.Eq(entry_after, sp.Symbol(r"\\mathrm{candidate}"))),
            sp.Implies(sp.Symbol(r"\\mathrm{rollback}"), sp.Eq(entry_after, entry_history[-1])))
    better = {version: dict(facts) for version, facts in METRICS.items()}
    better["v2"]["accuracy"] = round(METRICS["v1"]["accuracy"] + 0.05, 4)
    trial = json.loads(json.dumps(registry_now))
    print("promote a hypothetical v2 five points better:", promote_if_better(trial, "v2", better)[0], trial)
    print("rollback ->", rollback(trial), trial)
    ''', slides=["3.1 #20"])

    # --- slides 21-24: MLflow, signature, alias ------------------------------------------
    nb.md("### The same record in MLflow\n\n"
          "*Slides: \"MLflow: record, signature and registry\", \"Definition — model "
          "signature\", \"The signature checks a request in four steps before the model sees "
          "it\" and \"Definition — release by alias\".* MLflow is free software that writes "
          "one record per training run [@zaharia2018, § 3]. The slide shows a screenshot of the "
          "MLflow demonstration server, which is not reproduced; the next cell reads the "
          "module's own store instead.")
    explain(
        "Read back what the store recorded for the two training runs.",
        "The record is only worth something if it can be queried months later.",
        "Opens the working copy's MLflow store and lists, for each run, the settings, the "
        "data window and checksum, the metrics, and the registered version with its aliases.",
        "One registered model, `aboard`, two versions, and the alias `champion` on version 1. "
        "(Zaharia et al. describe tracking, projects and models; the registry and its "
        "aliases were added to MLflow later, and are documented with the tool rather than in "
        "that paper.)")
    nb.code('''
    client = open_store()
    experiment = client.get_experiment_by_name(EXPERIMENT)
    versions = {int(v.version): v for v in client.search_model_versions(f"name='{REGISTERED_MODEL}'")}
    record = []
    for run in sorted(client.search_runs([experiment.experiment_id]), key=lambda r: r.info.run_name):
        version = next(v for v in versions.values() if v.run_id == run.info.run_id)
        record.append({"run": run.info.run_name, "registered version": int(version.version),
                       "aliases": ", ".join(sorted(version.aliases)),
                       "n_estimators": run.data.params["n_estimators"], "max_depth": run.data.params["max_depth"],
                       "data checksum": run.data.params["data_checksum"][:16] + "…",
                       "accuracy": run.data.metrics["accuracy"], "size_bytes": int(run.data.metrics["size_bytes"])})
    display(pd.DataFrame(record).set_index("run"))
    ''')
    explain(
        "Write the signature the store records, and read the champion's.",
        "A model knows its columns by position. The signature is the list of names, types "
        "and order recorded with it, enforced every time it is asked [@breck2019].",
        "Writes the three statements of the definition card in sympy, entry by entry for "
        "i = 1 … k — acceptance as a product of indicators, one per entry — then loads `models:/aboard@champion` and prints the input names the "
        "platform will enforce.",
        "speed, rssi1, rssi2, rssiC, in that order. A signature must not be trusted to catch "
        "a column whose name is right and unit wrong.")
    nb.equation("signature", '''
    i_, k_ = sp.Symbol("i", integer=True), sp.Symbol("k", integer=True, positive=True)
    S_, names_, types_ = sp.IndexedBase("S"), sp.IndexedBase(r"\\mathrm{name}"), sp.IndexedBase(r"\\mathrm{type}")
    x_ = sp.Symbol("x")
    column_of = lambda j: sp.Symbol(rf"x[\\mathrm{{name}}_{{{j}}}]")
    for_each_entry = sp.And(sp.Le(1, i_), sp.Le(i_, k_))
    indicator = sp.Function(r"\\mathbf{1}")
    formula(sp.Eq(S_[i_], sp.Tuple(names_[i_], types_[i_]), evaluate=False), for_each_entry)
    # "for every i" as a product of indicators, since sympy has no quantifier
    formula(sp.Eq(indicator(sp.Function(r"\\mathrm{accept}")(x_)),
                  sp.Product(indicator(sp.And(sp.Symbol(r"\\mathrm{name}_i \\in \\mathrm{columns}(x)"),
                                              sp.Eq(sp.Function(r"\\mathrm{type}")(column_of("i")), types_[i_]))),
                             (i_, 1, k_))))
    formula(sp.Eq(sp.Indexed(sp.IndexedBase(r"\\mathrm{input}"), i_), column_of("i")), for_each_entry)
    champion_model = load_registered(CHAMPION)
    enforced = champion_model.metadata.get_input_schema().input_names()
    print("the champion's signature:", enforced)
    assert enforced == FEATURES
    ''', slides=["3.1 #22"])
    explain(
        "Define the logging of a training run and the alias move, as the Lab 1 solution "
        "writes them.",
        "Part (b) of Lab 1: the same two models in a real registry, with their signature, "
        "and release as the move of one alias.",
        "Copies `log_training_run` and `promote_by_alias` from `solutions/lab_01.py`. Their "
        "docstrings credit the signature to Zaharia et al. (2018), as the deck does; that "
        "paper describes the run record only (see the closing notes).",
        "Logging a run registers a new version; promoting moves one alias.")
    nb.source(f"{S}/lab_01.py", "log_training_run", "promote_by_alias",
              cite="[@zaharia2018, § 3, the run record only — the signature, registry and alias "
                   "are the deck's attribution, see the closing notes]")
    explain(
        "Write the definition of release by alias, and read where the two aliases point.",
        "An alias is the fifty-line registry's `approved` entry, kept by the platform: the "
        "service asks for a name and an alias, and the registry answers with a version.",
        "Writes the three statements of the definition card in sympy — the alias as a "
        "function from (name, alias) to a version, and a move as its value afterwards, "
        "alias′ — then asks the store where `champion` and `challenger` point.",
        "The service resolves `models:/aboard@champion` and never learns a version number.")
    nb.equation("alias", '''
    alias_of, alias_after = sp.Function(r"\\mathrm{alias}"), sp.Function(r"\\mathrm{alias}'")
    champion_ = (sp.Symbol(r"\\mathrm{name}"), sp.Symbol(r"\\mathrm{champion}"))
    formula(sp.Eq(alias_of(*champion_), sp.Symbol(r"\\mathrm{version}")))
    formula(sp.Implies(sp.Symbol(r"\\mathrm{promote}"), sp.Eq(alias_after(*champion_), sp.Symbol(r"\\mathrm{version}_{\\mathrm{new}}"))),
            sp.Implies(sp.Symbol(r"\\mathrm{rollback}"), sp.Eq(alias_after(*champion_), sp.Symbol(r"\\mathrm{version}_{\\mathrm{previous}}"))))
    print("champion ->", client.get_model_version_by_alias(REGISTERED_MODEL, CHAMPION).version,
          "  challenger ->", client.get_model_version_by_alias(REGISTERED_MODEL, CHALLENGER).version)
    ''', slides=["3.1 #24"])

    # --- slides 25-28: the metric and its drivers -----------------------------------------
    nb.md("""
    ### The promotion metric, and what feeds it

    *Slides: "A metric is measured against labels, and the labels are measurements too",
    "Definition — the promotion metric", "What feeds the metric — four drivers, measured
    on this module's two models" and "Weights — how much each driver counts".*

    A label is itself a measurement, written by a person, and its error rate is not
    nought — so the gate needs a margin. The promotion metric is one number, agreed
    before any candidate exists; here it is accuracy on later rows neither model trained
    on. One number is a deliberate simplification: it is fed by several drivers, and a
    driver weighted zero is lost. This module gives every driver except accuracy a
    weight of zero, and says so.
    """)
    explain(
        "Bring in the deck's figure code, so the slides' figures are drawn by the slides' "
        "own functions.",
        "Five figures of 3.1 and 3.2 come from `slides/make_figs.py`, and it also measures "
        "the work each forest does per request.",
        "Copies the palette, the prices, the departure constants and the measuring functions "
        "from `make_figs.py`, verbatim.",
        "From here on a figure labelled *the slide's own code* is the deck's figure, "
        "recomputed from the models trained above.")
    nb.source(FIGS, "BLUE", "COST_FALSE_POSITIVE", "SEATS", "CONTRACT_SPEED_MARGIN",
              "scored_test_set", "measure_work_per_request", "total_cost",
              "cost_of_always_aboard", "cost_of_always_not_aboard", "cheapest_threshold",
              "jensen_departure", "load_artefact_model")
    explain(
        "Show the deck's figures in place, and score the test period once for every later cell.",
        "`make_figs.py`'s `write()` saves `figures/<name>.png` for the deck; a notebook should "
        "show the picture under the code that drew it. And Blocks one to four all ask the same "
        "question of the same rows: what did the model in service say on the test period?",
        "Defines `write()` with the script's layout, drawing the figure in place — the one "
        "function of `make_figs.py` that is not copied — then runs `scored_test_set()`: the "
        "approved artefact, its prepared test rows, its probabilities and the truth.",
        "3,241 test rows, split by time, about half of them aboard.")
    nb.code('''
    def write(figure, name, height=620, width=1100):
        """make_figs.py's write(), shown in place instead of written to figures/."""
        figure.update_layout(template="plotly_white", font=dict(size=15),
                             title_font=dict(size=19), margin=dict(t=70, l=80, r=40, b=70))
        display(Image(figure.to_image(format="png", width=width, height=height, scale=1)))


    ARTEFACT_V1, X_TEST, P_TEST, Y_TEST = scored_test_set()
    print(f"test period: {len(Y_TEST):,} rows, {int(Y_TEST.sum()):,} aboard, "
          f"{int((Y_TEST == 0).sum()):,} not aboard")
    ''')
    explain(
        "Define the figure of the four drivers, as the deck's script writes it.",
        "Quality, operating cost, footprint and robustness. The slide draws the two drivers "
        "on which the two models differ most.",
        "Copies `figure_gate()` from `make_figs.py`, verbatim.",
        "The next cell draws it on this notebook's two models.")
    nb.source(FIGS, "figure_gate")
    explain(
        "Measure the four drivers on the two models and draw the two that differ most.",
        "Operating cost is counted, not timed: a forest answers by walking each tree from "
        "root to leaf, and `decision_path` counts the nodes visited, which is the same on "
        "every machine — a stopwatch is not.",
        "Runs `figure_gate()` on `metrics.json`, then counts decision nodes per request over "
        "the test period with `measure_work_per_request()`, and checks every number the "
        "slide prints.",
        "The candidate does 17.7 times the work and is 258 times larger on disk, for 8.9 "
        "points less accuracy. Only the first driver reaches the gate.")
    nb.figure("gate", '''
    figure_gate(METRICS)
    WORK_PER_REQUEST = {v: measure_work_per_request(load_artefact_model(v), X_TEST) for v in ("v1", "v2")}
    agrees("accuracy on later rows, model in service", METRICS["v1"]["accuracy"], 0.818, 3)
    agrees("accuracy on later rows, candidate", METRICS["v2"]["accuracy"], 0.7291, 4)
    agrees("work per request, model in service (decision nodes)", WORK_PER_REQUEST["v1"]["decision_nodes_per_request"], 575.4, 1)
    agrees("work per request, candidate (decision nodes)", WORK_PER_REQUEST["v2"]["decision_nodes_per_request"], 10165.1, 1)
    WORK_MULTIPLE = round(WORK_PER_REQUEST["v2"]["decision_nodes_per_request"] / WORK_PER_REQUEST["v1"]["decision_nodes_per_request"], 1)
    agrees("the candidate's work, times the model in service's", WORK_MULTIPLE, 17.7, 1)
    agrees("size on disk, candidate, MB", METRICS["v2"]["size_bytes"] / 1e6, 94.92, 2)
    agrees("size on disk, model in service, MB", METRICS["v1"]["size_bytes"] / 1e6, 0.37, 2)
    agrees("how many times larger", METRICS["v2"]["size_bytes"] / METRICS["v1"]["size_bytes"], 258, 0)
    ''', slides=["3.1 #27"], treatment="exact: the slide's own code")

    # --- slides 31-33: value and cost ---------------------------------------------------------
    nb.md("### Value and cost, both in euro\n\n"
          "*Slides: \"Definition — value and cost, both in euro\", \"Figure — the Pareto front "
          "in cost and value\" and \"Boiled down — the same gate, as five MLflow calls\".* "
          "(The weighted cost function and the Pareto front in accuracy and work are on hidden "
          "slides and are not reproduced.)")
    explain(
        "Write value, cost and net benefit, and work the slide's example for the model in "
        "service.",
        "With both sides in euro no weights are needed: the best option is the one with the "
        "largest net benefit V − C [@provost2013, ch. 7].",
        "Builds the three definitions in sympy and evaluates them with the slide's "
        "illustrative prices — a million requests a year, 1 € gross value per request, 2 € "
        "per mistake, 10⁻⁵ € per unit of work, 100 € per development hour, 200 hours — and "
        "the model's measured accuracy (0.818 at the cutoff 0.5) and work (575.4 nodes).",
        "V = 6.36 × 10⁵ €, C = 2.58 × 10⁴ €, net 6.10 × 10⁵ €, as on the slide.")
    nb.equation("value_and_cost", '''
    R, g, e_m, e_w, e_h, h, acc, wk = sp.symbols(r"R g e_{m} e_{w} e_{h} h a w", positive=True)
    V_expr = R * g - R * (1 - acc) * e_m                  # the latency and size penalties are nought here
    C_expr = h * e_h + R * wk * e_w
    V_, C_ = sp.symbols("V C")
    penalty = lambda what: sp.Symbol(rf"P_{{\\mathrm{{{what}}}}}")
    formula(sp.Eq(C_, sp.Add(sp.Symbol(r"C_{\\mathrm{development}}"), sp.Symbol(r"C_{\\mathrm{operation}}"), evaluate=False)),
            sp.Eq(V_, sp.Add(sp.Symbol(r"V_{\\mathrm{gross}}"), -penalty("mistakes"), -penalty("latency"), -penalty("size"),
                             evaluate=False)),
            sp.Eq(sp.Symbol(r"\\mathrm{Net}"), sp.Add(V_, -C_, evaluate=False)))
    formula(sp.Eq(sp.Symbol(r"P_{\\mathrm{mistakes}}"), R * (1 - acc) * e_m, evaluate=False),
            sp.Eq(sp.Symbol(r"C_{\\mathrm{operation}}"), R * wk * e_w, evaluate=False))
    PRICES = {R: 10**6, g: 1, e_m: 2, e_w: sp.Rational(1, 10**5), e_h: 100, h: 200}
    V_service = float(V_expr.subs(PRICES | {acc: sp.Rational(818, 1000)}))
    C_service = float(C_expr.subs(PRICES | {wk: sp.Float(WORK_PER_REQUEST["v1"]["decision_nodes_per_request"])}))
    agrees("V, model in service, 10^5 euro", V_service / 1e5, 6.36, 2)
    agrees("C, model in service, 10^4 euro", C_service / 1e4, 2.58, 2)
    agrees("net benefit, model in service, 10^5 euro", (V_service - C_service) / 1e5, 6.10, 2)
    ''', slides=["3.1 #31"])
    explain(
        "Redraw the slide's Pareto front in cost and value, and place the two measured "
        "models on it.",
        "An option is dominated if another is both cheaper and more valuable; the options "
        "nobody dominates are the only ones worth arguing about, and in euro the best of "
        "them is the one with the largest V − C.",
        "Draws the slide's seven points and its two lines from the chart's own values — five "
        "of the seven options are illustrative — then computes the model in service's point "
        "and the candidate's from their measured accuracy and work at the slide's prices, and "
        "the line of equal net benefit through tuned boosting, V − C = 6.25 × 10⁵ €.",
        "The model in service lands on the slide's point. The candidate does not: its "
        "position on the slide needs penalties and development hours the deck does not "
        "state. The slide's dashed line is also a little steeper than a line of equal net "
        "benefit.")
    nb.figure("pareto_cost_value", '''
    names = ["Rule-based baseline", "Logistic regression", "Model in service", "Tuned boosting: best",
             "Small neural net", "Ensemble", "Candidate (large net)"]
    cost_1e4 = [0.45, 0.92, 2.58, 5.5, 11.5, 15, 16.2]           # the slide's chart values
    value_1e5 = [4, 5.6, 6.36, 6.8, 6, 7.05, 4.28]
    front = [0, 1, 2, 3, 5]
    fig = go.Figure()
    fig.add_scatter(x=[cost_1e4[i] for i in front], y=[value_1e5[i] for i in front], mode="lines",
                    line=dict(color="#76818B", width=2), name="Pareto front (slide)")
    fig.add_scatter(x=[0, 16.5], y=[6.25, 8.05], mode="lines", line=dict(color="#5CAF8D", dash="dash"),
                    name="'equal net benefit' as the slide draws it")
    fig.add_scatter(x=[0, 16.5], y=[6.25, 6.25 + 1.65], mode="lines", line=dict(color="#2E8B57", dash="dot"),
                    name="V − C = 6.25 × 10⁵ € exactly")
    fig.add_scatter(x=cost_1e4, y=value_1e5, mode="markers+text", text=names,
                    textposition=["middle right", "middle right"] + ["top center"] * 5,
                    marker=dict(size=11, color=["#76818B"] * 3 + ["#5CAF8D"] + ["#76818B"] * 2 + ["#DF8E2E"]),
                    name="options (slide; five illustrative)")
    C_candidate = float(C_expr.subs(PRICES | {wk: sp.Float(WORK_PER_REQUEST["v2"]["decision_nodes_per_request"])}))
    V_candidate = float(V_expr.subs(PRICES | {acc: sp.Rational(7291, 10000)}))
    fig.add_scatter(x=[C_service / 1e4, C_candidate / 1e4], y=[V_service / 1e5, V_candidate / 1e5],
                    mode="markers", marker=dict(size=16, symbol="x", color="#C0392B"),
                    name="measured here, at the slide's prices")
    fig.update_layout(title="The Pareto front in cost and value (illustrative, with the two measured models)",
                      xaxis_title="cost C, × 10⁴ € per year — lower is better",
                      yaxis_title="value V, × 10⁵ € per year — higher is better",
                      xaxis_range=[0, 18], yaxis_range=[3, 8.5], legend=dict(orientation="h", y=-0.2))
    show(fig, "pareto_cost_value", height=600)
    agrees("tuned boosting's net benefit, 10^5 euro (6.80 - 0.55)", 6.80 - 0.55, 6.25, 2)
    agrees("its lead over the model in service, 10^4 euro", (6.25e5 - 6.10e5) / 1e4, 1.5, 1)
    beside("candidate, V and C at the slide's prices (10^5 and 10^4 euro)",
           f"{V_candidate / 1e5:.2f} and {C_candidate / 1e4:.2f}", "4.28 and 16.2",
           "the slide adds latency and size penalties, and more development hours, that it does not state")
    beside("slope of the slide's dashed line", f"{(8.05 - 6.25) / 16.5:.3f} (10^5 € per 10^4 €)", "0.100 for equal V − C",
           "the dashed line ends at 8.05 where equal net benefit gives 7.90; it misses tuned boosting by 0.05 × 10^5 €")
    ''', slides=["3.1 #32"], treatment="illustrative: the slide's chart values redrawn exactly, "
                                       "with the two measured models placed on it")
    explain(
        "Compute the promotion cost J = 1 − accuracy the slide logs to MLflow.",
        "With every weight but accuracy at nought, the cost function is one minus accuracy, "
        "and the gate compares it for the two versions.",
        "Evaluates J for both.",
        "0.2709 against 0.182: higher cost, so the alias stays on v1.")
    nb.code('''
    agrees("J for the candidate", 1 - METRICS["v2"]["accuracy"], 0.2709, 4)
    agrees("J for the champion", 1 - METRICS["v1"]["accuracy"], 0.182, 3)
    ''')

    # --- slides 34-35: gate and verdict -----------------------------------------------------
    nb.md("### The gate and the promotion verdict\n\n"
          "*Slides: \"Definition — the gate\" and \"Definition — the promotion verdict\".*")
    explain(
        "Write the gate, and draw the comparison the slide draws.",
        "A candidate is promoted only if it beats the model in service by more than the "
        "margin δ, which is nought here, so an equal score is refused.",
        "Writes the rule in sympy and evaluates it on the two measured accuracies, and on a "
        "tie.",
        "0.0889 short: refused; and a tie is refused too.")
    nb.equation("gate", '''
    m_c, m_a, delta = (sp.Symbol(r"m(\\mathrm{candidate})"), sp.Symbol(r"m(\\mathrm{approved})"), sp.Symbol(r"\\delta"))
    gate_rule = sp.StrictGreaterThan(m_c, sp.Add(m_a, delta, evaluate=False))
    formula(sp.Equivalent(sp.Symbol(r"\\mathrm{promote}(\\mathrm{candidate})"), gate_rule))
    decided = gate_rule.subs({m_c: METRICS["v2"]["accuracy"], m_a: METRICS["v1"]["accuracy"], delta: 0})
    print("promote the candidate?", bool(decided))
    print("a tie promoted?", bool(gate_rule.subs({m_c: 0.818, m_a: 0.818, delta: 0})))
    ''', slides=["3.1 #34"])
    explain(
        "Draw the comparison the slide draws, from `metrics.json`.",
        "The slide shows the two scores as bars on one accuracy axis, with the model in "
        "service as the line to beat.",
        "Draws the two accuracies as horizontal bars with the line at the model in service, "
        "and records the one difference from the slide.",
        "The slide's drawn caption reads \"≥\", while its definition, the lab and the check "
        "use \">\" — at δ = 0 only \">\" refuses a tie.")
    nb.figure("gate_bar", '''
    shortfall = METRICS["v1"]["accuracy"] - METRICS["v2"]["accuracy"]
    fig = go.Figure(go.Bar(y=["Candidate", "In service"], x=[METRICS["v2"]["accuracy"], METRICS["v1"]["accuracy"]],
                           orientation="h", marker_color=["#E07B39", "#2A78D6"],
                           text=[f"Candidate · {METRICS['v2']['accuracy']:.4f}", f"In service · {METRICS['v1']['accuracy']:.3f}"],
                           textposition="inside"))
    fig.add_vline(x=METRICS["v1"]["accuracy"], line=dict(color="#52514E", dash="dash"),
                  annotation_text=f"{shortfall:.4f} short → refused", annotation_position="bottom right")
    fig.update_layout(title="Promote only if candidate > in service + δ. Here δ = 0.",
                      xaxis_title="accuracy on later rows", xaxis_range=[0, 1], showlegend=False)
    show(fig, "gate_bar", height=300)
    agrees("the candidate's shortfall", shortfall, 0.0889, 4)
    beside("the rule the slide's drawn caption states", "candidate > in service + δ (definition, lab, check)",
           "candidate ≥ in service + δ", "a slip in the drawing; with δ = 0 the '≥' would promote a tie, "
                                          "which the definition card and Lab 1 refuse")
    ''', slides=["3.1 #34"], treatment="lab data: the slide's drawn bar, from metrics.json")
    explain(
        "Write the promotion verdict.",
        "The gate compares one number. The release decision is made by a person who sees "
        "everything the gate refuses to look at, in the unit the operator pays in.",
        "Writes the three rules of the definition card in sympy, with hold as neither of the "
        "other two.",
        "Three calls, each a condition on measured numbers.")
    nb.equation("verdict", '''
    m_ = sp.Function("m")
    K_ = sp.Function("K")
    delta_back = sp.Symbol(r"\\delta_r")
    roll_back = sp.Lt(m_(sp.Symbol(r"\\mathrm{now}")),
                      sp.Add(m_(sp.Symbol(r"\\mathrm{at\\ promotion}")), -delta_back, evaluate=False))
    promote_call = sp.And(sp.Lt(K_(sp.Symbol(r"\\mathrm{candidate}")), K_(sp.Symbol(r"\\mathrm{approved}"))),
                          sp.Lt(K_(sp.Symbol(r"\\mathrm{candidate}")), K_(sp.Symbol(r"\\mathrm{trivial}"))),
                          sp.Le(sp.Symbol("p_{95}"), sp.Symbol(r"\\mathrm{budget}")))
    formula(sp.Equivalent(sp.Symbol(r"\\mathrm{roll\\ back}"), roll_back))
    formula(sp.Equivalent(sp.Symbol(r"\\mathrm{promote}"), promote_call))
    formula(sp.Equivalent(sp.Symbol(r"\\mathrm{hold}"),
                          sp.And(sp.Not(sp.Symbol(r"\\mathrm{promote}")), sp.Not(sp.Symbol(r"\\mathrm{roll\\ back}")))))
    ''', slides=["3.1 #35"])
    explain(
        "Define the verdict as the Lab 1 solution writes it.",
        "The verdict is where cost, latency and size come back in, with a reason a person "
        "can check [@provost2013, ch. 7–8; @sculley2015].",
        "Copies the verdict's constants and `promotion_verdict` from `solutions/lab_01.py`, "
        "verbatim.",
        "Promote, hold or roll back — with a reason built out of measured numbers. A more "
        "accurate candidate that is dearer at the priced cutoff is held.")
    nb.source(f"{S}/lab_01.py", "VERDICT_CALLS", "EVIDENCE_KEYS", "REGRESSION_MARGIN",
              "LATENCY_PERCENTILE", "promotion_verdict",
              cite="[@provost2013, ch. 7–8; @sculley2015]")

    # --- slides 36-41: the margin ------------------------------------------------------------
    nb.md("### The gate margin\n\n"
          "*Slides: \"Definition — the gate margin\", \"Every symbol in the margin formula, and "
          "what makes it move\", \"The multiplier z is a confidence choice, not a magic number\", "
          "\"The number of labels a margin needs is found in three steps\" and \"The formula "
          "assumes independent labels; tumbling windows are not\".*")
    explain(
        "Write the smallest margin measurement error can justify, and prove it is the "
        "Wilson score interval's half-width.",
        "If three labels in a hundred are wrong, a lead of one in a hundred is noise. The "
        "margin must be at least the size of the measurement error [@wilson1927; @brown2001].",
        "Builds the slide's formula in k and n, builds the Wilson half-width in the observed "
        "share p̂ = k/n, and simplifies the difference of their squares; then evaluates the "
        "margin for the champion's own count on the 3,241 test rows.",
        "The two are the same function. On these rows a lead below about 1.3 points is "
        "noise — the candidate's 8.9-point deficit is not.")
    nb.equation("gate_margin", '''
    k, n, z = sp.symbols("k n z", positive=True)
    p_hat = sp.Symbol(r"\\hat{p}", positive=True)
    margin_slide = z / (n + z**2) * sp.sqrt(k * (n - k) / n + z**2 / 4)
    wilson_half = z * sp.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    formula(sp.GreaterThan(sp.Symbol(r"\\delta"), margin_slide), sp.Eq(z, sp.Float("1.96", 3)))
    print("slide margin² - Wilson half-width² at p̂ = k/n, simplified:",
          sp.simplify(margin_slide**2 - wilson_half.subs(p_hat, k / n)**2))
    right = int(((P_TEST >= 0.5).astype(int) == Y_TEST).sum())
    margin_here = float(margin_slide.subs({k: right, n: len(Y_TEST), z: stats.norm.ppf(0.975)}))
    print(f"the champion: k = {right:,} right of n = {len(Y_TEST):,}; smallest defensible margin δ = {margin_here:.4f}")
    ''', slides=["3.1 #36"])
    explain(
        "Read z off the normal distribution rather than a table.",
        "z counts how many standard errors the margin covers; 1.96 covers 95 in 100 with the "
        "error split between two tails, 1.645 puts it all in one — the right choice for a gate "
        "that only asks whether the candidate is better.",
        "Computes the two-sided quantiles the slide tabulates, and the one-sided one.",
        "The table is four quantiles of one curve. Fix z before the scores are seen.")
    nb.code('''
    for level, stated in ((0.90, 1.645), (0.95, 1.960), (0.99, 2.576), (0.999, 3.291)):
        agrees(f"z, two-sided, {100 * level:g} per cent", stats.norm.ppf(1 - (1 - level) / 2), stated, 3)
    agrees("z, one-sided, 95 per cent", stats.norm.ppf(0.95), 1.645, 3)
    ''')
    explain(
        "Invert the margin for the number of labels it needs.",
        "A small margin is expensive: the labels grow with one over the margin squared.",
        "Solves δ = z √(p(1 − p)/n) for n in sympy and evaluates the slide's example — one "
        "point of margin at an expected accuracy of 0.9 — then converts it into hours of "
        "five-minute windows.",
        "About 3,460 labels, 288 hours, 12 days — if the windows are independent. One fault "
        "that lasts thirty minutes fills six windows with the same label, so with correlated "
        "labels 12 days is the floor, not the answer.")
    nb.equation("labels_for_margin", '''
    p_, d_, n_ = sp.symbols("p delta n", positive=True)
    labels_needed_expr = sp.solve(sp.Eq(d_, z * sp.sqrt(p_ * (1 - p_) / n_)), n_)[0]
    formula(sp.Eq(sp.Symbol(r"n"), labels_needed_expr, evaluate=False))
    labels = float(labels_needed_expr.subs({z: 1.96, p_: 0.9, d_: 0.01}))
    agrees("labels for one point at 0.9, thousands", labels / 1000, 3.46, 2)
    agrees("minutes of five-minute windows, 10^4", 3.46e3 * 5 / 1e4, 1.73, 2)
    agrees("hours", 3.46e3 * 5 / 60, 288, 0)
    agrees("days", 3.46e3 * 5 / 60 / 24, 12, 0)
    agrees("halving the margin multiplies the labels by", float(labels_needed_expr.subs({z: 1.96, p_: 0.9, d_: 0.005})) / labels, 4, 0)
    ''', slides=["3.1 #39", "3.1 #40"])
    nb.md("*Slide: \"Many classes or a regression target: same gate, different noise "
          "estimate\".* With C classes and a gate on one class rate, keep Wilson and raise z "
          "(Bonferroni [@goodman1965]). With a regression target, compare per-row "
          "differences of absolute error with a paired t multiplier [@student1908], or "
          "bootstrap the mean difference for RMSE [@efron1979]. δ is then in the units of "
          "the target, and it is not Glass's delta, the effect size of Module 4 [@glass1976].")
    explain(
        "Check the slide's two regression examples and its Bonferroni z.",
        "The δ² rule holds for any gate: halve the margin, quadruple the labels.",
        "Evaluates n ≈ z² s_d² / δ² as the slide writes it, with z² = 3.84, and exactly; "
        "then the z values for five classes.",
        "61.4, so 62 rows, and 246 at half the margin. The slide's z for five classes is "
        "2.58, which is the quantile at 1 − 0.05/(2C), the two-sided Bonferroni value; the "
        "slide's words say 1 − 0.05/C, whose quantile is 2.33.")
    nb.equation("regression_gate", '''
    s_d, delta_r = sp.symbols("s_d delta", positive=True)
    rows_needed = z**2 * s_d**2 / delta_r**2
    formula(sp.Eq(sp.Symbol("n"), rows_needed, evaluate=False))
    agrees("rows, s_d = 2.0, delta = 0.5, z² = 3.84", float(rows_needed.subs({z: sp.sqrt(sp.Float(3.84)), s_d: 2, delta_r: 0.5})), 61.4, 1)
    agrees("rows, rounded up", math.ceil(float(rows_needed.subs({z: 1.96, s_d: 2, delta_r: 0.5}))), 62, 0)
    agrees("rows at delta = 0.25", float(rows_needed.subs({z: 1.96, s_d: 2, delta_r: 0.25})), 246, 0)
    beside("z for five classes", f"{stats.norm.ppf(1 - 0.05 / 5):.3f} at 1 − 0.05/C; "
           f"{stats.norm.ppf(1 - 0.05 / 10):.3f} at 1 − 0.05/(2C)", "2.58 'for 1 − 0.05/C'",
           "the slide's number is the two-sided Bonferroni quantile; its formula in words is the one-sided one")
    ''', slides=["3.1 #41"])
    nb.md("*Slide: \"The gate decides in four steps, and writes its reason down\".* Take rows "
          "from a later period than either model trained on; compute the agreed metric for "
          "both; promote only if the lead exceeds the margin; whatever the outcome, write both "
          "scores and the decision into the record.")

    # --- the laboratory ----------------------------------------------------------------------
    nb.md("""
    ## Laboratory 1 — The gate, and the record behind it

    *Slide: "Lab 1 — The gate, and the record behind it".* Part (a): the fifty-line
    registry. Part (b): the same two models in MLflow, and the champion alias. Part (c):
    the verdict, written last, once Labs 3 and 4 have measured the costs and the
    latency. Twenty-five minutes, and ten more for part (c).
    """)
    nb.statement(f"{LABS}/01_the_gate.py")
    nb.md("""
    ### The solution, step by step

    The five functions were defined above, where the deck introduces each concept:
    `promote_if_better` and `rollback` with the registry, `log_training_run` and
    `promote_by_alias` with MLflow, `promotion_verdict` with the verdict. First the lab
    file's own demonstration, run on the solved functions; then the solution's own
    demonstration, `python3 solutions/lab_01.py`, one paragraph of its `__main__` block
    per cell.
    """)
    explain(
        "Run the stub's own `__main__` block against the solved functions.",
        "It is what a student sees when the file is complete.",
        "The lines below are the stub's demonstration, verbatim.",
        "v2 refused, and the registry unchanged.")
    nb.step(f"{LABS}/01_the_gate.py", 0)
    explain(
        "Let the solution's paths resolve in the working copy.",
        "The solution puts its MLflow store beside the exercises folder it finds from "
        "`__file__`, which a notebook does not have.",
        "Points `__file__` at `solutions/lab_01.py` in the working copy.",
        "Its store goes under the copy's `out/lab_01_store`, not under your `exercises/`.")
    nb.code('__file__ = str(WORK / "solutions" / "lab_01.py")')
    solution_paragraphs(nb, explain, f"{S}/lab_01.py", [
        ("Import what the demonstration needs beyond the functions above.",
         "The block brings its own plotting and narration tools, and `shutil` to clear an old "
         "store.",
         "Imports `contextlib`, `io` and `shutil`, plotly, and the narration helpers; "
         "`_narrate` resolves to the module the set-up registered.",
         "Nothing runs yet."),
        ("Start the narration.",
         "Every line the solution prints goes through `say`, stamped with the seconds since "
         "the notebook began.",
         "Creates the narrator for Lab 1 and prints the demonstration's one-line summary.",
         "The timings at the start of each line differ from run to run; nothing else does."),
        ("Read what the two versions measured.",
         "The gate decides on these numbers and on nothing else.",
         "Loads `metrics.json` and prints each version's accuracy on the later 30 per cent of "
         "rows, its size on disk and its training rows.",
         "v1 at 0.8180, v2 at 0.7291 and 258 times larger."),
        ("Part (a): ask the gate about the measured candidate, then about a hypothetical "
         "better one, then roll back.",
         "The check grades three behaviours: a worse candidate refused with the registry "
         "untouched, a better one promoted, and rollback returning the previous version.",
         "Reads `registry.json`, asks `promote_if_better` about v2, prints the registry after "
         "the refusal, raises v2's accuracy five points in a copy of the metrics, asks again, "
         "and rolls back.",
         "Refused with both scores quoted; promoted when better; the rollback returns v1."),
        ("Part (b): log both models to a fresh MLflow store, with their signature, and put "
         "the alias on the better one.",
         "The real registry keeps the same record as the fifty-line one, and in addition "
         "enforces the columns.",
         "Deletes and reopens `out/lab_01_store`; for each version wraps the artefact as a "
         "pipeline and logs its parameters, metrics, environment and a signature inferred from "
         "200 complete rows, sending MLflow's own messages to a sink; then moves `champion` to "
         "the more accurate version.",
         "Two registered versions, and `champion` on v1."),
        ("Read the registry back, and the champion's signature.",
         "A record is only as useful as the query that reads it months later.",
         "Fetches each registered version and its run through `MlflowClient`, tabulates alias, "
         "accuracy, size and settings, then loads the champion by alias and prints the input "
         "names its signature enforces.",
         "The table the check reads, and speed, rssi1, rssi2, rssiC in that order."),
        ("Draw what matters against what is persuasive.",
         "A candidate 258 times larger looks like progress; the gate looks only at accuracy "
         "on later rows.",
         "Draws two bar charts side by side — accuracy on later data, megabytes on disk — "
         "through the notebook's `save_figure`.",
         "The bigger bar belongs to the worse model."),
        ("Part (c): import what the verdict's measurements need.",
         "The verdict is handed evidence measured here — Lab 3's bills and Lab 4's "
         "percentile — rather than numbers quoted from a slide.",
         "Imports NumPy and `time`.",
         "Nothing runs yet."),
        ("Take the test period, its truth and the operator's prices.",
         "Both bills must be counted on the same rows at the same cutoff.",
         "Splits the table by time at 70 per cent, marks the aboard rows, and sets the priced "
         "cutoff 0.2 with a false positive at 1 and a false negative at 4.",
         "The 3,241 test rows of Block three."),
        ("Define the bill of one version at the priced cutoff.",
         "Each version must be prepared with its own stored transform, or the comparison "
         "measures the preparation rather than the model.",
         "Fills and scales the test rows with the version's stored medians, means and standard "
         "deviations, asks for probabilities, and prices the false positives at 1 and the false "
         "negatives at 4; returns the bill, the prepared rows and the artefact.",
         "One function, applied to both versions in the next cell."),
        ("Price both versions, and the best rule that uses no model.",
         "The verdict promotes only a candidate cheaper than the model in service and cheaper "
         "than using no model at all.",
         "Bills v1 and v2, and takes the cheaper of the two no-model rules: answer aboard to "
         "everyone, or not aboard to everyone.",
         "Three bills for the verdict to compare."),
        ("Measure the 95th-percentile duration of one request, on this machine.",
         "A latency budget can only be checked against a duration that happened.",
         "Times 200 single-row predictions in milliseconds, sorts them and takes the "
         "nearest-rank 95th percentile. The rows timed are `prepared_v1` on `artefact_v1`, "
         "the model in service; the next cell files the result as `candidate_latency_p95`.",
         "A number that changes from run to run, which is why the deck reports work in "
         "decision nodes instead."),
        ("Hand the verdict its evidence, twice.",
         "The verdict must be right on the measured case and on the uncomfortable one: a "
         "candidate more accurate than the model in service and still dearer.",
         "Gathers the ten measured numbers, prints them, asks `promotion_verdict`, then raises "
         "only the candidate's accuracy five points above v1 and asks again.",
         "HOLD, because the candidate's realised cost is higher; and HOLD again for the more "
         "accurate but dearer candidate. Accuracy alone does not buy a promotion."),
        ("Say what the check grades.",
         "A student should know what a pass means before running the check.",
         "Prints the solution's one-line summary of the check's requirements.",
         "Nothing is computed here."),
    ])
