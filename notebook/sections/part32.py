"""Part 3.2 — Block four: training-serving skew, dataset shift and latency
percentiles; Laboratory 4 (Module3.2.pptx slides 1-28)."""

S = "Module 3/exercises/solutions"
LABS = "Module 3/exercises/labs"
FIGS = "Module 3/slides/make_figs.py"

from sections.block1 import solution_paragraphs  # noqa: E402


def build(nb, explain) -> None:
    nb.md("""
    ---
    # Part 3.2 — Deploying a model as a service, II

    *Deck: `Module3.2.pptx`.* The second lecture opens with a recap poster, the course's
    position slide, the case and the layered-data picture of 3.1. The poster is an
    AI-generated infographic and the others are the video frame and the medallion
    picture, none reproduced (see `figure_map.json`); the route of the case is the map
    drawn from the slice at the start of Part 3.1.

    ## Block four — Training–serving skew, and latency percentiles
    """)

    # --- slides 7-9: the skew ------------------------------------------------------------------
    nb.md("### The failure where nothing goes wrong\n\n"
          "*Slides: \"Look first: one model, the same 3,241 requests, two batches, all requests "
          "successful\", \"27.1 per cent of decisions changed and every request returned 200\" "
          "and \"A model does not know its columns by name, only by position\".*")
    explain(
        "Send the same 3,241 requests twice — once as trained, once with two columns swapped "
        "in the preparation — and draw both batches of answers.",
        "Every request succeeds both times, the status is 200 and the response time is "
        "unchanged. The model knows position one, two and three, not names: hand it signal "
        "strength where it expects speed and it answers just as confidently.",
        "Copies `figure_skew()` from `make_figs.py`, verbatim: it swaps the first two "
        "prepared columns, speed and rssi1, asks the champion again and draws both batches.",
        "The next cell runs it on the test period.")
    nb.source(FIGS, "figure_skew")
    explain(
        "Run the swap on the 3,241 test rows and count what changed.",
        "The damage of a skew is measured in decisions, at the cutoff the service uses.",
        "Runs `figure_skew()` on the champion and the prepared test rows, then counts the "
        "decisions that changed at the priced cutoff and the accuracy before and after.",
        "27.1 per cent of decisions changed and accuracy fell from 0.5261 to 0.4428, with no "
        "error anywhere. Only the spread of the probabilities changed — which a watcher of "
        "the outputs could have caught.")
    nb.figure("skew", '''
    SKEWED = figure_skew(ARTEFACT_V1, X_TEST, P_TEST)
    before_swap, after_swap = P_TEST >= PRICED, SKEWED >= PRICED
    agrees("decisions changed, per cent", 100 * (before_swap != after_swap).mean(), 27.1, 1)
    agrees("accuracy before the swap, at 0.2", ((before_swap.astype(int)) == Y_TEST).mean(), 0.5261, 4)
    agrees("accuracy after the swap, at 0.2", ((after_swap.astype(int)) == Y_TEST).mean(), 0.4428, 4)
    ''', slides=["3.2 #7", "3.2 #8"], treatment="exact: the slide's own code")

    # --- slides 10-12: definition and the pipeline ------------------------------------------------
    explain(
        "Write training–serving skew and its cure.",
        "Skew is the input prepared one way at fitting and another at asking; the most common "
        "cause is two code paths for the same preparation [@breck2019, § 4; @huyen2022]. The "
        "cure is one preparation, looped over the *stored* order and filled from the *stored* "
        "median, never from the request's keys or live traffic.",
        "Writes the two statements of the definition card in sympy: skew as the two "
        "preparations disagreeing, and the one preparation as a fill then a z-score, each "
        "with the stored constants of the j-th field in the stored order.",
        "One preparation, whose every constant was fixed at fitting.")
    nb.equation("skew", '''
    x_ = sp.Symbol("x")
    formula(sp.Equivalent(sp.Symbol(r"\\mathrm{skew}"),
                          sp.Ne(sp.Function(r"\\mathrm{prepare}_{\\mathrm{serve}}")(x_),
                                sp.Function(r"\\mathrm{prepare}_{\\mathrm{train}}")(x_))))
    stored_constant = lambda what: sp.Symbol(rf"\\mathrm{{{what}}}_{{f_j}}")      # fitted once, on the training rows
    x_fj, v_ = sp.IndexedBase("x")[sp.Symbol("f_j")], sp.Symbol("v_j")
    formula(sp.Eq(v_, sp.Piecewise((x_fj, sp.Ne(x_fj, sp.Symbol(r"\\mathrm{null}"))), (stored_constant("median"), True))),
            sp.Eq(sp.Symbol(r"\\mathrm{input}_j"), (v_ - stored_constant("mean")) / stored_constant("std"), evaluate=False))
    ''', slides=["3.2 #10"])
    explain(
        "Define the cure as the Lab 4 solution writes it.",
        "The cure loops over the *stored* order and fills from the *stored* median, never "
        "from the request's keys or live traffic.",
        "Copies Lab 4's constants and `prepare` from `solutions/lab_04.py`, verbatim.",
        "The next cell checks it against the service's own preparation.")
    nb.source(f"{S}/lab_04.py", "LAB", "PRICED_THRESHOLD", "TARGET", "SKEW_EXCHANGE", "prepare",
              cite="[@breck2019; @huyen2022]")
    explain(
        "Check that `prepare()` is the service's preparation, whatever order the keys arrive in.",
        "Two implementations of one preparation are how skew starts; the lab's must equal "
        "the stored transform's to the last bit.",
        "Cuts the test period from the table, sends every test row through `prepare()` with "
        "its keys reversed, and compares with `apply_transform()` on the same rows.",
        "Identical to the last bit, whatever order the request's keys arrive in.")
    nb.code('''
    whole_table = build_table()
    TEST_ROWS = whole_table.iloc[int(len(whole_table) * 0.7):].reset_index(drop=True)
    stored = ARTEFACT_V1["transform"]
    by_prepare = np.array([prepare({f: (None if pd.isna(r[f]) else float(r[f])) for f in reversed(FEATURES)}, stored)[0]
                           for _, r in TEST_ROWS.iterrows()])
    by_service = apply_transform(TEST_ROWS, stored).to_numpy()
    print(f"prepare() with the keys reversed against apply_transform(), {len(TEST_ROWS):,} rows: "
          f"largest difference {np.abs(by_prepare - by_service).max():.1e}")
    ''')
    nb.md("""
    *Slides: "One example pipeline. This service prepares its input in four steps" and
    "Skew can be caught. Four checks, from cheapest to strongest".* The four steps are
    the stored order, the stored median, and the stored mean and standard deviation;
    any cleaning, imputation, encoding or scaling of Modules 1 and 2 could stand in each
    step, as long as each is fitted on the training rows and replayed at serving. The
    four checks: validate the request against the stored schema; compare live inputs
    with training inputs field by field; compare live outputs with training outputs —
    the spread of the probabilities, which is exactly what changed in the skew figure
    above; and run the same preparation on a stored table and on single requests and
    require identical numbers, which is Lab 4 [@breck2019, § 3–4].
    """)
    explain(
        "Record where the third check's changing spread is actually drawn.",
        "Slide 12 says \"Slide 74 shows this spread changing\". This deck has no slide 74; "
        "the number belongs to the earlier single deck, whose slide 74 is now hidden in 3.1 "
        "and shows the cost-ratio stress test.",
        "Prints the difference and adds it to the closing table.",
        "The spread is the skew figure above, 3.2 slides 7 and 8.")
    nb.code('''
    beside("where the third check's changing spread is shown", "the skew figure, 3.2 slides 7-8 (above)",
           "\\"Slide 74\\"", "a reference to a slide number of the earlier single deck; 3.1 slide 74 is hidden "
                            "and shows the cost-ratio stress test, not the spread")
    ''')

    # --- slides 13-14: causing it ---------------------------------------------------------------
    explain(
        "Write the measurement of a caused skew.",
        "A failure you have produced is one you recognise. The share is of *decisions*, "
        "because a probability that moves without crossing the cutoff costs nothing.",
        "Writes the definition card's formula in sympy — the share of the n rows whose "
        "decision at t differs once the exchange π is applied — and the two conditions on it.",
        "A caused skew is a share of decisions, no failures, and nought after the cure.")
    nb.equation("caused_skew", '''
    row, rows_n, t_ = sp.Symbol("i", integer=True), sp.Symbol("n", integer=True, positive=True), sp.Symbol("t")
    pi_ = sp.Symbol(r"\\pi")
    one = sp.Function(r"\\mathbf{1}")
    p_i, p_i_pi = sp.IndexedBase("p")[row], sp.IndexedBase(r"p^{\\pi}")[row]
    changed = sp.Function(r"\\mathrm{changed}")
    formula(sp.Eq(changed(pi_), sp.Sum(one(sp.Ne(one(sp.Ge(p_i, t_)), one(sp.Ge(p_i_pi, t_)))), (row, 1, rows_n)) / rows_n))
    formula(sp.Eq(sp.Function(r"\\mathrm{failures}")(pi_), 0),
            sp.Eq(changed(sp.Symbol(r"\\pi_{\\mathrm{stored\\ order}}")), 0))
    ''', slides=["3.2 #13"])
    explain(
        "Define the caused skew as the Lab 4 solution writes it.",
        "The lab causes the skew in the requests themselves — two fields exchanged, the row "
        "built in arrival order — and then cures it.",
        "Copies `cause_and_cure_skew` from `solutions/lab_04.py`, verbatim.",
        "The next cell runs it for the slide's two exchanges.")
    nb.source(f"{S}/lab_04.py", "cause_and_cure_skew", cite="[@breck2019; @huyen2022]")
    explain(
        "Measure the caused skew for the slide's two exchanges, then for all six.",
        "The damage depends on which two columns are exchanged, so one pair is not the "
        "whole story.",
        "Runs `cause_and_cure_skew` on the test period for speed with rssi1 and for rssi2 "
        "with rssiC; then measures all six exchanges the way `make_figs.py` does, by swapping "
        "prepared columns, and compares with the recorded `measured.json`.",
        "27.1 per cent for speed and rssi1, 6.9 for rssi2 and rssiC — the damage is a "
        "property of the pair — no failed request in either, and nought after the cure.")
    nb.code('''
    for exchange, stated in ((("speed", "rssi1"), 27.1), (("rssi2", "rssiC"), 6.9)):
        report = cause_and_cure_skew(TEST_ROWS, ARTEFACT_V1, exchange, PRICED_THRESHOLD)
        agrees(f"{exchange[0]} and {exchange[1]} exchanged: decisions changed, per cent",
               report["decisions_changed_percent"], stated, 1)
        print(f"    failures {report['failures']}, changed after the cure {report['decisions_changed_percent_after_cure']}")
    pairs = {}
    for first in range(len(FEATURES)):
        for second in range(first + 1, len(FEATURES)):
            exchanged = X_TEST.copy()
            exchanged[:, [first, second]] = exchanged[:, [second, first]]
            moved = ARTEFACT_V1["model"].predict_proba(exchanged)[:, 1] >= PRICED
            pairs[f"{FEATURES[first]} and {FEATURES[second]}"] = round(100 * float((before_swap != moved).mean()), 1)
    recorded = json.loads((SLIDES / "measured.json").read_text())["skew_by_exchange"]["value"]
    by_pair = pd.DataFrame({"decisions changed, per cent": pairs})
    by_pair["slides/measured.json"] = list(recorded.values())
    display(by_pair)
    assert list(pairs.values()) == list(recorded.values())
    ''')
    nb.md("*Slide: \"Lab 4 causes the skew in five steps, then measures it\".* Send every "
          "request the right way; swap two fields and build the row in arrival order; send "
          "again; count the changed decisions; rebuild in the stored order and confirm nothing "
          "changes.")

    # --- slides 15-17: one preparation, two paths -------------------------------------------------
    nb.md("### One preparation, two paths\n\n"
          "*Slides: \"One preparation, written once, is used for one request and for a whole "
          "table\", \"Definition — one preparation, two paths\" and \"The two paths are compared "
          "in three steps, and must agree to a billionth\".*")
    explain(
        "Write the agreement the two paths owe each other.",
        "A batch answers a whole table at once. If it is a second implementation of the "
        "preparation, there are two things to keep correct, maintained by different people "
        "[@huyen2022].",
        "Writes the definition card's inequality in sympy, row by row.",
        "The two paths must agree to a billionth on every row.")
    nb.equation("two_paths", '''
    row_ = sp.Symbol("i", integer=True)
    batch_i = sp.Indexed(sp.IndexedBase(r"\\mathrm{batch}(\\mathrm{frame})"), row_)
    single_i = sp.Function(r"\\mathrm{model}")(sp.Function(r"\\mathrm{prepare}")(sp.Indexed(sp.IndexedBase(r"\\mathrm{row}"), row_)))
    formula(sp.Lt(sp.Abs(sp.Add(batch_i, -single_i, evaluate=False), evaluate=False),
                  sp.Pow(sp.Pow(10, 9, evaluate=False), -1, evaluate=False)))
    ''', slides=["3.2 #16"])
    explain(
        "Define the batch path as the Lab 4 solution writes it.",
        "A batch answers a whole table at once, and must reuse the request path's "
        "preparation rather than a second implementation of it.",
        "Copies `batch_predict` from `solutions/lab_04.py`, verbatim.",
        "The next cell compares it with the request path.")
    nb.source(f"{S}/lab_04.py", "batch_predict", cite="[@huyen2022]")
    explain(
        "Compare the batch path with the request path on four hundred test rows.",
        "Agreement must be checked on real rows, empty signal strengths included.",
        "Answers the first 400 test rows once as a table with `batch_predict` and once one "
        "request at a time through `prepare()`, and prints the largest difference.",
        "A difference of exactly nought is the normal result, because it is the same code.")
    nb.code('''
    rows400 = TEST_ROWS.head(400)
    batch = batch_predict(rows400, ARTEFACT_V1)
    single = np.array([ARTEFACT_V1["model"].predict_proba(prepare(
        {f: (None if pd.isna(r[f]) else float(r[f])) for f in FEATURES}, stored))[0][1] for _, r in rows400.iterrows()])
    print(f"batch path against request path, 400 rows: largest difference {np.abs(batch - single).max():.1e}")
    ''')

    # --- slides 18-21: dataset shift --------------------------------------------------------------
    nb.md("""
    ### Skew is in the pipeline; shift is in the world

    *Slides: "Skew is a change in the pipeline, and dataset shift is a change in the
    world" and "Definition — dataset shift".* Skew is one measurement prepared two ways,
    and one preparation cures it. Dataset shift is the world changing between fitting
    and asking, and no pipeline cures it: the inputs, the share of aboard, or the
    instrument may change. A changed unit of measurement is what Storkey calls domain
    shift [@storkey2009, § 1.8]; the deck calls skew its self-inflicted case. That reading
    is the deck's: Storkey's examples of domain shift — units, lighting, inflation — all
    come from outside the pipeline, and the shift he calls deliberate is a rebalanced
    training set [@storkey2009, § 1.3, § 1.7]. A model must not be assumed valid under
    shift because it was valid without it; Module 4 measures how different is different,
    and Module 5 watches for it.
    """)
    explain(
        "Draw a speed distribution on the day the model was fitted and on the next day.",
        "The slide draws two constructed bells for its example \"trains now run faster, so "
        "speed values move\". The shipped vehicle slice has a real, modest version of it. It "
        "is the vehicle's speed, standing in for the concept: the model's own `speed` "
        "feature is the phone's, in the generated traces, and those do not shift.",
        "Reads the slice and draws, for each day, the size of the speed — its absolute value "
        "— while the vehicle moves. The absolute value matters: on 22 January about half the "
        "moving readings are negative, because the vehicle went back and forth on one stretch "
        "(see the route map), so the signed means differ mostly by sign. Then prints the "
        "same summaries, and the mean of the model's `speed` feature in the generated "
        "traces for each day.",
        "Moving at 1.51 metres per second on average on 22 January and 1.79 on 23 January: "
        "a modest covariate shift, P(x) changing. Whether P(aboard | speed) also moved, "
        "unlabelled traffic cannot say — and the lab data cannot show it, since the "
        "generator uses one set of rates for every day.")
    nb.figure("speed_two_days", '''
    bus = pd.read_csv("data/bus_slice.csv.gz", low_memory=False)
    day = pd.to_datetime(bus["utc_time"], utc=True).dt.date.astype(str)
    moving = bus["speed"].ne(0)
    fig = go.Figure()
    for which, colour, name in (("2020-01-22", "#2A78D6", "22 January — the day the model was fitted on"),
                                ("2020-01-23", "#DF8E2E", "23 January — the day it is asked about")):
        here = (day == which) & moving
        fig.add_histogram(x=bus.loc[here, "speed"].abs(), histnorm="probability density", xbins=dict(start=0, end=3.6, size=0.1),
                          marker_color=colour, opacity=0.6, name=name)
    fig.update_layout(barmode="overlay",
                      title="The vehicle's speed while moving, by day — a stand-in for the slide's shifted feature",
                      xaxis_title="size of the vehicle's speed, |speed|, metres per second", yaxis_title="density",
                      legend=dict(orientation="h", y=-0.2))
    show(fig, "speed_two_days", height=480)
    for which in ("2020-01-22", "2020-01-23"):
        speeds = bus.loc[day == which, "speed"]
        in_motion = speeds[speeds.ne(0)]
        phones = pd.read_parquet(f"data/phones_{which}.parquet")["speed"]
        print(f"{which}: vehicle moving {100 * speeds.ne(0).mean():.1f} % of {len(speeds):,} readings, "
              f"mean |speed| while moving {in_motion.abs().mean():.2f} m/s (median {in_motion.abs().median():.2f}), "
              f"negative {100 * (in_motion < 0).mean():.1f} % of moving readings, signed mean {speeds.mean():.3f}; "
              f"the model's speed feature (generated phones) mean {phones.mean():.2f} m/s")
    ''', slides=["3.2 #19", "3.2 #21"], treatment="lab data: the slides' constructed bells replaced "
                                                  "by the size of the vehicle's speed on the two days, "
                                                  "a stand-in for the model's feature")
    explain(
        "Answer the slide's challenge on a sigmoid: does the average prediction follow the "
        "prediction at the average input?",
        "The model is nonlinear. By Jensen's inequality the average of a convex function "
        "lies above the function of the average, and below it for a concave one "
        "[@jensen1906]; in risk analysis the mistake is called the flaw of averages "
        "[@savage2009].",
        "Draws the slide's curve — the logistic function, which its chart samples at the "
        "integers from −8 to 8 — and computes, by numerical integration, the average output "
        "for normal inputs of spread 0.5 and 2, centred left of the midpoint, at it, and "
        "right of it; then writes Jensen's inequality in sympy for a curve bending up "
        "(second derivative at least nought) and one bending down.",
        "Question 1: no — the average output is not f(mean x). Question 2: yes — with the mean "
        "fixed, a wider spread moves the average output. Question 3: left of the midpoint, "
        "where the curve is convex, it pushes the average up; right of it, where it is "
        "concave, down; at the midpoint the two cancel.")
    nb.figure("sigmoid", '''
    from scipy import integrate
    xs = np.linspace(-8, 8, 401)
    logistic = lambda x: 1 / (1 + np.exp(-x))
    chart = [0, 0.001, 0.002, 0.007, 0.018, 0.047, 0.119, 0.269, 0.5, 0.731, 0.881, 0.953, 0.982, 0.993, 0.998, 0.999, 1]
    assert np.allclose(logistic(np.arange(-8, 9)), chart, atol=6e-4), "the slide's curve is the logistic function"
    fig = go.Figure()
    fig.add_scatter(x=xs, y=logistic(xs), mode="lines", line=dict(color="#2A78D6", width=3), name="f(x) = P(aboard | x)")
    rows = []
    for centre in (-2.0, 0.0, 2.0):
        for spread, symbol in ((0.5, "circle"), (2.0, "diamond")):
            mean_output = integrate.quad(lambda x: logistic(x) * stats.norm.pdf(x, centre, spread), -40, 40)[0]
            rows.append({"mean of x": centre, "spread of x": spread, "f(mean x)": round(float(logistic(centre)), 4),
                         "mean of f(x)": round(mean_output, 4), "Jensen gap": round(mean_output - float(logistic(centre)), 4)})
            fig.add_scatter(x=[centre], y=[mean_output], mode="markers", marker=dict(size=12, symbol=symbol, color="#C0392B"),
                            name=f"mean of f(x), x ~ N({centre:g}, {spread:g}²)")
    fig.update_layout(title="A highly nonlinear model: sigmoid output against one input x",
                      xaxis_title="x (low … midpoint … high)", yaxis_title="f(x)", legend=dict(orientation="h", y=-0.25))
    show(fig, "sigmoid", height=560)
    display(pd.DataFrame(rows))
    f_, x_ = sp.Function("f"), sp.Symbol("x")
    E_ = sp.Function(r"\\mathbb{E}")
    bend = sp.Derivative(f_(x_), (x_, 2))
    formula(sp.Implies(sp.Ge(bend, 0), sp.Ge(E_(f_(x_)), f_(E_(x_)))),       # convex
            sp.Implies(sp.Le(bend, 0), sp.Le(E_(f_(x_)), f_(E_(x_)))))       # concave
    ''', slides=["3.2 #20"], treatment="exact: the slide's curve is the logistic function, "
                                       "sampled at the integers; the challenge answered numerically")
    explain(
        "Write the joint distribution two ways and name the three shifts.",
        "Shift means one factor of P(x, y) changed between training and serving: P(x) with "
        "P(y | x) fixed is covariate shift [@storkey2009, § 1.4]; P(y) with P(x | y) fixed is "
        "prior probability, or label, shift [@storkey2009, § 1.5]; P(y | x) itself is concept "
        "shift. The three-way naming is the deck's, after {@morenotorres2012}; Storkey's "
        "chapter uses neither \"concept shift\" nor \"label shift\", and lists more kinds "
        "of shift than three.",
        "Writes the product rule in sympy, both factorisations.",
        "Covariate shift is drawn above on the real speeds. The other two cannot be drawn on "
        "the lab data, and the next two figures say why.")
    nb.equation("dataset_shift", '''
    P_ = sp.Function("P")
    x_s, y_s = sp.symbols("x y")
    given = lambda a, b: sp.Symbol(rf"{a} \\mid {b}")
    formula(sp.Eq(P_(x_s, y_s), P_(x_s) * P_(given("y", "x"))), sp.Eq(P_(x_s, y_s), P_(y_s) * P_(given("x", "y"))))
    ''', slides=["3.2 #21"])
    explain(
        "Draw concept shift as the slide draws it.",
        "Concept shift is P(y | x) changing: after a sensor is recalibrated, the same speed "
        "means a different chance of aboard. The generated phone traces hold the rule fixed "
        "on both days by construction, so there is none to show.",
        "Redraws the slide's two curves from the chart's own values.",
        "Illustrative: the serving curve is the training curve moved right by about three and "
        "a half units.")
    nb.figure("concept_shift", '''
    training = [0, 0.001, 0.002, 0.007, 0.018, 0.047, 0.119, 0.269, 0.5, 0.731, 0.881, 0.953, 0.982, 0.993, 0.998, 0.999, 1]
    serving = [0, 0, 0, 0, 0.001, 0.001, 0.004, 0.01, 0.027, 0.069, 0.168, 0.354, 0.599, 0.802, 0.917, 0.968, 0.988]
    fig = go.Figure()
    fig.add_scatter(y=training, mode="lines", line=dict(color="#2A78D6", width=3), name="Training")
    fig.add_scatter(y=serving, mode="lines", line=dict(color="#DF8E2E", width=3), name="Serving")
    fig.update_layout(title="Concept shift: P(y | x) changes — chance of aboard at each x (illustrative)",
                      xaxis=dict(tickvals=[0, 16], ticktext=["low", "high"], title="x"), yaxis_title="P(aboard | x)",
                      yaxis_range=[0, 1])
    show(fig, "concept_shift", height=420)
    ''', slides=["3.2 #21"], treatment="illustrative: the slide's chart values, redrawn exactly")
    explain(
        "Draw label shift as the slide draws it, beside the lab data's own shares.",
        "Label shift is P(y) changing: a new timetable puts more people aboard, and each class "
        "looks the same as before.",
        "Redraws the slide's bars (70/30 then 35/65, illustrative), and measures the aboard "
        "share of the generated traces on the two days.",
        "The generated traces are aboard 56.3 per cent of the time on both days — "
        "`make_phones.py` uses one measured share for every day — so the lab data carries no "
        "label shift.")
    nb.figure("label_shift", '''
    shares = {which: 100 * pd.read_parquet(f"data/phones_{which}.parquet")["label2"].eq("IN").mean()
              for which in ("2020-01-22", "2020-01-23")}
    fig = make_subplots(rows=1, cols=2, subplot_titles=("the slide (illustrative)", "the generated phone traces"))
    for name, values, colour in (("Training", [70, 30], "#2A78D6"), ("Serving", [35, 65], "#DF8E2E")):
        fig.add_bar(x=["not aboard", "aboard"], y=values, name=name, marker_color=colour, row=1, col=1)
    for (which, share), colour in zip(shares.items(), ("#2A78D6", "#DF8E2E")):
        fig.add_bar(x=["not aboard", "aboard"], y=[100 - share, share], name=which, marker_color=colour,
                    opacity=0.6, row=1, col=2)
    fig.update_yaxes(title_text="share of rows, per cent", range=[0, 100])
    fig.update_layout(title="Label shift: P(y) changes — share of rows in each class", barmode="group")
    show(fig, "label_shift", height=440)
    for which, share in shares.items():
        print(f"{which}: aboard {share:.1f} per cent of rows")
    ''', slides=["3.2 #21"], treatment="illustrative: the slide's bars redrawn exactly, beside "
                                       "the lab data's measured shares")

    # --- slides 22-24: latency ---------------------------------------------------------------------
    nb.md("### Is the model fast enough to serve?\n\n"
          "*Slides: \"Second half of block four — is the model fast enough to serve?\", "
          "\"Definition — percentile latency, nearest rank\" and \"The percentile is computed in "
          "three steps, and the answer is a real request\".*")
    explain(
        "Write the latency gate and the nearest-rank percentile, and work the slide's example.",
        "A model can pass the accuracy gate and still be too slow. Report the 95th "
        "percentile, not the mean: the tail is what the slow user sees [@dean2013]. Nearest "
        "rank reports a duration that actually occurred [@hyndman1996, Definition 1].",
        "Writes both in sympy — the percentile as the r-th smallest duration, the gate as "
        "an implication — then takes the slide's hundred requests, 94 at 10 ms and 6 at 1 s, "
        "and computes each row of its table.",
        "Mean 69 ms describes nobody; the 95th percentile says 1 s. The throughput the slide "
        "prints, about 1,450 requests a second, is a hundred divided by the mean in seconds; "
        "served one after another, a hundred requests that take 6.94 s in total go through at "
        "14.4 a second.")
    nb.equation("percentile", '''
    p_q, n_q = sp.symbols("p n", positive=True)
    rank = sp.ceiling(p_q * n_q / 100)
    r_q, i_q = sp.Symbol("r", integer=True, positive=True), sp.Symbol("i", integer=True, positive=True)
    ordered_at = lambda j: sp.Symbol(rf"x_{{({sp.latex(j)})}}")      # x_(j), the j-th smallest duration
    formula(sp.Eq(sp.Function("Q")(p_q), ordered_at(r_q)), sp.Eq(r_q, rank),
            sp.Le(ordered_at(i_q), ordered_at(i_q + 1)))
    formula(sp.Implies(sp.Symbol(r"\\mathrm{promote}"), sp.Le(sp.Symbol("p_{95}"), sp.Symbol(r"\\mathrm{time\\ budget}"))))
    hundred = [10.0] * 94 + [1000.0] * 6
    ordered = sorted(hundred)
    at = lambda q: ordered[int(rank.subs({p_q: q, n_q: len(ordered)})) - 1]
    agrees("mean, ms", np.mean(hundred), 69, 0)
    agrees("median (p50), ms", at(50), 10, 0)
    agrees("p95, ms", at(95), 1000, 0)
    agrees("p99, ms", at(99), 1000, 0)
    agrees("maximum, ms", max(hundred), 1000, 0)
    beside("throughput of the same 100 requests", f"{len(hundred) / (sum(hundred) / 1000):.1f} requests per second, served one after another",
           "~1,450 req/s", f"1,450 = 100 / {np.mean(hundred) / 1000:.3f} s divides the count by the mean in seconds, not by the total")
    agrees("work per request, candidate against baseline (decision nodes)", WORK_MULTIPLE, 17.7, 1)
    ''', slides=["3.2 #22", "3.2 #23"])
    explain(
        "Define the percentile as the Lab 4 solution writes it.",
        "Software packages disagree by default, so a report must say which definition it used.",
        "Copies `percentile_latency` from `solutions/lab_04.py`, verbatim.",
        "The next cell checks which definition it is.")
    nb.source(f"{S}/lab_04.py", "percentile_latency", cite="[@hyndman1996; @dean2013]")
    explain(
        "Check that the lab's percentile is Hyndman and Fan's Definition 1.",
        "Definition 1 is the inverse of the empirical distribution function, which NumPy "
        "implements as its `inverted_cdf` method [@hyndman1996, Definition 1].",
        "Compares `percentile_latency` with NumPy's `inverted_cdf` at every whole percentile, "
        "on the slide's hundred requests and on 200 seeded log-normal durations.",
        "The same function. Every percentile it reports is a duration that happened.")
    nb.code('''
    durations = np.random.default_rng(SEED).lognormal(1.0, 0.8, 200)
    for sample in (hundred, durations):
        worst = max(abs(percentile_latency(sample, q) - np.percentile(sample, q, method="inverted_cdf"))
                    for q in range(1, 101))
        print(f"{len(sample)} durations: largest difference from inverted_cdf over p = 1..100: {worst}")
    ''')

    # --- the laboratory ----------------------------------------------------------------------------
    nb.md("""
    ## Laboratory 4 — Cause the skew, then cure it

    *Slide: "Lab 4 — Cause the skew, then cure it".* Part (a): swap two fields in every
    request and measure what changed; the check measures the same share its own way, and
    the two have to agree; then cure it. Part (b): ask the champion by name, and watch a
    renamed column be refused. Twenty-five minutes; causing the skew is most of it.
    """)
    nb.statement(f"{LABS}/04_skew_and_speed.py")
    explain(
        "Define the platform's cure as the Lab 4 solution writes it.",
        "The registered model matches columns by name, restores the stored order and hands "
        "the artefact its own preparation; a renamed or missing column is refused rather than "
        "answered wrongly. (The deck and the docstring credit the signature to Zaharia et al. "
        "(2018); that paper predates MLflow's signatures — see the closing notes.)",
        "Copies `ask_registered` from `solutions/lab_04.py`, verbatim.",
        "The last of the five functions; the others were defined above.")
    nb.source(f"{S}/lab_04.py", "ask_registered",
              cite="[@zaharia2018, the deck's attribution — see the closing notes]")
    nb.md("### The solution, step by step\n\nFirst the lab file's own demonstration; then the "
          "solution's own, `python3 solutions/lab_04.py`, one paragraph of its `__main__` block "
          "per cell.")
    explain(
        "Run the stub's own `__main__` block against the solved functions.",
        "It is what a student sees when the file is complete.",
        "The lines below are the stub's demonstration, verbatim.",
        "A mean probability for the last 500 rows, and the 95th percentile of 1 to 100: 95.")
    nb.step(f"{LABS}/04_skew_and_speed.py", 0)
    solution_paragraphs(nb, explain, f"{S}/lab_04.py", [
        ("Import what the demonstration needs beyond the functions above.",
         "The block draws with plotly and narrates with `_narrate`.",
         "Imports plotly and the narration helpers the set-up registered.",
         "Nothing runs yet."),
        ("Start the narration.",
         "Every line the solution prints goes through `say`, stamped with the seconds since "
         "the notebook began.",
         "Creates the narrator for Lab 4 and prints the demonstration's one-line summary.",
         "The timings differ from run to run; nothing else does."),
        ("Load the champion and cut the test period.",
         "The skew and the speed are measured on the rows the model never trained on.",
         "Loads v1's artefact and its stored transform, loads the table and keeps the later "
         "30 per cent of rows.",
         "The same 3,241 rows as the rest of the notebook."),
        ("Show the skew on one request.",
         "One visible wrong answer teaches more than a percentage; a change in the third "
         "decimal would teach that the failure is small.",
         "Finds the first complete test row on which exchanging the first and last prepared "
         "positions, speed and rssiC, moves the probability by more than 0.1; prepares it with "
         "its keys in the stored order and reversed; and asks the raw pickle about the right "
         "row and the exchanged one.",
         "Reversed keys change nothing, because `prepare()` loops over the stored order; the "
         "exchanged row is answered without a word, with a different probability."),
        ("Cause the skew over the whole test period, then cure it.",
         "The slide's number is produced here, in the requests rather than in a matrix.",
         "Runs `cause_and_cure_skew` with the lab's exchange, speed and rssi1, and again with "
         "rssi2 and rssiC, and prints the share of decisions changed, the accuracy before and "
         "after, the failed requests and the share changed after the cure.",
         "27.1 and 6.9 per cent of decisions changed, no failed request, and nought after the "
         "cure."),
        ("Draw both batches of probabilities.",
         "The only visible trace of the skew is the spread of the answers.",
         "Prepares the test rows, swaps the first two columns, asks the model both ways and "
         "overlays the two histograms.",
         "Every request succeeded both times; only the shape of the probabilities moved."),
        ("Apply the platform's cure: ask the registered model by name.",
         "The registered model knows its columns by name, so it can reorder a request or "
         "refuse it.",
         "Loads `models:/aboard@champion`, prints its signature, and asks it about the "
         "reversed request, a request with rssiC renamed, and one with rssi1 missing.",
         "The reversed request gets the right probability; the renamed and the missing column "
         "are refused (`None`, which the service turns into a 422)."),
        ("Measure honest speed on this machine.",
         "A latency is reported by nearest rank, so each percentile is a duration that "
         "happened.",
         "Times 200 single requests to the champion in milliseconds, prints the mean and the "
         "50th, 95th and 99th percentiles, repeats the textbook case of 94 fast and 6 slow "
         "requests, and draws the sorted durations with the percentiles marked.",
         "The durations change from run to run — they belong on no slide; the textbook case "
         "gives a mean of 69.4 ms and a 95th percentile of 1,000 ms."),
        ("Prove the batch path is the request path.",
         "Two implementations of one preparation are two things to keep correct.",
         "Answers the whole test period with `batch_predict`, and again one request at a "
         "time through `prepare()`, and prints the largest difference and the two means.",
         "A difference of nought over 3,241 rows."),
        ("Say what the check grades.",
         "A student should know what a pass means before running the check.",
         "Prints the solution's one-line summary of the check's requirements.",
         "Nothing is computed here."),
    ])

    # --- slides 26-28 ------------------------------------------------------------------------------
    nb.md("## One request through the whole day\n\n"
          "*Slides: \"One request travels through the whole day in seven steps\", \"The same "
          "request ends with a priced decision and a signed answer\" and \"Three kinds of bias "
          "were met today, and two are still ahead\".*")
    explain(
        "Follow one request — speed 1.2 metres per second and three signal strengths — "
        "through the four blocks.",
        "Four blocks, one request, and about forty lines of code: the door, the stored "
        "preparation, the registry's version, the priced cutoff and the signed answer.",
        "Checks the request at the door, refuses it with one field absent, answers it with "
        "one signal strength empty, and answers it in full at 0.2 and at 0.5.",
        "At 0.2 it is aboard; at one half the same request would answer not aboard, and a "
        "passenger would wait. Months later, a complaint about this answer is settled by "
        "reading the answer itself.")
    nb.code('''
    print("1-2. at the door:", validate(GOOD_REQUEST) or "no complaint")
    print("3.   rssi1 absent:", respond({k: v for k, v in GOOD_REQUEST.items() if k != "rssi1"}, CONTRACT, SERVED, 0.2)["status"],
          "  rssi1 empty:", respond({**GOOD_REQUEST, "rssi1": None}, CONTRACT, SERVED, 0.2)["status"])
    print("4-5. prepared in the stored order:", [round(v, 4) for v in prepare(GOOD_REQUEST, SERVED["transform"])[0]])
    answer_priced = respond(GOOD_REQUEST, CONTRACT, SERVED, 0.2)
    answer_half = respond(GOOD_REQUEST, CONTRACT, SERVED, 0.5)
    print("6-7. the signed answer:", answer_priced)
    print("     at one half:      ", answer_half)
    assert answer_priced["decision"] == "aboard" and answer_half["decision"] == "not aboard"
    ''')
    explain(
        "Recompute the numbers of the bias table.",
        "Bias here is a systematic error, one that leans the same way every time: the "
        "inherited cutoff (modelling), the accuracy that picks the wrong cutoff (evaluation), "
        "and the same model prepared two ways (deployment). Dataset shift and feedback loops "
        "are Modules 4 and 5.",
        "Checks the three measurements the slide prints against the values computed above.",
        "22.6 per cent, 0.818 against 0.5261, 27.1 per cent.")
    nb.code('''
    agrees("modelling: saved by the priced cutoff, per cent", 100 * (1 - COST_PRICED / COST_HALF), 22.6, 1)
    agrees("evaluation: accuracy at 0.5", ((P_TEST >= 0.5).astype(int) == Y_TEST).mean(), 0.818, 3)
    agrees("evaluation: accuracy at 0.2", ((P_TEST >= 0.2).astype(int) == Y_TEST).mean(), 0.5261, 4)
    agrees("deployment: decisions changed by the swap, per cent", 100 * (before_swap != after_swap).mean(), 27.1, 1)
    ''')
