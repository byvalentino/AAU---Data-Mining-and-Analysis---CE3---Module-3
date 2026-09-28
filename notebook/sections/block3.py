"""Block three — what a mistake costs: the priced cutoff, calibration and the
departure where the price bends; Laboratory 3 (Module3.1.pptx slides 57-79)."""

S = "Module 3/exercises/solutions"
LABS = "Module 3/exercises/labs"
FIGS = "Module 3/slides/make_figs.py"
JENSEN = "standalone/jensens_inequality/slides/make_figs.py"

from sections.block1 import solution_paragraphs  # noqa: E402


def build(nb, explain) -> None:
    nb.md("""
    ---
    ## Block three — What a mistake costs

    *Slides: "A probability becomes a decision only when it is compared with a cutoff",
    "One half assumes both mistakes cost the same, and they rarely do" and "The cheapest
    cutoff for the operator makes the passenger wait".*

    The model answers with a probability; the service must answer with a decision. A
    false positive sends a bus for nobody: price C_FP = 1 vehicle-hour. A false negative
    leaves a passenger waiting: a second vehicle plus a contract penalty of about three,
    so C_FN = 4. The prices are the operator's, not the model builder's — a passenger
    would set them differently — and a cutoff that minimises cost favours whoever wrote
    the cost. The passenger's only protection is C_FN: whoever sets it decides how long
    passengers wait.
    """)
    explain(
        "Define the block's prices and constants as the Lab 3 solution writes them.",
        "Every number of this block — the two prices, the departure's seats and mean demand, "
        "the calibration bands — is set once, in the lab, and read from there.",
        "Copies the lab number, C_FP = 1, C_FN = 4, 12 seats, a mean demand of 10, the band "
        "edges and the smallest band worth believing from `solutions/lab_03.py`, verbatim.",
        "The operator's prices, in vehicle-hours, are now names the rest of the block uses.")
    nb.source(f"{S}/lab_03.py", "LAB", "COST_FALSE_POSITIVE", "COST_FALSE_NEGATIVE", "SEATS",
              "MEAN_DEMAND", "CALIBRATION_EDGES", "CALIBRATION_MINIMUM_ROWS")

    # --- slides 60-62: the threshold ---------------------------------------------------------
    nb.md("### The cost-sensitive threshold\n\n"
          "*Slides: \"Definition — the cost-sensitive threshold\" and \"The cutoff is derived in "
          "four steps, and the model appears in none of them\".*")
    explain(
        "Derive the cutoff from the two prices, symbolically.",
        "Answering aboard costs C_FP times the chance of not aboard; answering not aboard "
        "costs C_FN times the chance of aboard. Answer aboard when the first is no larger "
        "[@elkan2001, § 1.3].",
        "Solves (1 − p) C_FP = p C_FN for p with sympy, checks that the difference of the "
        "two costs rises with p (so the rule is p at or above the solution), displays the "
        "threshold and the decision rule, and evaluates the operator's prices (1, 4) and a "
        "passenger's (0, any).",
        "t* = C_FP / (C_FP + C_FN) = 0.2. The model appears nowhere in it, so two services "
        "may share a model and differ in their cutoff; and it needs a calibrated p.")
    nb.equation("threshold", '''
    p = sp.Symbol("p", positive=True)
    C_fp, C_fn = sp.symbols(r"C_{fp} C_{fn}", positive=True)
    advantage = p * C_fn - (1 - p) * C_fp            # cost of "not aboard" minus cost of "aboard"
    boundary = sp.solve(sp.Eq(advantage, 0), p)[0]
    t_star = C_fp / (C_fp + C_fn)
    formula(sp.Le((1 - p) * C_fp, p * C_fn), sp.Le(C_fp, p * (C_fp + C_fn)),
            sp.Ge(p, t_star))
    print("the two costs are equal at p =", boundary, "; the difference rises with p at rate",
          sp.diff(advantage, p), "> 0, so aboard is the cheaper answer for every p at or above it")
    assert sp.simplify(boundary - t_star) == 0
    formula(sp.Eq(sp.Symbol("t^{*}"), t_star, evaluate=False),
            sp.Equivalent(sp.Symbol(r"\\mathrm{decide\\ aboard}"), sp.Ge(p, sp.Symbol("t^{*}"))))
    agrees("operator's cutoff, C_FP = 1, C_FN = 4", t_star.subs({C_fp: 1, C_fn: 4}), 0.2, 3)
    print("passenger's cutoff, C_FP = 0: t* =", sp.limit(t_star, C_fp, 0))
    ''', slides=["3.1 #60", "3.1 #62"])
    explain(
        "Define the threshold as the Lab 3 solution writes it.",
        "The lab states the cutoff as one line of arithmetic on the two prices.",
        "Copies `threshold_from_costs` from `solutions/lab_03.py`, verbatim.",
        "The next cell checks it against the sympy expression.")
    nb.source(f"{S}/lab_03.py", "threshold_from_costs", cite="[@elkan2001, § 1.3]")
    explain(
        "Check that the lab's function is the formula, and fix the priced cutoff.",
        "The check grades the ratio on twenty-nine pairs of costs, not only the round ones.",
        "Turns the sympy threshold into a function and compares it with "
        "`threshold_from_costs` on a grid of 29 cost pairs; then sets `PRICED` from the "
        "operator's prices and records how the slide writes the boundary.",
        "Identical everywhere. The rule is written p ≥ t* in the definition, the lab and the "
        "check; the derivation slide ends with p > t* — for a continuous score the two differ "
        "on a set of probability nought, and at equality either answer costs the same.")
    nb.code('''
    as_function = sp.lambdify((C_fp, C_fn), t_star)
    pairs = [(a, b) for a in (0.5, 1, 2, 3.7, 10) for b in (0.25, 1, 4, 7.5, 20, 33)][:29]
    worst = max(abs(threshold_from_costs(a, b) - as_function(a, b)) for a, b in pairs)
    print(f"{len(pairs)} cost pairs: largest difference between the lab's function and the formula {worst:.1e}")
    PRICED = threshold_from_costs(COST_FALSE_POSITIVE, COST_FALSE_NEGATIVE)
    beside("the decision rule at the boundary", "p ≥ t* (definition card, Lab 3, check, figures)",
           "p > t* (last line of the derivation slide)",
           "the derivation's strict inequality; Elkan (2001) notes either class is optimal at equality")
    ''')
    explain(
        "Draw how a probability becomes a priced decision, on the model's own test rows.",
        "The slide draws the probability axis cut at t = 0.2: left of it the service says "
        "not aboard and risks a false negative at 4; right of it, aboard, and a false "
        "positive at 1.",
        "Plots the champion's probabilities on the 3,241 test rows, aboard and not aboard "
        "separately, with the cutoff, and counts the mistakes each side of it makes.",
        "The false positives are many and cheap, the false negatives few and dear: that "
        "trade is the whole point of the cutoff.")
    nb.figure("priced_decision", '''
    said = P_TEST >= PRICED
    false_positives = int((said & (Y_TEST == 0)).sum())
    false_negatives = int((~said & (Y_TEST == 1)).sum())
    fig = go.Figure()
    for truth_value, colour, name in ((1, "#2A78D6", "truth: aboard"), (0, "#E07B39", "truth: not aboard")):
        fig.add_histogram(x=P_TEST[Y_TEST == truth_value], xbins=dict(start=0, end=1, size=0.025),
                          marker_color=colour, opacity=0.7, name=name)
    fig.add_vline(x=PRICED, line=dict(color="#C0392B", width=3), annotation_text="t = 0.2",
                  annotation_position="bottom right")
    fig.add_annotation(x=0.19, y=1, yref="paper", xanchor="right", showarrow=False, align="right",
                       text=f"Predict: NOT aboard<br>truth aboard → false negative, C_FN = 4<br>{false_negatives} here")
    fig.add_annotation(x=0.62, y=1, yref="paper", showarrow=False, align="left",
                       text=f"Predict: aboard<br>truth not aboard → false positive, C_FP = 1<br>{false_positives} here")
    fig.update_layout(barmode="overlay", title="How a probability becomes a priced decision: t = C_FP / (C_FP + C_FN) = 0.2",
                      xaxis_title="the model's probability of aboard, p", yaxis_title="test rows",
                      legend=dict(orientation="h", y=-0.2))
    show(fig, "priced_decision", height=500)
    print(f"K(0.2) = {COST_FALSE_POSITIVE:g} x {false_positives} + {COST_FALSE_NEGATIVE:g} x {false_negatives} "
          f"= {COST_FALSE_POSITIVE * false_positives + COST_FALSE_NEGATIVE * false_negatives:,.0f} vehicle-hours")
    ''', slides=["3.1 #63"], treatment="lab data: the slide's drawn axis, with the model's own "
                                       "probabilities on the test rows")

    # --- slides 64-66: realised cost -----------------------------------------------------------
    nb.md("### The realised cost of a threshold\n\n"
          "*Slides: \"Definition — realised cost of a threshold\", \"The realised cost is counted "
          "in four steps, one test row at a time\" and \"A line of arithmetic saves 22.6 per cent "
          "against the usual cutoff\".*")
    explain(
        "Write the realised cost.",
        "Price and cost are different things: the price is fixed per mistake, the cost is "
        "the price times how often the mistake happened. The realised cost is a count on a "
        "finite set of rows, not a forecast [@elkan2001; @provost2013, ch. 7].",
        "Writes K(t) in sympy with the two counts as symbols, and each count as a sum of "
        "indicators over the n rows, then checks the slide's toy example.",
        "Ten false positives and five false negatives cost 30.")
    nb.equation("realised_cost", '''
    n_fp, n_fn = sp.symbols("n_FP n_FN", nonnegative=True)
    K = C_fp * n_fp + C_fn * n_fn
    row, rows_n = sp.Symbol("i", integer=True), sp.Symbol("n", integer=True, positive=True)
    p_i, y_i, t_ = sp.IndexedBase("p")[row], sp.IndexedBase("y")[row], sp.Symbol("t")
    one = sp.Function(r"\\mathbf{1}")
    formula(sp.Eq(sp.Symbol("K(t)"), K, evaluate=False))
    formula(sp.Eq(n_fp, sp.Sum(one(sp.And(sp.Ge(p_i, t_), sp.Eq(y_i, 0))), (row, 1, rows_n))),
            sp.Eq(n_fn, sp.Sum(one(sp.And(sp.Lt(p_i, t_), sp.Eq(y_i, 1))), (row, 1, rows_n))))
    agrees("toy example: 10 false positives and 5 false negatives", K.subs({C_fp: 1, C_fn: 4, n_fp: 10, n_fn: 5}), 30, 0)
    ''', slides=["3.1 #64"])
    explain(
        "Define the realised cost as the Lab 3 solution writes it.",
        "The lab counts the two mistakes on the rows and prices each.",
        "Copies `realised_cost` from `solutions/lab_03.py`, verbatim.",
        "The next cell prices the two cutoffs with it.")
    nb.source(f"{S}/lab_03.py", "realised_cost", cite="[@elkan2001; @provost2013, ch. 7]")
    explain(
        "Price the priced cutoff and the habit on the 3,241 test rows.",
        "The saving is the whole argument of the block, so it is counted, not estimated.",
        "Computes the realised cost at 0.2 and at 0.5 with the lab's function, checks both "
        "and their difference against the slide, and against `make_figs.py`'s own count.",
        "1,590 at 0.2 against 2,054 at 0.5: the priced cutoff saves 464.")
    nb.code('''
    COST_PRICED = realised_cost(P_TEST, Y_TEST, PRICED, COST_FALSE_POSITIVE, COST_FALSE_NEGATIVE)
    COST_HALF = realised_cost(P_TEST, Y_TEST, 0.5, COST_FALSE_POSITIVE, COST_FALSE_NEGATIVE)
    agrees("realised cost at 0.2", COST_PRICED, 1590, 0)
    agrees("realised cost at 0.5", COST_HALF, 2054, 0)
    agrees("saving", COST_HALF - COST_PRICED, 464, 0)
    assert COST_PRICED == total_cost(P_TEST, Y_TEST, PRICED), "the lab and make_figs.py count differently"
    ''')
    explain(
        "Define the cost-curve figure as the deck's script writes it.",
        "The slide's picture is drawn by `make_figs.py`; the notebook draws it with the same "
        "function.",
        "Copies `figure_threshold()` from `make_figs.py`, verbatim.",
        "The next cell draws it on the model's test rows.")
    nb.source(FIGS, "figure_threshold")
    explain(
        "Draw the cost of every cutoff from 0 to 1, with the priced one and the habit marked.",
        "The formula never looked at the curve; it used only the two prices. The picture "
        "shows whether it lands near the bottom anyway.",
        "Runs `figure_threshold()` on the test rows, checks the saving the slide prints, and "
        "finds the exact cheapest cutoff by scanning the predicted probabilities.",
        "22.6 per cent saved against 0.5. The curve is flat near its bottom — the cheapest "
        "cutoff on these rows is 0.217 at 1,580 — so the prices need only be roughly right.")
    nb.figure("threshold", '''
    figure_threshold(P_TEST, Y_TEST)
    agrees("saving against 0.5, per cent", 100 * (1 - COST_PRICED / COST_HALF), 22.6, 1)
    cheapest_at, cheapest_cost = cheapest_threshold(P_TEST, Y_TEST)
    print(f"the cheapest cutoff on these rows: {cheapest_at:.3f}, costing {cheapest_cost:,.0f} "
          f"(make_figs.py records {json.loads((SLIDES / 'measured.json').read_text())['cheapest_threshold']['value']})")
    ''', slides=["3.1 #66"], treatment="exact: the slide's own code")

    # --- slides 67-69 -------------------------------------------------------------------------
    nb.md("### Against a rule that uses no model\n\n"
          "*Slides: \"How much is the model worth? Compare it with a rule that uses no model\", "
          "\"Four rules ranked by what they cost the operator\" and \"The cheaper service is the "
          "less accurate one\".*")
    explain(
        "Rank the four rules by what they cost, with their accuracy beside.",
        "A saving against the habit measures the cutoff; a saving against the best rule "
        "that uses no model measures the model. Quote both — quoting only the first oversells "
        "the service [@provost2013, ch. 7].",
        "Prices always-not-aboard, the 0.5 cutoff, always-aboard and the 0.2 cutoff on the "
        "3,241 rows, with each rule's accuracy.",
        "The trained model with the library's cutoff costs more than answering aboard to "
        "everything. The cheaper service, at 0.2, is the less accurate one: accuracy prices "
        "every mistake at one.")
    nb.code('''
    rules = [("Always answer not aboard", 1.01, False), ("The usual cutoff of 0.5", 0.5, True),
             ("Always answer aboard", 0.0, False), ("The priced cutoff of 0.2", PRICED, True)]
    ranked = pd.DataFrame([{"rule": name, "cutoff": ("1" if cut > 1 else f"{cut:g}"),
                            "cost on the test rows": realised_cost(P_TEST, Y_TEST, cut, COST_FALSE_POSITIVE, COST_FALSE_NEGATIVE),
                            "accuracy": round(float(((P_TEST >= cut).astype(int) == Y_TEST).mean()), 4),
                            "uses the model": uses} for name, cut, uses in rules]).set_index("rule")
    display(ranked)
    costs = ranked["cost on the test rows"]
    agrees("always not aboard", costs.iloc[0], 6312, 0)
    agrees("always aboard, the no-model floor", costs.iloc[2], 1663, 0)
    agrees("accuracy of always aboard (the aboard base rate)", ranked["accuracy"].iloc[2], 0.4869, 4)
    agrees("accuracy at 0.5", ranked["accuracy"].iloc[1], 0.818, 3)
    agrees("accuracy at 0.2", ranked["accuracy"].iloc[3], 0.5261, 4)
    agrees("0.5 is worse than no model by, per cent", 100 * (COST_HALF / costs.iloc[2] - 1), 24, 0)
    agrees("0.2 saves against the floor, per cent", 100 * (1 - COST_PRICED / costs.iloc[2]), 4.4, 1)
    ''')

    # --- slides 70-73: calibration -------------------------------------------------------------
    nb.md("### Calibration: is the probability honest?\n\n"
          "*Slides: \"Definition — calibration: is the probability honest?\", \"Definition — "
          "honesty is checked in bands, not one row at a time\", \"The reliability diagram is "
          "the band check drawn as a picture, in five steps\" and \"This model is not honest: it "
          "promises more aboard than happens\".*")
    explain(
        "Write calibration, and look at the band the slide quotes.",
        "The cutoff 0.2 trusted p to be the real chance of aboard. A forecaster is well "
        "calibrated if, among the cases where the prediction was x, the long-run frequency is "
        "also x [@degroot1983, § 2]; good accuracy or a good area under the ROC curve does "
        "not make the probabilities honest [@niculescu2005, § 1].",
        "Writes the definition in sympy and measures the band from 0.6 to 0.8.",
        "The model promises 0.74 there and 49 in 100 were aboard. The slide says it said "
        "p ≈ 0.74 on one row in five; on these test rows that band holds one row in "
        "nineteen.")
    nb.equation("calibration", '''
    s_ = sp.Symbol("s", real=True)
    formula(sp.Eq(sp.Function("P")(sp.Symbol(r"Y = 1 \\mid \\mathrm{score} = s")), s_),
            sp.Contains(s_, sp.Interval(0, 1)))
    band = (P_TEST >= 0.6) & (P_TEST < 0.8)
    agrees("the band 0.6 to 0.8 promises", P_TEST[band].mean(), 0.74, 2)
    agrees("and delivers", Y_TEST[band].mean(), 0.49, 2)
    beside("share of test rows in that band", f"{int(band.sum())} of {len(P_TEST):,} = 1 in {len(P_TEST) / band.sum():.0f}",
           "1 in 5", "the slide's proportion does not match the band's row count; the 1,678 rows of the band "
                     "0.2 to 0.4 are the ones near one in two")
    ''', slides=["3.1 #70"])
    explain(
        "Write the band check.",
        "One row cannot check one probability. Group rows with similar p into bands of width "
        "0.2, compare the mean promise with the share that happened, and drop a band too "
        "small to believe.",
        "Writes the three statements of the definition card in sympy, with membership of "
        "band j written as an indicator b_ij over the n rows, so the band's size, its mean "
        "promise and its share that happened are sums.",
        "The largest gap is the one-number summary of calibration.")
    nb.equation("reliability", '''
    row, band, rows_n = sp.Symbol("i", integer=True), sp.Symbol("j", integer=True), sp.Symbol("n", integer=True, positive=True)
    p_i, y_i, e_ = sp.IndexedBase("p")[row], sp.IndexedBase("y")[row], sp.IndexedBase("e")
    in_band = sp.Symbol("b_{ij}")
    size_j, r_ = sp.Abs(sp.Symbol("B_j")), sp.Symbol("r", positive=True)
    one = sp.Function(r"\\mathbf{1}")
    formula(sp.Eq(in_band, one(sp.And(sp.Le(e_[band], p_i), sp.Lt(p_i, e_[band + 1])))),
            sp.Eq(size_j, sp.Sum(in_band, (row, 1, rows_n))), sp.Ge(size_j, r_))
    formula(sp.Eq(sp.Symbol(r"\\mathrm{predicted}_j"), sp.Sum(in_band * p_i, (row, 1, rows_n)) / size_j),
            sp.Eq(sp.Symbol(r"\\mathrm{actual}_j"), sp.Sum(in_band * y_i, (row, 1, rows_n)) / size_j))
    formula(sp.Eq(sp.Symbol(r"\\mathrm{largest\\ gap}"),
                  sp.Function(r"\\max_j")(sp.Abs(sp.Symbol(r"\\mathrm{predicted}_j") - sp.Symbol(r"\\mathrm{actual}_j")))))
    ''', slides=["3.1 #71"])
    explain(
        "Define the band check as the Lab 3 solution writes it.",
        "The check grades the bands against its own arithmetic and against two invented "
        "models.",
        "Copies `reliability` from `solutions/lab_03.py`, verbatim. The last band is closed at "
        "the top, so a probability of exactly one is not quietly dropped.",
        "The function the diagram below is drawn from.")
    nb.source(f"{S}/lab_03.py", "reliability", cite="[@degroot1983; @niculescu2005, § 4]")
    explain(
        "Define the reliability diagram as the deck's script writes it.",
        "The slide's picture is drawn by `make_figs.py`.",
        "Copies `figure_reliability()` from `make_figs.py`, verbatim.",
        "The next cell feeds it the lab's bands.")
    nb.source(FIGS, "figure_reliability")
    explain(
        "Draw the reliability diagram of the model in service.",
        "Points below the diagonal promise more than happens; with inflated p, the formula "
        "sends buses too often.",
        "Computes the bands with the lab's `reliability()`, rounds them as `make_figs.py` "
        "records them, runs `figure_reliability()`, and scores two invented models: one "
        "that answers 0.6 to everything and one that answers 0.3 and 0.7 and means it.",
        "Below the diagonal almost everywhere; the worst band misses by 0.247. The derived "
        "cutoff is a starting point, and a sweep over cutoffs measures what the dishonesty "
        "costs. Of the two invented models, the more accurate one is the badly calibrated one.")
    nb.figure("reliability", '''
    DIAGRAM = reliability(P_TEST, Y_TEST, CALIBRATION_EDGES, CALIBRATION_MINIMUM_ROWS)
    bands = {label: {"rows": b["rows"], "predicted": round(b["predicted"], 3), "actual": round(b["actual"], 3)}
             for label, b in DIAGRAM["bands"].items()}
    figure_reliability(bands)
    agrees("band 0.6-0.8, promised", bands["p60_80"]["predicted"], 0.738, 3)
    agrees("band 0.6-0.8, happened", bands["p60_80"]["actual"], 0.491, 3)
    agrees("largest gap", DIAGRAM["largest_gap"], 0.247, 3)
    loud, loud_truth = np.full(2000, 0.6), np.r_[np.ones(1760, int), np.zeros(240, int)]
    quiet = np.r_[np.full(1000, 0.3), np.full(1000, 0.7)]
    quiet_truth = np.r_[np.ones(300, int), np.zeros(700, int), np.ones(700, int), np.zeros(300, int)]
    for name, scores, outcomes in (("answers 0.6 to everything", loud, loud_truth),
                                   ("answers 0.3 and 0.7, and means it", quiet, quiet_truth)):
        print(f"  invented model that {name:34} accuracy {((scores >= 0.5) == outcomes).mean():.2f}, "
              f"largest gap {reliability(scores, outcomes)['largest_gap']:.2f}")
    ''', slides=["3.1 #73"], treatment="exact: the slide's own code, on the lab's reliability()")

    # --- slides 75-78: the departure -----------------------------------------------------------
    nb.md("### Second use of the prices: planning seats\n\n"
          "*Slides: \"Second use of the prices: planning seats, where demand is uncertain\", "
          "\"Jensen's inequality: cost of the average day ≠ average cost of the days\", \"A plan "
          "checked on the average forecast is optimistic\" and \"The mean cost of the departure "
          "is computed in four steps\".* (The stress test of the cutoff at six cost ratios is on "
          "a hidden slide; it returns as a practice question at the end.)")
    explain(
        "Write the cost at the mean and the mean cost, and compute the second exactly.",
        "One departure, 12 seats, demand X Poisson with mean 10, a missed passenger priced at "
        "4. The shortfall max(X − 12, 0) bends, and for a bending cost the average of the "
        "costs is at least the cost of the average [@jensen1906]. Poisson is the law of "
        "independent arrivals; the capacity question is the classic inventory problem "
        "[@arrow1951, § 3].",
        "Writes both in sympy, with the Poisson probabilities as sympy's own Poisson mass "
        "function, and sums the series exactly — the tail above 12 as the mean minus the "
        "finite sum below it.",
        "Cost at the mean 0; mean cost 2.124 vehicle-hours. The plan checked on the average "
        "day reports nothing wrong.")
    nb.equation("jensen_departure", '''
    from sympy.stats import Poisson, density
    lam, c, d = sp.symbols(r"\\lambda c d", positive=True)
    C_miss = sp.Symbol(r"C_{\\mathrm{fn}}")
    demand = Poisson("D", lam)
    mass = density(demand)(d)                          # lambda^d e^(-lambda) / d!
    formula(sp.Eq(sp.Symbol(r"g(E[D])"), C_miss * sp.Max(0, sp.Symbol("E[D]") - c), evaluate=False))
    formula(sp.Eq(sp.Symbol(r"E[g(D)]"), C_miss * sp.Sum((d - c) * mass, (d, c + 1, sp.oo)), evaluate=False),
            sp.Eq(sp.Symbol("P(D = d)"), mass))
    # E[max(D - c, 0)] = E[D] - c + sum_{d <= c} (c - d) P(D = d): an exact, finite sum
    shortfall = 10 - 12 + sum((12 - dd) * sp.exp(-10) * sp.Integer(10)**dd / sp.factorial(dd) for dd in range(13))
    exact_mean_cost = 4 * shortfall
    agrees("mean cost, exact (sympy)", sp.N(exact_mean_cost, 30), 2.124, 3)
    ''', slides=["3.1 #75"])
    explain(
        "Define the departure's two numbers as the Lab 3 solution writes them.",
        "The optional fourth function of the lab computes the cost at the mean demand and "
        "the mean cost over the demand.",
        "Copies `cost_at_mean_versus_mean_cost` from `solutions/lab_03.py`, verbatim.",
        "The next cell compares it with the exact sum.")
    nb.source(f"{S}/lab_03.py", "cost_at_mean_versus_mean_cost", cite="[@jensen1906; @arrow1951, § 3]")
    explain(
        "Check the lab's departure against the exact sum, and the slide's three example days.",
        "A numerical sum can be truncated or rounded; the exact one cannot.",
        "Runs the lab's function with 12 seats, a mean demand of 10 and a miss priced at 4, "
        "compares its mean cost with the sympy value to 10⁻¹², and prices days with 13, 15 "
        "and 18 passengers.",
        "The same 2.124 both ways; one, three and six passengers left behind cost 4, 12 "
        "and 24.")
    nb.code('''
    at_mean, mean_cost = cost_at_mean_versus_mean_cost(SEATS, MEAN_DEMAND, COST_FALSE_NEGATIVE)
    print(f"lab: cost at the mean {at_mean}, mean cost {mean_cost:.12f};  exact {sp.N(exact_mean_cost, 13)}")
    assert abs(mean_cost - float(exact_mean_cost)) < 1e-12
    for passengers, stated in ((13, 4), (15, 12), (18, 24)):
        agrees(f"cost of a day with {passengers} passengers", COST_FALSE_NEGATIVE * max(passengers - SEATS, 0), stated, 0)
    ''')
    explain(
        "Define the two routes of Jensen's inequality, with the standalone deck's own code.",
        "The picture is the inequality itself: average the inputs and apply the curve, or "
        "apply the curve and average the outputs. For a curve that bends upward the second "
        "is higher. The picture and its argument follow a video lecture by {@rich2020}.",
        "Copies the sample, the curve and `draw_two_routes()` from "
        "`standalone/jensens_inequality/slides/make_figs.py`, verbatim. The sample is thirty "
        "seeded normal draws; the curve is exp(0.75 x).",
        "Nothing is drawn yet; the gap between the two routes is already computed, as `GAP`.")
    nb.source(JENSEN, "BLUE", "GREY", "SCREEN", "LAYOUT", "RATE", "SEED", "SPREAD", "SAMPLE_SIZE",
              "TRUNCATE_AT", "draw_sample", "SAMPLE", "curve", "slope", "MEAN_INPUT", "OUTPUTS",
              "OUTPUT_OF_MEAN", "MEAN_OUTPUT", "GAP", "RATIO", "grid", "X_LOW", "Y_LOW", "X_RUG",
              "Y_RUG", "rugs", "axis_marks", "frame", "draw_two_routes", cite="[@rich2020; @jensen1906]")
    explain(
        "Show the standalone script's figure in place instead of writing it to disk.",
        "There `save()` writes `figures/<name>.png` for its deck.",
        "Defines `save()` with the script's layout and font sizes, drawing the figure in place "
        "— the one function of that script that is not copied.",
        "`draw_two_routes()` runs unchanged in the next cell.")
    nb.code('''
    def save(fig, name, width=1100, height=740):
        """The standalone script's save(), shown in place instead of written to figures/."""
        settings = dict(LAYOUT)
        if fig.layout.margin.t is not None:
            settings.pop("margin")
        fig.update_layout(width=width, height=height, **settings)
        for note in fig.layout.annotations:
            if note.font.size is None or note.font.size == 16:
                note.font.size = SCREEN
        fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID)
        fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID)
        display(Image(fig.to_image(format="png", width=width, height=height, scale=1)))
        return name
    ''')
    explain(
        "Draw the two routes, and write the inequality they illustrate.",
        "The slide shows this picture; the inequality is its one-line summary.",
        "Runs `draw_two_routes()`, checks the gap the slide prints, and writes Jensen's "
        "inequality for a sample of n inputs in sympy: if the curve bends upward (its second "
        "derivative is positive), the curve at the mean input is at most the mean of the "
        "curve's outputs.",
        "A gap of 0.28 in the picture — the same shape as the departure's 2.124 "
        "vehicle-hours.")
    nb.figure("two_routes", '''
    draw_two_routes()
    agrees("the gap in the picture", GAP, 0.28, 2)
    f_, u_ = sp.Function("f"), sp.Symbol("x")
    j_, n_j = sp.Symbol("i", integer=True), sp.Symbol("n", integer=True, positive=True)
    X_i = sp.IndexedBase("X")[j_]
    formula(sp.Implies(sp.Gt(sp.Derivative(f_(u_), (u_, 2)), 0),
                       sp.Le(f_(sp.Sum(X_i, (j_, 1, n_j)) / n_j), sp.Sum(f_(X_i), (j_, 1, n_j)) / n_j)))
    ''', slides=["3.1 #76"], treatment="exact: the standalone deck's own code")
    explain(
        "Put the course's seed back.",
        "The standalone script has its own `SEED`, 20260903, and its cell just set the "
        "global name to it; the module's code reads 20200122.",
        "Restores the module's seed.",
        "Nothing below is drawn from the Jensen sample's seed.")
    nb.code('SEED = 20200122')
    explain(
        "Define the departure figure as the deck's script writes it.",
        "The slide's picture is drawn by `make_figs.py`.",
        "Copies `figure_jensen()` from `make_figs.py`, verbatim.",
        "The next cell feeds it the departure's numbers.")
    nb.source(FIGS, "figure_jensen")
    explain(
        "Draw the departure: the cost of each day's demand, the chance of that demand, and the "
        "two numbers.",
        "The same shape as the previous picture, with the module's own numbers.",
        "Computes the departure with `jensen_departure()` (copied from `make_figs.py` "
        "earlier), draws it with `figure_jensen()`, checks the two numbers the slide prints, "
        "and records where the slide's footer says they come from.",
        "Cost at the mean 0, mean cost 2.124. And the capacity that balances the two prices "
        "is the 0.8 quantile of demand, 13 seats — the same ratio as the cutoff, read as a "
        "demand quantile.")
    nb.figure("jensen", '''
    departure = jensen_departure()
    figure_jensen(departure)
    agrees("cost at the mean demand", departure["cost_at_mean_demand"], 0, 3)
    agrees("mean cost over the days", departure["mean_cost"], 2.124, 3)
    print(f"critical fractile C_fn / (C_fp + C_fn) = {departure['critical_fractile']}, "
          f"capacity at it: {departure['capacity_at_critical_fractile']} seats")
    beside("where the slide's footer says these numbers come from", "the Poisson mass function (make_figs.py, jensen_departure)",
           "\\"service/models.py, trained on Module 2's generated table\\"",
           "the figure's own credit line is right; the numbers line on that slide is the deck's template footer")
    ''', slides=["3.1 #77"], treatment="exact: the slide's own code")

    # --- the laboratory --------------------------------------------------------------------------
    nb.md("""
    ## Laboratory 3 — Price the threshold

    *Slide: "Lab 3 — Price the threshold: you write the three functions this block
    used".* `threshold_from_costs`, `realised_cost`, `reliability`, and an optional
    fourth, `cost_at_mean_versus_mean_cost`. What you take away: the prices, not the
    model, decide the cutoff; the cost, not accuracy, decides whether it is good; and a
    probability must be checked before it is trusted.
    """)
    nb.statement(f"{LABS}/03_the_threshold.py")
    nb.md("### The solution, step by step\n\n"
          "The four functions were defined above. First the lab file's own demonstration; then "
          "the solution's own, one paragraph of its `__main__` block per cell.")
    explain(
        "Run the stub's own `__main__` block against the solved functions.",
        "It is what a student sees when the file is complete.",
        "The lines below are the stub's demonstration, verbatim.",
        "0.200 against the habit's 0.500.")
    nb.step(f"{LABS}/03_the_threshold.py", 0)
    solution_paragraphs(nb, explain, f"{S}/lab_03.py", [
        ("Import what the demonstration needs beyond the functions above.",
         "The block draws with plotly, tabulates with pandas and narrates with `_narrate`.",
         "Imports plotly, pandas and the narration helpers the set-up registered. From here "
         "on the cells are `python3 solutions/lab_03.py`, one paragraph of its `__main__` "
         "block per cell.",
         "Nothing runs yet."),
        ("Start the narration.",
         "Every line the solution prints goes through `say`, stamped with the seconds since "
         "the notebook began.",
         "Creates the narrator for Lab 3 and prints the demonstration's one-line summary.",
         "The timings differ from run to run; nothing else does."),
        ("Load the champion and cut the test period.",
         "The cost must be counted on rows the model never trained on, and later in time "
         "than the rows it did.",
         "Loads v1's artefact and its stored transform, loads the table, and keeps the later "
         "30 per cent of rows.",
         "The same 3,241 rows as the rest of the notebook."),
        ("Score the test period.",
         "Prices multiply probabilities, so the probabilities come first.",
         "Fills and scales the test rows with the stored medians, means and standard "
         "deviations, asks the forest for the probability of aboard, and marks the truth.",
         "A probability and a true answer for every test row."),
        ("Derive the cutoff from the two prices.",
         "The cutoff is the operator's arithmetic, not the model's.",
         "Calls `threshold_from_costs(1, 4)` and prints the ratio.",
         "0.200, computed without looking at the model."),
        ("Price four policies on the test rows.",
         "A saving against the habit measures the cutoff; a saving against the best rule "
         "without a model measures the model.",
         "Computes the realised cost at 0.2, at 0.5, for answering aboard to everyone and for "
         "answering not aboard to everyone, and prints the two savings.",
         "22.6 per cent saved against 0.5, and 4.4 per cent against the no-model floor of "
         "1,663 — which 0.5 itself does not beat."),
        ("Tabulate the four policies with their accuracy beside their cost.",
         "Accuracy prices both mistakes the same; the table shows where that misleads.",
         "Builds one table of threshold, realised cost and accuracy for the four policies.",
         "The cheapest policy is not the most accurate one."),
        ("Check calibration band by band, and on two invented models.",
         "The derivation assumed the probabilities were honest without saying so.",
         "Computes the reliability bands of the test period, prints the largest gap, then "
         "scores a model that answers 0.6 to everything and one that answers 0.3 and 0.7 and "
         "means it.",
         "Largest gap 0.247; and of the two invented models, the more accurate is the one "
         "whose probabilities mean nothing."),
        ("Draw the reliability bands.",
         "The table of bands is easier to read as distance from the diagonal.",
         "Plots each band's promise against what happened, sized by its rows, beside the "
         "diagonal of perfect calibration.",
         "Almost every band sits below the diagonal: the model promises more aboard than "
         "happens."),
        ("Draw the realised cost at every cutoff.",
         "The formula chose 0.2 from the prices alone; the curve shows where that lands.",
         "Prices 201 cutoffs from 0 to 1, draws the curve with the no-model floor, and marks "
         "the priced cutoff and the habit.",
         "0.2 sits near the flat bottom; 0.5 sits above the floor."),
        ("Price the departure where the cost bends.",
         "A plan checked at the average demand is optimistic when the cost of a shortfall "
         "bends.",
         "Calls `cost_at_mean_versus_mean_cost(12, 10, 4)`, prints both numbers, and the "
         "capacity at the C_FN / (C_FP + C_FN) quantile of Poisson demand.",
         "0 at the mean, 2.124 on average; 13 seats at the 0.8 quantile."),
        ("Say what the check grades.",
         "A student should know what a pass means before running the check.",
         "Prints the solution's one-line summary of the check's requirements, with the "
         "departure's mean cost.",
         "Nothing new is computed here."),
    ])

    # --- appendix ---------------------------------------------------------------------------------
    nb.md("### Appendix — how to calibrate a model's probabilities\n\n"
          "*Slide: \"Appendix — how to calibrate a model's probabilities\" (3.1 and 3.2).* Do not "
          "retrain the model: fit a small second function g on its scores, on rows it never "
          "trained on, and check the result with the reliability bands on a third set.")
    explain(
        "Write the three calibration maps, and try the first two on the model in service.",
        "Platt scaling fits an S-shape and corrects best a sigmoid-shaped distortion; "
        "isotonic regression fits any never-decreasing step function, and so corrects any "
        "monotone distortion, but overfits when calibration rows are few "
        "[@niculescu2005, § 1–2; @platt1999; @zadrozny2002]; temperature scaling divides a "
        "network's logits by one number [@guo2017].",
        "Writes the three maps in sympy — the isotonic map as a function g that never "
        "decreases, temperature scaling as a softmax over the K classes — then splits the 3,241 test rows alternately into a "
        "calibration half and an evaluation half, fits a logistic curve and an isotonic step "
        "function to the model's probabilities on the first half, and measures the largest "
        "reliability gap on the second half before and after.",
        "Isotonic regression cuts the largest gap from about 0.28 to 0.08 without touching the "
        "forest, fitted on about 1,600 rows — above the thousand or so calibration cases from "
        "which Niculescu-Mizil and Caruana found it as good as Platt scaling or better "
        "[@niculescu2005, § 5]. Platt scaling does not help here: this forest over-promises "
        "in almost every band rather than bending both ends towards the middle. And honest probabilities do not make 0.2 cheaper on "
        "these rows — calibration buys the right to trust p, and the check afterwards is what "
        "tells you whether the priced cutoff still holds.")
    nb.equation("calibration_maps", '''
    a_, b_, s_, T_ = sp.symbols("a b s T")
    s_1, s_2 = sp.symbols("s_1 s_2")
    g_ = sp.Function("g")
    k_c, j_c, K_c = sp.Symbol("k", integer=True), sp.Symbol("j", integer=True), sp.Symbol("K", integer=True, positive=True)
    z_ = sp.IndexedBase("z")
    formula(sp.Eq(sp.Symbol(r"p_{\\mathrm{Platt}}"), 1 / (1 + sp.exp(a_ * s_ + b_)), evaluate=False))
    formula(sp.Eq(sp.Symbol(r"p_{\\mathrm{isotonic}}"), g_(s_)), sp.Implies(sp.Le(s_1, s_2), sp.Le(g_(s_1), g_(s_2))))
    formula(sp.Eq(sp.Symbol(r"p_{\\mathrm{temperature},\\,k}"),
                  sp.exp(z_[k_c] / T_) / sp.Sum(sp.exp(z_[j_c] / T_), (j_c, 1, K_c))))
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression
    fit_half, check_half = np.arange(len(P_TEST)) % 2 == 0, np.arange(len(P_TEST)) % 2 == 1
    platt = LogisticRegression().fit(P_TEST[fit_half].reshape(-1, 1), Y_TEST[fit_half])
    isotonic = IsotonicRegression(out_of_bounds="clip").fit(P_TEST[fit_half], Y_TEST[fit_half])
    for name, calibrated in (("raw forest", P_TEST[check_half]),
                             ("Platt scaling", platt.predict_proba(P_TEST[check_half].reshape(-1, 1))[:, 1]),
                             ("isotonic regression", isotonic.predict(P_TEST[check_half]))):
        gap = reliability(calibrated, Y_TEST[check_half])["largest_gap"]
        cost = realised_cost(calibrated, Y_TEST[check_half], PRICED, COST_FALSE_POSITIVE, COST_FALSE_NEGATIVE)
        print(f"  {name:20} largest gap {gap:.3f}   realised cost at 0.2 on the evaluation half {cost:,.0f}")
    ''', slides=["3.1 #84", "3.2 #36"])
