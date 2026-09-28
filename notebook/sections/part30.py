"""Part 3.0 — Beer and diapers: from a story to a decision (slides/Module3.0.pptx)."""

SIM = "Module 3/slides/simulate_receipts.py"
FIG = "Module 3/slides/make_figs_3.0.py"


def build(nb, explain, lines, figure_lines) -> None:
    nb.md("""
    ---
    # Part 3.0 — Beer and diapers: from a story to a decision

    *Deck: `Module3.0.pptx`. Slides: "Beer and diapers — from a story to a decision",
    the aims, and "What you need already, and three beliefs to drop".*

    The session connects Modules 1 and 2 to a question outside any pipeline: a pattern
    in receipts is not yet a decision. It works backwards from a sales curve to the
    data, the model and the test, and ends with a definition of data mining and data
    science. Three beliefs are dropped on the way: that items bought together are bought
    because of each other — they may only share a purchase cycle, such as the weekly
    shop [@manchanda1999]; that a pattern true in every group is true in the whole
    [@simpson1951]; and that averaging over windows makes data normal — it does so only
    when the windows are alike.

    **The story and the finding** (*slide: "The story is a legend; the documented finding
    is a time-of-day pattern"*). The legend says a shop moved beer beside diapers and
    sales rose; no record of that exists. What is documented is a 1992 Teradata analysis
    of 1.2 million Osco Drug receipts, in which beer and diapers were bought together
    more often between 17:00 and 19:00; the products were not moved, so no sales effect
    was ever measured [@power2002]. No link with age or sex was reported [@whitehorn2006].

    **The data of this part.** There is no real dataset for this case, so the deck
    simulates one year of till receipts, and every number and figure of Module 3.0
    comes from that simulation. The notebook runs the same simulation, with the same
    seed, and recomputes every number from it.
    """)

    # --- the simulation --------------------------------------------------------------
    nb.md("### One year of simulated receipts")
    explain(
        "Simulate the year of receipts every 3.0 slide is drawn from.",
        "A causal argument needs to know the truth it is arguing about. A simulation "
        "states it: four shopper groups (man or woman, baby at home or not), each with a "
        "chance of buying beer and a chance of buying diapers given beer; a card that "
        "parents hold more often; a summer peak for beer; an evening boost for parents at "
        "17 and 18 h. The receipt records none of the groups — only card holders are known.",
        "Runs `slides/simulate_receipts.py` verbatim: 365 days, a footfall per day drawn "
        "from a gamma law (weekdays about 1,500 receipts, weekends about 2,400), and one "
        "row per receipt with the hour, the day type, the card holder's segment if there is "
        "a card, and whether beer and diapers were bought. It writes the receipts to "
        "`receipts.csv.gz` in the working copy.",
        "About 630,000 receipts. By construction, beer and diapers go together inside "
        "every group and apart in the whole — Simpson's paradox, planted so it can be found.")
    lines(SIM, "import json", "    df.to_csv(f, index=False)",
          cite="[@simpson1951; @power2002]")
    explain(
        "Measure the simulated year the way the deck measures it.",
        "Every number on the 3.0 slides is one of these measurements.",
        "Runs the second half of `simulate_receipts.py` verbatim: the two-by-two table of "
        "beer against diapers for all receipts, for card holders, for parents and for "
        "non-parents, with support, confidence and lift; the share of receipts with both "
        "items by hour and by day; the daily beer counts by day type; and the layout "
        "example. It writes `measured_3.0.json` and the daily and hourly tables.",
        "The next cell checks that this run reproduces the recorded file exactly.")
    lines(SIM, "# ---------------------------------------------------------------- measures",
          'json.dump(m, open("measured_3.0.json", "w"), indent=1)',
          cite="[@agrawal1993; @brin1997]")
    explain(
        "Compare this run's measurements with the ones the deck was built from.",
        "`slides/measured_3.0.json` is the recorded archive of the 3.0 deck's numbers. If the "
        "simulation is deterministic, the two files are the same.",
        "Loads the recorded file and compares it with the one just written, key by key.",
        "Identical: every 3.0 number below is the deck's number, recomputed.")
    nb.code('''
    recorded = json.loads((SLIDES / "measured_3.0.json").read_text())
    rerun = json.loads(Path("measured_3.0.json").read_text())
    differing = [key for key in recorded if recorded[key] != rerun.get(key)]
    assert not differing, f"the simulation no longer reproduces measured_3.0.json: {differing}"
    print(f"measured_3.0.json reproduced exactly: {len(recorded)} entries, "
          f"{rerun['all']['receipts']:,} receipts, seed {rerun['seed']}")
    ''')

    # --- slide 5 --------------------------------------------------------------------
    nb.md("### A single year of sales cannot separate a layout change from the season\n\n"
          "*Slide: \"A single year of sales cannot separate a layout change from the season\".* "
          "The five 3.0 figures are drawn by the deck's own script, `slides/make_figs_3.0.py`, "
          "on the simulation above.")
    explain(
        "Load the simulated tables the way the figure script loads them.",
        "Running the deck's figure script on this run's output is the surest way to "
        "reproduce the deck's figures.",
        "Runs the first lines of `make_figs_3.0.py` verbatim: matplotlib settings, the "
        "palette, and the three tables the simulation wrote. Its figures are saved to "
        "`figures_3.0/` inside the working copy.",
        "The next five figures are the deck's, recomputed.")
    lines(FIG, "import gzip", "    df = pd.read_csv(f)")
    explain(
        "Draw one year of weekly beer-and-diaper margin.",
        "A rise in summer is expected every year: beer sells more when it is warm. To blame "
        "a rise on the shop, you need the sales that would have happened anyway — last "
        "year's curve, or shops that did not change.",
        "Sums the daily counts by week and prices them at an example margin of 6 DKK per "
        "beer and 20 DKK per pack of diapers.",
        "The curve peaks in summer by construction. One year of it cannot say whether a "
        "layout change in June did anything.")
    figure_lines("sales_curve", FIG, "# 1. One year of weekly revenue",
                 'fig.savefig(OUT / "fig_sales_curve.png")',
                 'show_png(OUT / "fig_sales_curve.png")', slides=["3.0 #5"],
                 treatment="exact: the deck's own script on the deck's own simulation")

    # --- slide 6 --------------------------------------------------------------------
    nb.md("### The chance of buying beer depends on who buys\n\n"
          "*Slide: \"The chance of buying beer depends on who buys, and that is a hypothesis\".*")
    explain(
        "Write the conditional probability the slide defines, and check the simulation "
        "against the guesses it was built on.",
        "A probability is a share of receipts; a conditional probability is the same share "
        "inside one group. The slide's table is the simulation's input, not a measurement.",
        "Displays the definition, then measures the share of card holders' receipts carrying "
        "beer in each segment and prints it beside the value the simulation was given. The "
        "measured shares differ slightly because the summer season multiplies the chance.",
        "The hypothesis in, the shares out: the simulation does what it says.")
    nb.equation("conditional_probability", '''
    beer, w = sp.symbols(r"\\mathrm{beer} w")
    P = sp.Function("P")
    formula(sp.Eq(P(sp.Symbol(r"\\mathrm{beer} \\mid w")),
                  P(sp.Symbol(r"\\mathrm{beer\\ and\\ } w")) / P(w), evaluate=False))
    carded = df.dropna(subset=["segment_if_card"])
    measured = carded.groupby("segment_if_card")["beer"].mean()
    print("chance a receipt carries beer — the slide's value (the simulation's input) "
          "and the share measured on card holders' receipts")
    for segment, given in P_BEER.items():
        print(f"  {segment:16} slide {given:.2f}   measured {measured[segment]:.3f}")
    ''', slides=["3.0 #6"])

    # --- slide 7 --------------------------------------------------------------------
    nb.md("### A receipt records what was bought and when, not who\n\n"
          "*Slide: \"A receipt records what was bought and when, not who bought it\".*")
    explain(
        "Draw the three daily counts a receipt file yields.",
        "Each day gives three counts — beer only, diapers only, both — and each count adds "
        "one yes or no per receipt. Only card holders say who they are.",
        "Runs block 2 of `make_figs_3.0.py` verbatim, then checks the card holders' share.",
        "The rare count, receipts with both, is the one the story is about, and the one "
        "a daily average treats worst.")
    figure_lines("daily_counts", FIG, "# 2. Three daily counts",
                 'fig.savefig(OUT / "fig_daily_counts.png")',
                 'show_png(OUT / "fig_daily_counts.png")\n'
                 'agrees("loyalty-card receipts, per cent of all", 100 * m["card_share"], 35, 0)',
                 slides=["3.0 #7"],
                 treatment="exact: the deck's own script on the deck's own simulation")

    # --- slide 8 --------------------------------------------------------------------
    nb.md("### Averaging inside windows gives a bell only when the windows are alike\n\n"
          "*Slide: \"Averaging inside windows gives a bell curve only when the windows are "
          "alike\".*")
    explain(
        "Show the daily beer count as two bells, and the daily share as one.",
        "The central limit theorem needs alike outcomes. Weekdays and weekends differ in "
        "footfall, so a daily count mixes two populations; dividing by the day's receipts "
        "removes that split, while the season still spreads the values.",
        "Runs block 3 of `make_figs_3.0.py` verbatim and prints the two means.",
        "Two bells, one share. And the theorem must not be applied to small counts such as "
        "the rare receipts with both items.")
    figure_lines("two_bells", FIG, "# 3. Daily beer count",
                 'fig.savefig(OUT / "fig_two_bells.png")',
                 'show_png(OUT / "fig_two_bells.png")\n'
                 'print(f"receipts with beer per day: weekdays {m[\'daily_beer_count_weekday_mean\']}, '
                 'weekends {m[\'daily_beer_count_weekend_mean\']} (means)")',
                 slides=["3.0 #8"],
                 treatment="exact: the deck's own script on the deck's own simulation")

    # --- slide 9 --------------------------------------------------------------------
    nb.md("### The window decides what can be seen\n\n"
          "*Slide: \"A daily window hides the 17:00 to 19:00 pattern that an hourly window "
          "shows\".*")
    explain(
        "Count the share of receipts with both items per hour, and per day.",
        "The documented finding was an hour-of-day pattern [@power2002]. A window must be "
        "shorter than the pattern it is meant to show.",
        "Runs block 4 of `make_figs_3.0.py` verbatim and prints the share in the evening "
        "hours against the rest.",
        "Per hour, 17 and 18 h stand out; per day, only the season is left.")
    figure_lines("hour_vs_day", FIG, "# 4. Hourly window",
                 'fig.savefig(OUT / "fig_hour_vs_day.png")',
                 'show_png(OUT / "fig_hour_vs_day.png")\n'
                 'print(f"share of receipts with both items: 17-18 h {100 * m[\'share_both_17_18\']:.2f} %, '
                 'other hours {100 * m[\'share_both_other_hours\']:.2f} %")',
                 slides=["3.0 #9"],
                 treatment="exact: the deck's own script on the deck's own simulation")

    # --- slides 10-11 ---------------------------------------------------------------
    nb.md("### Support, confidence and lift\n\n"
          "*Slides: \"Three numbers say how often two items share a receipt\" and \"Step by "
          "step: lift by hand from the year's two-by-two table of receipts\".*")
    explain(
        "Write the three association measures, and check that the two ways the deck writes "
        "lift are one function.",
        "Support and confidence are the measures of association-rule mining "
        "[@agrawal1993]; lift, which its authors called interest, divides the share with "
        "both by the share independence predicts [@brin1997].",
        "Builds each measure in sympy from the counts n, n_B, n_D and n_BD, and simplifies "
        "the difference between lift written with shares and lift written with counts.",
        "The difference is zero: the slide's two forms are the same number.")
    nb.equation("support_confidence_lift", '''
    n, nB, nD, nBD = sp.symbols("n n_B n_D n_BD", positive=True)
    support, confidence = nBD / n, nBD / nB
    lift_counts = nBD * n / (nB * nD)
    lift_shares = (nBD / n) / ((nB / n) * (nD / n))
    formula(sp.Eq(sp.Symbol(r"\\mathrm{support}"), support, evaluate=False),
            sp.Eq(sp.Symbol(r"\\mathrm{confidence}"), confidence, evaluate=False),
            sp.Eq(sp.Symbol(r"\\mathrm{lift}"), lift_counts, evaluate=False))
    print("lift as shares minus lift as counts, simplified:", sp.simplify(lift_shares - lift_counts))
    ''', slides=["3.0 #10"])
    explain(
        "Compute the three numbers by hand from the year's table, as the slide does.",
        "A library's answer can only be checked by someone who has computed it once.",
        "Reads the four cells of the two-by-two table from the measurements and recomputes "
        "support, confidence and lift, with the shares of beer and of diapers the slide "
        "rounds to 0.236 and 0.108.",
        "Lift 0.50: across all receipts, beer buyers carry diapers less often than chance.")
    nb.code('''
    table = m["all"]
    for what, key, stated in (("beer and diapers", "both", 8054), ("beer, no diapers", "beer_no_diapers", 140583),
                              ("diapers, no beer", "diapers_no_beer", 60125), ("neither", "neither", 421170),
                              ("with beer", "beer", 148637), ("with diapers", "diapers", 68179),
                              ("all receipts", "receipts", 629932)):
        agrees(f"receipts: {what}", table[key], stated, 0)
    n_all = table["receipts"]
    agrees("support = 8,054 / 629,932", table["both"] / n_all, 0.0128, 4)
    agrees("confidence = 8,054 / 148,637", table["both"] / table["beer"], 0.054, 3)
    agrees("share with beer", table["beer"] / n_all, 0.236, 3)
    agrees("share with diapers", table["diapers"] / n_all, 0.108, 3)
    agrees("lift", float(lift_counts.subs({n: n_all, nB: table["beer"], nD: table["diapers"],
                                           nBD: table["both"]})), 0.50, 2)
    ''')

    # --- slide 12 ------------------------------------------------------------------
    nb.md("### Simpson's paradox in the card holders' receipts\n\n"
          "*Slide: \"Beer buyers avoid diapers in the whole, and seek them inside every group\".*")
    explain(
        "Recompute the slide's table: the diaper share with and without beer, for parents, "
        "non-parents and all card holders.",
        "An association inside every group can reverse when the groups are pooled "
        "[@simpson1951]. Which table to believe is not a question the counts can answer; "
        "the causal story answers it — here, parenthood drives both purchases "
        "[@pearl2014].",
        "Builds the table from the measurements and checks every cell the slide prints.",
        "Lift 1.36 and 1.35 inside the groups, 0.56 pooled. Use the lift inside each group.")
    nb.code('''
    rows = []
    for label, key in (("Parents", "parents"), ("Non-parents", "non_parents"), ("All card holders", "card_holders")):
        g = m[key]
        rows.append({"group": label, "receipts": g["receipts"], "with beer": g["beer"],
                     "of which diapers": g["both"], "share": round(100 * g["diapers_share_given_beer"], 1),
                     "without beer": g["receipts"] - g["beer"], "of which diapers ": g["diapers_no_beer"],
                     "share ": round(100 * g["diapers_share_given_no_beer"], 1), "lift": g["lift"]})
    display(pd.DataFrame(rows).set_index("group"))
    for label, key, receipts, beer_, both_, share_b, share_nb, lift in (
            ("parents", "parents", 87578, 6928, 4089, 59.0, 42.2, 1.36),
            ("non-parents", "non_parents", 132287, 36551, 220, 0.6, 0.4, 1.35),
            ("all card holders", "card_holders", 219865, 43479, 4309, 9.9, 19.5, 0.56)):
        g = m[key]
        agrees(f"{label}: receipts", g["receipts"], receipts, 0)
        agrees(f"{label}: with beer", g["beer"], beer_, 0)
        agrees(f"{label}: with both", g["both"], both_, 0)
        agrees(f"{label}: diaper share with beer, per cent", 100 * g["both"] / g["beer"], share_b, 1)
        agrees(f"{label}: diaper share without beer, per cent",
               100 * g["diapers_no_beer"] / (g["receipts"] - g["beer"]), share_nb, 1)
        agrees(f"{label}: lift", g["lift"], lift, 2)
    ''')

    # --- slide 13 ------------------------------------------------------------------
    nb.md("### Reweighting to the population's shares\n\n"
          "*Slide: \"Reweighting to 24 % parents fixes the diaper rate: 10.8 %, not 17.6 %\".*")
    explain(
        "Write the post-stratified rate.",
        "Parents hold loyalty cards more often, so they are 39.8 % of card holders' receipts "
        "but 24 % of all shoppers. Any rate parents drive is inflated among card holders: "
        "selection bias. The card changes who you see, not how each group shops.",
        "Writes the rate as the sum over groups of the group's share times the group's rate, "
        "in sympy, and evaluates it with the card holders' shares and with the population's.",
        "17.6 % against 10.8 %: the same two group rates, recombined with different shares.")
    nb.equation("reweighting", '''
    share, rate_parent, rate_other = sp.symbols(r"\\pi \\rho_{\\mathrm{parent}} \\rho_{\\mathrm{other}}", positive=True)
    rate = share * rate_parent + (1 - share) * rate_other
    formula(sp.Eq(sp.Symbol(r"\\hat{\\rho}"), rate, evaluate=False))
    parents, others = m["parents"], m["non_parents"]
    r_parent = parents["diapers"] / parents["receipts"]
    r_other = others["diapers"] / others["receipts"]
    agrees("parents' diaper rate, per cent (38,147 / 87,578)", 100 * r_parent, 43.6, 1)
    agrees("non-parents' diaper rate, per cent (589 / 132,287)", 100 * r_other, 0.45, 2)
    agrees("parents' share of card holders' receipts", m["parents_share_of_card_holders"], 0.398, 3)
    uncorrected = float(rate.subs({share: 0.398, rate_parent: 0.436, rate_other: 0.0045}))
    corrected = float(rate.subs({share: 0.24, rate_parent: 0.436, rate_other: 0.0045}))
    agrees("uncorrected, card-holder shares, per cent", 100 * uncorrected, 17.6, 1)
    agrees("corrected, all-shopper shares, per cent", 100 * corrected, 10.8, 1)
    agrees("diaper receipts to stock for, per 10,000 receipts, corrected", 10000 * round(corrected, 3), 1080, 0)
    agrees("the same, uncorrected", 10000 * round(uncorrected, 3), 1760, 0)
    ''', slides=["3.0 #13"])
    explain(
        "Draw the reweighting as bars, and check it against the truth the simulation knows.",
        "The slide draws the three steps as boxes. The simulation can do what a shop cannot: "
        "it knows every receipt's diaper status, so the corrected rate can be compared with "
        "the rate over all receipts.",
        "Draws the two group rates, then the two recombined rates, and beside them the share "
        "of all 629,932 receipts that carry diapers.",
        "The corrected 10.8 % matches the whole-population share; the card holders' 17.6 % "
        "overstates it by more than 60 per cent.")
    nb.figure("reweighting", '''
    everyone = m["all"]["diapers"] / m["all"]["receipts"]
    labels = ["parents (card)", "non-parents (card)", "recombined with card<br>holders' shares, 39.8 %",
              "recombined with all<br>shoppers' shares, 24 %", "all receipts<br>(the simulation's truth)"]
    values = [100 * r_parent, 100 * r_other, 100 * uncorrected, 100 * corrected, 100 * everyone]
    fig = go.Figure(go.Bar(x=labels, y=values, marker_color=["#2A78D6", "#2A78D6", "#C0392B", "#2E8B57", "#52514E"],
                           text=[f"{v:.2f} %" for v in values], textposition="outside"))
    fig.update_layout(title="Post-stratification: the same group rates, recombined with the right shares",
                      yaxis_title="receipts carrying diapers, per cent", yaxis_range=[0, 52], showlegend=False)
    show(fig, "reweighting", height=460)
    agrees("share of all receipts carrying diapers, per cent", 100 * everyone, 10.8, 1)
    ''', slides=["3.0 #13"], treatment="lab data: the slide's three drawn steps as bars from "
                                       "the simulation, with its known truth beside them")

    # --- slide 14 ------------------------------------------------------------------
    nb.md("### A quantity the shop decides\n\n"
          "*Slide: \"Sales depend on who shops and on a shelf distance the shop decides\".*")
    explain(
        "Draw the two layouts, far and near, with the distance d between the shelves.",
        "d is set by the shop, not chosen by customers and not recorded on receipts. A "
        "quantity set by a decision is an intervention, written do(d) [@pearl2009]: "
        "receipts from layout A say nothing about layout B.",
        "Runs block 5 of `make_figs_3.0.py` verbatim.",
        "To learn P(both | w, do(d)) for the other layout, the other layout must be tried.")
    figure_lines("floor_plan", FIG, "# 5. Floor plan",
                 'fig.savefig(OUT / "fig_floor.png")',
                 'show_png(OUT / "fig_floor.png")', slides=["3.0 #14"],
                 treatment="exact: the deck's own drawing code (a conceptual diagram)")

    # --- slide 15 ------------------------------------------------------------------
    nb.md("### The expected margin of a chosen layout\n\n"
          "*Slide: \"Expected margin for a chosen layout sums over who buys and what they buy\".*")
    explain(
        "Write the expected weekly margin under a layout, and the shelf effect.",
        "The margin sums over shopper groups w and products k: every product is in the sum, "
        "so a sales loss in another aisle counts too.",
        "Builds E[M | do(d)] = N Σ_w P(w) Σ_k α_k P(k | w, do(d)) in sympy, then evaluates the "
        "slide's example: diapers only, N = 10,000 receipts per week, an assumed 20 DKK margin "
        "per pack, the corrected shares 24 % and 76 %, and the card holders' rates 43.6 % and "
        "0.45 %.",
        "21,612 DKK per week for one layout. The shelf effect is the difference between two "
        "such numbers, and only an experiment measures both.")
    nb.equation("expected_margin", '''
    N_, alpha, w_, k_, W, K_ = sp.symbols(r"N \\alpha w k W K")
    Pw = sp.Function("P")(w_)
    Pk = sp.Function("P")(sp.Symbol(r"k \\mid w, do(d)"))
    margin_formula = N_ * sp.Sum(Pw * sp.Sum(sp.Indexed(alpha, k_) * Pk, (k_, 1, K_)), (w_, 1, W))
    formula(sp.Eq(sp.Symbol(r"\\mathbb{E}[M \\mid do(d)]"), margin_formula, evaluate=False))
    expected_under = lambda layout: sp.Symbol(rf"\\mathbb{{E}}[M \\mid do(d_{{\\mathrm{{{layout}}}}})]")
    shelf_effect = sp.Add(expected_under("near"), -expected_under("far"), evaluate=False)
    formula(sp.Eq(sp.Symbol(r"\\Delta_{\\mathrm{shelf}}"), shelf_effect, evaluate=False))
    parent_part = sp.Rational(24, 100) * 20 * sp.Rational(436, 1000)
    other_part = sp.Rational(76, 100) * 20 * sp.Rational(45, 10000)
    agrees("parents' term, DKK per receipt", parent_part, 2.0928, 4)
    agrees("others' term, DKK per receipt", other_part, 0.0684, 4)
    agrees("margin per receipt, DKK", parent_part + other_part, 2.1612, 4)
    agrees("margin per week, 10,000 receipts, DKK", 10000 * (parent_part + other_part), 21612, 0)
    lay = m["layout_example"]
    beside("the simulation's own layout example (measured_3.0.json, not on a slide): margin per day, far and near",
           f"{lay['margin_far']:,} and {lay['margin_near']:,} DKK, gain {lay['gain']} DKK",
           "N = 10,000 per week",
           f"that example uses the simulation's {lay['N']:,} receipts a day, beer as well as diapers, "
           "and a hypothesised 10 % rise in parents' beer; the slide's example is diapers only")
    ''', slides=["3.0 #15"])

    # --- slides 16-17 --------------------------------------------------------------
    nb.md("""
    ### Testing a layout

    *Slides: "To test a layout, a random draw decides which stores get it" and "If stores
    cannot be drawn at random, use difference in differences".*

    **A/B test.** Choose the outcome first — total margin per receipt, not beer plus
    diapers; let chance decide which half of the stores gets layout B; run whole weeks;
    compare the two averages [@dreze1994]. Chance cannot favour busy stores, so the two
    groups differ only in the layout. Few stores, or shoppers crossing between groups,
    spoil the test [@kohavi2020].

    **Difference in differences.** When the shelves have already moved, or managers
    picked the stores, compare how much each group *changed* [@card1994]. Season and
    prices hit both groups and subtract out. The assumption — parallel trends — fails
    if another event, such as local roadworks, hits one group only; a flat gap before
    the move supports it, and only a random draw makes it hold by design. With one
    store only, intervention analysis [@box1975] or a synthetic control [@abadie2010]
    also rest on assumptions.
    """)
    explain(
        "Write the difference-in-differences estimator and evaluate the slide's example.",
        "Subtracting the unchanged stores' change removes whatever hit both groups.",
        "Builds the estimator in sympy and substitutes the slide's invented margins per "
        "receipt: 6.10 before and 6.40 after in the changed stores, 6.05 and 6.25 in the "
        "unchanged ones.",
        "0.10 DKK per receipt — an effect of the layout only if the trends were parallel.")
    nb.equation("did", '''
    mean_margin = lambda when, stores: sp.Symbol(rf"\\bar{{M}}^{{\\mathrm{{{when}}}}}_{{\\mathrm{{{stores}}}}}")
    Mca, Mcb = mean_margin("after", "changed"), mean_margin("before", "changed")
    Mua, Mub = mean_margin("after", "unchanged"), mean_margin("before", "unchanged")
    did = sp.Add(sp.Add(Mca, -Mcb, evaluate=False),                   # unevaluated, so the two
                 sp.Mul(-1, sp.Add(Mua, -Mub, evaluate=False), evaluate=False),   # changes stay visible
                 evaluate=False)
    formula(sp.Eq(sp.Symbol(r"\\mathrm{effect}"), did, evaluate=False))
    example = did.subs({Mca: sp.Rational(640, 100), Mcb: sp.Rational(610, 100),
                        Mua: sp.Rational(625, 100), Mub: sp.Rational(605, 100)})
    agrees("(6.40 - 6.10) - (6.25 - 6.05), DKK per receipt", example, 0.10, 2)
    ''', slides=["3.0 #17"])

    nb.md("""
    ### Transport, and a definition

    *Slides: "A beer-and-diapers rule from one country may not hold in another" and
    "Data science speaks mathematics and answers for the decisions it drives".*

    A rule describes one population: one chain, one country, one period. Behaviour
    measured in one society often differs in another, and the samples behavioural
    science draws on are unusually narrow [@henrich2010]. A rule transfers only if each
    shopper group buys each product at the same rate in both places — transportability,
    or external validity [@pearlbareinboim2014] — and once a store acts on a rule,
    behaviour can shift [@lucas1976]. Before using it abroad: rerun the analysis on
    local receipts, then test the layout locally.

    The deck closes on a definition, in the instructor's words: data mining finds
    patterns in data collected for other purposes [@hand2001]; data science carries
    them from a specific question to a decision, including the test that shows they
    hold [@tukey1962; @donoho2017]; and because its models often act on people directly,
    at scale, with no person in between, it answers for the decisions it drives
    [@aiact2024, Art. 14]. Module 3.1 takes a model from a pattern to a service whose
    decisions can be checked.
    """)
