"""The closing slides of 3.2, the two remaining appendices, and the practice questions
with their answers."""


def build(nb, explain) -> None:
    nb.md("""
    ## Two articles of the AI Act, built for other reasons

    *Slides: "Two Artificial Intelligence Act articles were built today, for other
    reasons" and "Compliance is not claimed, and the duty is deferred to December 2027".*

    - **Article 12, record-keeping** — events recorded automatically over the lifetime
      of the system, so its working can be traced [@aiact2024, Art. 12]. Built today: the
      training record of Block one and the history that is only ever added to.
    - **Article 14, human oversight** — a person can oversee the system, understand its
      answer, and step in or stop it [@aiact2024, Art. 14]. Built today: the approved
      entry and the rollback, rehearsed at every ordinary release.
    - **Supporting both** — the version and the cutoff carried in the body of every
      answer.

    Whether this service is high-risk depends on what it is used for, not on how it is
    built; Annex III lists the uses that carry the duty. The Annex III duties were to
    apply from 2 August 2026 and were deferred to 2 December 2027 by the Digital Omnibus
    [@omnibus2026]. The deferral changes when the duties apply, not what they require.
    *This is the text of the Regulation, not legal advice.*

    ## What you can now do

    *Slide: "What you can now do".* Release a model by changing one entry, and roll back
    with the same move. Refuse an impossible request before the model is asked, and name
    the field. Derive a cutoff from the two prices, and defend it against one half. Cause
    training–serving skew, see that nothing reports an error, and then cure it. Required
    reading: {@sculley2015}.
    """)

    # --- appendices -------------------------------------------------------------------------------
    nb.md("---\n# Appendices\n\nThe decks' three appendices (identical in 3.1 and 3.2). The "
          "first, on calibration, was worked at the end of Block three.")
    nb.md("### Appendix — where z comes from, and when one tail is enough\n\n"
          "*Slides: \"Appendix — where z comes from, and when one tail is enough\" (3.1 and 3.2).*")
    explain(
        "Derive z as a quantile of the standard normal, and show that a two-sided 95 per cent "
        "interval is a one-sided 97.5 per cent bound.",
        "Repeat a measurement many times and the results pile up in a bell around the true "
        "value; z counts standard errors from the centre. Choose the coverage first, then "
        "read z off the curve [@wilson1927; @brown2001].",
        "Writes the margin and the quantile definition in sympy, and evaluates the normal "
        "quantile function at the levels the appendix names.",
        "1.645 at 90 per cent two-sided (or 95 one-sided), 1.96 at 95, 2.576 at 99. Fix the "
        "direction before the scores are seen: choosing after the fact is a two-sided error "
        "rate in disguise.")
    nb.equation("z_quantiles", '''
    alpha_ = sp.Symbol(r"\\alpha", positive=True)
    Phi_inverse = sp.Function(r"\\Phi^{-1}")
    formula(sp.Eq(sp.Symbol(r"\\mathrm{margin}"), sp.Symbol("z") * sp.Symbol(r"\\mathrm{SE}")),
            sp.Eq(sp.Symbol(r"z_{\\text{two-sided}}"), Phi_inverse(1 - alpha_ / 2)),
            sp.Eq(sp.Symbol(r"z_{\\text{one-sided}}"), Phi_inverse(1 - alpha_)))
    agrees("90 per cent, two-sided", stats.norm.ppf(0.95), 1.645, 3)
    agrees("95 per cent, two-sided", stats.norm.ppf(0.975), 1.96, 2)
    agrees("99 per cent, two-sided", stats.norm.ppf(0.995), 2.576, 3)
    agrees("95 per cent, one-sided", stats.norm.ppf(0.95), 1.645, 3)
    print("a two-sided 95 % interval is a one-sided 97.5 % bound: z =", round(stats.norm.ppf(0.975), 3))
    ''', slides=["3.1 #85", "3.2 #37"])
    nb.md("### Appendix — the input contract in full\n\n"
          "*Slides: \"Appendix — the input contract in full\" (3.1 and 3.2).* Twenty rules across "
          "four fields; every refusal in Lab 2 names exactly one cell of this table.")
    explain(
        "Print the contract as the appendix tabulates it, from the dictionary Lab 2 enforces.",
        "A table typed onto a slide can drift from the code; one printed from the code cannot.",
        "Builds the table from `CONTRACT`.",
        "Speed present and never empty, −4.361 to 4.555 metres per second; the three signal "
        "strengths present, possibly empty, −120 to 0 decibel-milliwatts.")
    nb.code('''
    display(pd.DataFrame([{"field": field, "must be present": "yes" if rule["required"] else "no",
                           "may be empty": "yes" if rule["nullable"] else "no", "type": rule["type"],
                           "accepted range": f"{rule['min']:g} to {rule['max']:g}", "unit": rule["units"]}
                          for field, rule in CONTRACT.items()]).set_index("field"))
    ''')

    # --- practice -----------------------------------------------------------------------------------
    nb.md("""
    ---
    ## Practice

    1. **Would a cost-of-serving gate change the decision?** Add a second rule to
       `promote_if_better`: refuse a candidate that does more than twice the approved
       model's work per request. Success criterion: v2 is refused, and your reason names
       both numbers. (Work, not wall-clock latency, so that the answer is the same on
       every machine.)
    2. **How wrong can the prices be?** Sweep the cost ratio C_FN : C_FP over 1, 2, 3, 4,
       8 and 20, derive the cutoff for each, and compare its realised cost with one half's.
       Success criterion: you can name the ratios where the derivation loses, and say why
       — look at the reliability diagram again. (This is the hidden stress-test slide of
       3.1.)
    3. **Which column swap hurts most?** Swap each pair of the four features in turn and
       measure the accuracy at the priced cutoff. Success criterion: a table of six pairs
       ordered by damage, and one sentence on why the worst pair is the worst.

    Answers below — try first.
    """)
    explain(
        "Leave room for your own attempt.",
        "The answers follow directly below; an attempt made first is worth more than one "
        "made after reading them.",
        "Nothing: an empty cell with a comment.",
        "Write your workings here, with the functions defined above.")
    nb.code("# Your workings here.")
    nb.md("### Answers")
    explain(
        "Answer the three questions with the functions defined above.",
        "Each answer is a measurement, so it is computed rather than asserted.",
        "1: compares the two versions' decision nodes per request. 2: for each ratio, prices "
        "the derived cutoff and one half on the test rows — a tie is not a win, exactly as in "
        "the gate. 3: swaps each pair of prepared columns and measures accuracy at 0.2.",
        "v2 does 17.7 times the work and is refused. The derivation wins at three ratios, ties "
        "at one and loses at two — at the ratio 2 it costs 1,538 where one half costs 1,078 — "
        "because the model's probabilities are inflated and the derived cutoff trusts them. "
        "The worst swap puts a signal strength where speed belongs: speed is the feature the "
        "forest leans on most.")
    nb.code('''
    # 1. A cost-of-serving gate refuses v2 as well.
    work = {v: WORK_PER_REQUEST[v]["decision_nodes_per_request"] for v in ("v1", "v2")}
    print(f"v1 {work['v1']} nodes per request, v2 {work['v2']}: "
          f"{'refused' if work['v2'] > 2 * work['v1'] else 'allowed'} by a two-times work rule")

    # 2. The derived cutoff against one half, at six prices of a miss.
    recorded = json.loads((SLIDES / "measured.json").read_text())["derived_against_habit"]["value"]
    outcome = {"win": 0, "tie": 0, "loss": 0}
    print("\\nratio   cutoff   cost derived   cost at 0.5   result")
    for ratio in (1, 2, 3, 4, 8, 20):
        derived = threshold_from_costs(1.0, ratio)
        mine, habit = (realised_cost(P_TEST, Y_TEST, t, 1.0, ratio) for t in (derived, 0.5))
        result = "win" if mine < habit else ("tie" if mine == habit else "loss")
        outcome[result] += 1
        print(f"1:{ratio:<5} {derived:7.3f} {mine:14,.0f} {habit:13,.0f}   {result}")
        assert (mine, habit) == (recorded[f"1:{ratio}"]["cost_derived"], recorded[f"1:{ratio}"]["cost_at_half"])
    print("wins, ties, losses:", outcome, "- as slides/measured.json records them")

    # 3. The six swaps, ordered by the accuracy left.
    damage = []
    for first in range(len(FEATURES)):
        for second in range(first + 1, len(FEATURES)):
            variant = X_TEST.copy()
            variant[:, [first, second]] = variant[:, [second, first]]
            accuracy = float(((ARTEFACT_V1["model"].predict_proba(variant)[:, 1] >= PRICED).astype(int) == Y_TEST).mean())
            damage.append((f"{FEATURES[first]} <-> {FEATURES[second]}", round(accuracy, 4)))
    print()
    display(pd.DataFrame(sorted(damage, key=lambda item: item[1]), columns=["swapped pair", "accuracy at 0.2"])
            .set_index("swapped pair"))
    print(f"no swap: {((P_TEST >= PRICED).astype(int) == Y_TEST).mean():.4f}")
    print("importance of each feature in the forest:",
          {name: round(float(value), 3) for name, value in zip(FEATURES, ARTEFACT_V1["model"].feature_importances_)})
    ''')
    nb.md("""
    ### On the stand-in

    Everything in Parts 3.1 and 3.2 ran against two forests trained in seconds from the
    generated table. That is deliberate, and it is the argument for release by
    indirection: the service, the contract, the cutoff and the skew are indifferent to
    which model sits behind the name. When the modelling sessions hand over an artefact,
    it goes into `service/artefacts/`, the pointer and the alias move, and every number
    in this notebook changes while not one line of it does.
    """)
    explain(
        "Remove the working copy.",
        "It holds two trained forests, two MLflow stores and a 95-megabyte candidate three "
        "times over — several hundred megabytes nobody needs once the notebook has run.",
        "Moves back to `exercises/` and deletes the temporary folder made at the start.",
        "Your `exercises/` folder is as you left it, and nothing is left behind.")
    nb.code('''
    os.chdir(SOURCE)
    shutil.rmtree(WORK.parent)
    print("working copy removed:", not WORK.exists())
    ''')
