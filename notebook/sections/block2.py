"""Block two — checking every request before the model sees it; Laboratory 2
(Module3.1.pptx slides 44-56)."""

S = "Module 3/exercises/solutions"
LABS = "Module 3/exercises/labs"

from sections.block1 import solution_paragraphs  # noqa: E402


def build(nb, explain) -> None:
    nb.md("""
    ---
    ## Block two — Checking every request before the model sees it

    *Slide: "A model cannot object, so something in front of it has to".* Module 1 wrote
    down each measurement's unit, source, valid range and owner. That document had no
    power: nothing stopped a request from breaking it. A model will accept a walking
    speed of nine hundred metres per second and answer. A contract is the same
    declaration, checked before the model is asked [@breck2019].
    """)
    explain(
        "Write the contract's acceptance rule.",
        "Present and null are two rules, not one: a missing *field* is a request the "
        "service cannot answer — the model's door is four columns wide — while a null "
        "*value* is a measurement that was not made, which the stored median fills.",
        "Writes the three clauses of the definition card in sympy, one field f at a time, "
        "with acceptance as a product of indicators over the fields.",
        "A request is accepted only when every field passes all three clauses.")
    nb.equation("contract", '''
    f_, F_ = sp.symbols("f F", integer=True, positive=True)
    x_f = sp.Symbol("x[f]")
    null = sp.Symbol(r"\\mathrm{null}")
    field = lambda name: sp.Symbol(rf"\\mathrm{{{name}}}_f")
    clauses = sp.And(sp.Implies(field("required"), sp.Symbol(r"f \\in x")),
                     sp.Implies(sp.Eq(x_f, null), field("nullable")),
                     sp.Implies(sp.Ne(x_f, null),
                                sp.And(sp.Eq(sp.Function(r"\\mathrm{type}")(x_f), field("type")),
                                       sp.Le(sp.Symbol(r"\\min_f"), x_f), sp.Le(x_f, sp.Symbol(r"\\max_f")))))
    # "for every field f" as a product of indicators, since sympy has no quantifier
    formula(sp.Eq(sp.Function(r"\\mathbf{1}")(sp.Function(r"\\mathrm{accept}")(sp.Symbol("x"))),
                  sp.Product(sp.Function(r"\\mathbf{1}")(sp.Symbol(r"\\mathrm{ok}_f")), (f_, 1, F_))))
    formula(sp.Equivalent(sp.Symbol(r"\\mathrm{ok}_f"), clauses))
    ''', slides=["3.1 #46"])
    explain(
        "Define the contract as Lab 2 declares it.",
        "The contract is a Python dictionary, so the door, the check and the appendix table "
        "all read the same rules.",
        "Copies the lab number, the measured speed bounds, the margin, the nullable fields "
        "and `CONTRACT` from `solutions/lab_02.py`, verbatim.",
        "Four fields, five rules each. A contract must not bound a field at the measured "
        "extreme with no margin.")
    nb.source(f"{S}/lab_02.py", "LAB", "SPEED_MEASURED_MIN", "SPEED_MEASURED_MAX", "SPEED_MARGIN",
              "NULLABLE_BY_MEASUREMENT", "CONTRACT", cite="[@breck2019]")
    explain(
        "Check the contract's numbers against the measurements they come from.",
        "The numbers in a contract are measurements, not guesses. Speed is present on every "
        "training row, so an empty speed is refused; a signal strength is present only when "
        "its beacon was in range, so an empty one is allowed. Speed's bounds are Module 1's "
        "measured range for this vehicle, widened by one metre per second each way.",
        "Measures the share of the training day's rows on which each field carries a value, "
        "reads Module 1's measured speed range, and compares both with the contract and the "
        "slide.",
        "100 per cent for speed, 11.3 to 30.7 per cent for the signal strengths; −4.361 to "
        "4.555 metres per second accepted.")
    nb.code('''
    training_day = build_table()
    for field, stated in (("speed", 100.0), ("rssi1", 11.3), ("rssi2", 25.1), ("rssiC", 30.7)):
        agrees(f"rows on which {field} carries a value, per cent", 100 * training_day[field].notna().mean(), stated, 1)
    module1 = json.loads((COURSE / "Module 1" / "slides" / "measured.json").read_text())
    low, high = module1["speed_range_m_per_s"]["value"]
    agrees("Module 1's measured minimum speed", low, SPEED_MEASURED_MIN, 3)
    agrees("Module 1's measured maximum speed", high, SPEED_MEASURED_MAX, 3)
    agrees("contract minimum, m/s", CONTRACT["speed"]["min"], -4.361, 3)
    agrees("contract maximum, m/s", CONTRACT["speed"]["max"], 4.555, 3)
    print("nullable:", {field: rule["nullable"] for field, rule in CONTRACT.items()})
    ''')

    nb.md("### The rules, and the check at the door\n\n"
          "*Slides: \"A field carries five rules, and each rule has a plain meaning\" and \"The "
          "check at the door runs in four steps, and never asks the model early\".* Presence, "
          "emptiness, type, range, unit. For each required field confirm it is present; for "
          "each present value confirm type and range; if any note was written, refuse with "
          "every note; only otherwise pass the request on.")
    explain(
        "Define the check at the door as the Lab 2 solution writes it.",
        "A refusal that does not name its field is one the caller will retry unchanged. And "
        "in Python `True` is an instance of `int`, so a naive type check lets "
        "`\"speed\": true` through as one metre per second.",
        "Copies `_is_number` and `validate` from `solutions/lab_02.py`, verbatim.",
        "Every complaint names its field and the rule it broke.")
    nb.source(f"{S}/lab_02.py", "_is_number", "validate", cite="[@breck2019]")
    explain(
        "Ask the check about three requests: an impossible speed, a speed sent as `true`, "
        "and a valid request.",
        "The two failures are the ones a naive check lets through: a number far outside the "
        "measured range, and a boolean that Python counts as an integer.",
        "Defines the valid request used from here to the end of the notebook — speed 1.2 "
        "and three signal strengths — and prints `validate()` for each case.",
        "One complaint each for the first two, both naming `speed`; none for the valid "
        "request.")
    nb.code('''
    GOOD_REQUEST = {"speed": 1.2, "rssi1": -70.0, "rssi2": -80.0, "rssiC": -75.0}
    print(validate({**GOOD_REQUEST, "speed": 900.0}))
    print(validate({**GOOD_REQUEST, "speed": True}))
    print(validate(GOOD_REQUEST), "<- the valid request draws no complaint")
    ''')

    nb.md("""
    ### Status codes

    *Slides: "Definition — the status code" and "The four status codes this service
    returns, and who acts on each"* [@fielding2022].

    | Code | What it means | When this service returns it | What the caller then does |
    |---|---|---|---|
    | 200 | Accepted and answered | every rule in the contract was satisfied | reads the probability, decision, cutoff and version |
    | 422 | Well formed, contents unacceptable (§ 15.5.21) | a speed of nine hundred; a missing field; text where a number was declared | corrects the values and sends again |
    | 400 | The request itself was malformed (§ 15.5.1) | the message could not be read at all | corrects the program that builds the message |
    | 500 | The fault is ours (§ 15.6.1) | the model file failed to load; the registry could not be reached | escalates — a person is woken for this |

    A refusal must not be reported as 500, because 500 wakes an engineer for a caller's
    mistake.
    """)
    explain(
        "Write refusal and provenance.",
        "Validation happens before the model is asked, because a model given nonsense returns "
        "a number, not an objection. And the answer carries the version and the cutoff that "
        "produced it, so a complaint months later is settled by reading the answer itself "
        "[@fielding2022; @sculley2015].",
        "Writes the two statements of the definition card in sympy, with the complaints "
        "counted: none, or at least one.",
        "A refusal never reaches the model; an answer always carries its provenance.")
    nb.equation("refusal", '''
    complaints = sp.Symbol(r"|\\mathrm{complaints}|")
    status, calls = sp.Symbol(r"\\mathrm{status}"), sp.Symbol(r"\\mathrm{model\\ calls}")
    answer = sp.Symbol(r"\\mathrm{answer}")
    provenance = sp.Tuple(sp.Symbol("p"), sp.Symbol(r"\\mathrm{decision}"), sp.Symbol(r"\\mathrm{version}"), sp.Symbol("t"))
    formula(sp.Implies(sp.Gt(complaints, 0), sp.And(sp.Eq(status, 422), sp.Eq(calls, 0))))
    formula(sp.Implies(sp.Eq(complaints, 0), sp.And(sp.Eq(status, 200), sp.Eq(answer, provenance))))
    ''', slides=["3.1 #52"])
    explain(
        "Define the response as the Lab 2 solution writes it.",
        "Validate first, and only then ask the model; answer with the version and the cutoff "
        "that produced the decision [@fielding2022; @sculley2015].",
        "Copies `respond` from `solutions/lab_02.py`, verbatim.",
        "422 with the complaints, or 200 with probability, decision, version and threshold.")
    nb.source(f"{S}/lab_02.py", "respond", cite="[@fielding2022]")
    explain(
        "Answer one valid request, in the five steps of the slide.",
        "Check at the door, prepare with the stored constants, ask the model, compare with "
        "the cutoff, answer with provenance. Every classifier library uses 0.5 unless told "
        "otherwise; here the cutoff is 0.2, because a missed passenger costs four times a "
        "wasted trip.",
        "Loads the approved artefact, answers the request at 0.2, and again at 0.5.",
        "The same probability, 0.458, is aboard at 0.2 and not aboard at 0.5. Part 3.2 follows "
        "this request through the whole day.")
    nb.code('''
    SERVED = load_artefact("v1")
    SERVED["version"] = "v1"
    at_priced = respond(GOOD_REQUEST, CONTRACT, SERVED, 0.2)
    at_half = respond(GOOD_REQUEST, CONTRACT, SERVED, 0.5)
    print(at_priced)
    print(at_half)
    print(respond({**GOOD_REQUEST, "speed": 900.0}, CONTRACT, SERVED, 0.2))
    ''')
    nb.md("""
    ### Changing the contract, and who owns it

    *Slides: "Changing the contract: which changes are safe, and which one is a trap" and
    "Most failures of a model in use are organisational, not mathematical".* A safe
    change lets every old request still work (add an optional field, widen a range); a
    breaking change makes some fail at the door, and the caller gets a 422 and knows. The
    trap is a change that keeps a field's name and changes its meaning: the door accepts
    it, the model reads it, every answer is wrong and no error is raised. A breaking
    change therefore gets a new version of the service beside the old one. And a service
    has an owner — a named person and a deputy — and an agreement promising response time
    for the slowest requests, not the average; undeclared consumers — systems silently
    using a model's output — are a debt of their own [@sculley2015, § 2].
    """)

    # --- the laboratory ------------------------------------------------------------------------
    nb.md("""
    ## Laboratory 2 — The contract at the door

    *Slide: "Lab 2 — The contract at the door".* Write `validate` and `respond`. Eight
    impossible requests must each draw a refusal that names its own field; two must be
    accepted — the valid one, and one whose signal strength is empty. A counting stand-in
    replaces the model: ask it about an invalid request and fail. Twenty-five minutes.
    """)
    nb.statement(f"{LABS}/02_the_contract.py")
    nb.md("### The solution, step by step\n\n"
          "The functions were defined above. First the lab file's own demonstration.")
    explain(
        "Run the stub's own `__main__` block against the solved functions.",
        "It is what a student sees when the file is complete.",
        "The lines below are the stub's demonstration, verbatim (the stub's default cutoff "
        "is 0.5).",
        "One answer with provenance, one refusal naming speed.")
    nb.step(f"{LABS}/02_the_contract.py", 0)
    explain(
        "Let the solution's paths resolve in the working copy.",
        "The solution reads the vehicle slice relative to its own file.",
        "Points `__file__` at `solutions/lab_02.py` in the working copy.",
        "The next cells are `python3 solutions/lab_02.py`, one paragraph of its `__main__` "
        "block per cell.")
    nb.code('__file__ = str(WORK / "solutions" / "lab_02.py")')
    solution_paragraphs(nb, explain, f"{S}/lab_02.py", [
        ("Import what the demonstration needs beyond the functions above.",
         "The block tabulates with pandas, draws with plotly and narrates with `_narrate`.",
         "Imports pandas, plotly and the narration helpers the set-up registered.",
         "Nothing runs yet."),
        ("Start the narration.",
         "Every line the solution prints goes through `say`, stamped with the seconds since "
         "the notebook began.",
         "Creates the narrator for Lab 2 and prints the demonstration's one-line summary.",
         "The timings differ from run to run; nothing else does."),
        ("Load the approved model and state where the speed bounds come from.",
         "The contract's numbers must be traceable to a measurement.",
         "Loads v1's artefact, tags it with its version, and prints the contract's speed "
         "bounds beside Module 1's measured range and the margin.",
         "−4.361 to 4.555 metres per second: the measured range widened by one each way."),
        ("Write down one valid request and eight impossible ones.",
         "Each impossible request breaks exactly one rule, so each refusal can be checked "
         "for naming the right field.",
         "Builds the valid request and eight variants: a required field missing, a declared "
         "column absent, a None speed, a number as text, a boolean as a number, a speed above "
         "the maximum, a signal strength above zero and one below the minimum.",
         "Nothing is sent yet."),
        ("Send the eight, and then the two edge cases that must be accepted.",
         "A refusal must be 422 and name its field, the model must not be asked, and a null "
         "value must not be confused with an absent field.",
         "Answers each impossible request at the cutoff 0.2 and tabulates status, number of "
         "complaints and the first complaint; then compares an absent `rssi1` with a null "
         "`rssi1` and answers the null one.",
         "Eight 422s, each naming its field; the absent `rssi1` refused and the null one "
         "answered from the stored median."),
        ("Answer the valid request.",
         "The answer must carry its own provenance, so a complaint months later is settled "
         "by reading it.",
         "Answers the valid request at 0.2 and prints probability, decision, version and "
         "threshold, then states the boolean trap.",
         "Aboard at 0.2, from version v1."),
        ("Draw the contract's speed band against the speeds the vehicle recorded.",
         "A bound is only credible beside the data it was drawn from.",
         "Reads the vehicle slice relative to the solution's file, draws a histogram of its "
         "speeds with the measured and the contract bounds, and marks the refused 900.",
         "Every recorded speed lies inside the band; 900 is far outside it."),
        ("Say what the check grades.",
         "A student should know what a pass means before running the check.",
         "Prints the solution's one-line summary of the check's requirements.",
         "Nothing is computed here."),
    ])
