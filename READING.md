# Module 3 — Reading, and the exam question

## Required, before Module 4

**Sculley, D. et al. (2015). *Hidden Technical Debt in Machine Learning Systems*.
NeurIPS 28.**
<https://papers.neurips.cc/paper/5656-hidden-technical-debt-in-machine-learning-systems.pdf>

Free, nine pages. Its central claim is one diagram: the box marked "ML code" is
small, and everything around it — configuration, data collection, serving
infrastructure, monitoring — is where systems actually fail. Read it with today's
four labs in mind and count how many of its named debts you built a defence
against.

## Recommended — the record, and the registry

**Zaharia, M., Chen, A., Davidson, A., Ghodsi, A., Hong, S. A., Konwinski, A.,
Murching, S., Nykodym, T., Ogilvie, P., Parkhe, M., Xie, F. & Zumar, C. (2018).
*Accelerating the Machine Learning Lifecycle with MLflow*. IEEE Data Engineering
Bulletin 41(4), 39–45.**
<https://sites.computer.org/debull/A18dec/p39.pdf> — free, seven pages, by the
people who built the tool Lab 1 part (b) uses. Read it for the three ideas the
module teaches — a run is a record, a signature is a schema, a release is an
alias move — stated by their authors before the product grew around them.

**Schelter, S., Biessmann, F., Januschowski, T., Salinas, D., Seufert, S. &
Szarvas, G. (2018). *On Challenges in Machine Learning Model Management*. IEEE
Data Engineering Bulletin 41(4), 5–15.**
<https://sites.computer.org/debull/A18dec/p5.pdf> — free, in the same issue.
The other half of block one: what a team actually has to keep about a model, and
why "the notebook is on my laptop" stops working at two people.

**Breck, E., Polyzotis, N., Roy, S., Whang, S. E. & Zinkevich, M. (2019). *Data
Validation for Machine Learning*. Proceedings of Machine Learning and Systems
(MLSys) 1.** Free through the proceedings. Lab 2's contract and Lab 4's skew are
both here, from a team who ran the validation layer of a very large production
system: what a schema catches, what it does not, and what training–serving skew
costs when nobody is looking for it.

## Recommended — the decision, and its price


**Elkan, C. (2001). *The Foundations of Cost-Sensitive Learning*. IJCAI,
pages 973–978.**
<https://cseweb.ucsd.edu/~elkan/rescale.pdf> — free. Six pages containing the
threshold derivation from Lab 3, done properly, including the part this module
skips: how to rescale a training set when you cannot change the threshold.

**Provost, F. & Fawcett, T. (2013). *Data Science for Business*, chapters 7–8.
O'Reilly.** Through the AAU library. Expected value as the organising idea for
evaluating a model, written for people who have to justify a decision.

**Niculescu-Mizil, A. & Caruana, R. (2005). *Predicting Good Probabilities with
Supervised Learning*. ICML, 625–632.**
<https://doi.org/10.1145/1102351.1102430> — through the AAU library. The
reliability diagram on the slides, and the two repairs the module does not
apply: Platt scaling and isotonic regression. Read it to know what you would do
if you needed the probabilities themselves rather than their order.

**DeGroot, M. H. & Fienberg, S. E. (1983). *The Comparison and Evaluation of
Forecasters*. The Statistician 32(1/2), 12–22.**
<https://doi.org/10.2307/2987588> — through the library. Where calibration is
defined properly, and where a forecaster's score is split into calibration and
refinement. Short, and worth it for the idea that being right and being honest
are two different virtues.

**Arrow, K. J., Harris, T. & Marschak, J. (1951). *Optimal Inventory Policy*.
Econometrica 19(3), 250–272.** <https://doi.org/10.2307/1906813> — through the
library. The newsvendor problem, and where the critical fractile on the
departure slide comes from. Read section 1 only: the same ratio as Lab 3's
threshold, arrived at from stock rather than from classification.

**Jensen, J. L. W. V. (1906). *Sur les fonctions convexes et les inégalités entre
les valeurs moyennes*. Acta Mathematica 30, 175–193.**
<https://doi.org/10.1007/BF02418571> — free, and in French. Read it through
Module 4, which proves the inequality on the same twelve-seat departure this
module measures.

## Recommended — speed, and the standard

**Dean, J. & Barroso, L. A. (2013). *The Tail at Scale*. Communications of the
ACM 56(2), 74–80.** <https://doi.org/10.1145/2408776.2408794> — free from the
authors. Six pages on why the 95th percentile and not the mean, from the people
who had to make a service out of a thousand machines. The argument Lab 4 reports
in one number.

**Hyndman, R. J. & Fan, Y. (1996). *Sample Quantiles in Statistical Packages*.
The American Statistician 50(4), 361–365.**
<https://doi.org/10.1080/00031305.1996.10473566> — through the library. Nine
definitions of a sample quantile, numbered; the packages disagree with each
other by default. Lab 4 grades definition 1, nearest rank, because for latency
the honest answer is a duration that happened.

**Fielding, R., Nottingham, M. & Reschke, J. (2022). *HTTP Semantics*. Request
for Comments 9110, Internet Engineering Task Force.**
<https://www.rfc-editor.org/rfc/rfc9110> — free. Read §15.5.1 (400), §15.5.21
(422) and §15.6.1 (500) — three short paragraphs that settle the argument Lab 2
has about which code a refusal deserves.

**Kreuzberger, D., Kühl, N. & Hirschl, S. (2023). *MLOps: Overview, Definition,
and Architecture*. IEEE Access 11, 31866–31879.**
<https://doi.org/10.1109/ACCESS.2023.3262138> — free. Optional, and the widest
view: the practices of this module named and arranged as an architecture. Useful
for vocabulary when you meet a team that has all of it and calls it something
else.

**Huyen, C. (2022). *Designing Machine Learning Systems*, chapter 7 "Model
Deployment and Prediction Service". O'Reilly.** Through the library.

**MLflow documentation — Model Registry workflow.**
<https://mlflow.org/docs/latest/ml/model-registry/workflow/> — the industry tool
for what you built in fifty lines in Lab 1. Read it to recognise the same ideas
under different names.

**MLflow documentation — the scikit-learn flavour, and what it writes.**
<https://mlflow.org/docs/latest/api_reference/python_api/mlflow.sklearn.html> —
read `log_model`, and in particular `serialization_format`. This module pins
**MLflow 3.15.1**, and both of its logging sites name `cloudpickle` rather than
taking the default. MLflow 3 made `skops` the default, and `skops` refuses to
write a class it does not recognise — here `StoredTransform` in
`service/models.py`, which is the step that carries the fitted constants with
the model. The parameter means the same thing in MLflow 2 and 3, so naming it
is the fix; pinning the old major version would only postpone it. This is worth
five minutes because it is the ordinary shape of a platform upgrade: a default
changed, and the code that relied on the default stopped working.

**Armbrust, M. et al. (2021). *Lakehouse*. CIDR.**
<https://www.cidrdb.org/cidr2021/papers/cidr2021_paper17.pdf> — free, for the
platform comparison in block four.

> Nothing licensed is redistributed in this repository.


**Cortes-Peña, Y. R., Kumar, D., Singh, V. & Guest, J. S. (2020). *BioSTEAM: a
fast and flexible platform for the design, simulation and techno-economic
analysis of biorefineries under uncertainty*. ACS Sustainable Chemistry &
Engineering 8(8), 3302–3310.** <https://doi.org/10.1021/acssuschemeng.9b07040> —
free. Optional. Lab 3 priced a threshold from a cost ratio somebody gave you.
This is what the layer underneath looks like when the costs are themselves
modelled, and it is the answer to "where do the prices come from" — they come
from an engineering model that has its own uncertainty.

**Mowbray, M., Savage, T., Wu, C., Song, Z., Cho, B. A., Del Rio-Chanona, E. A. &
Zhang, D. (2021). *Machine learning for biochemical engineering: a review*.
Biochemical Engineering Journal 172, 108054.**
<https://doi.org/10.1016/j.bej.2021.108054> — through the AAU library. Optional,
and useful mainly as the outside view: what your tooling looks like to a
discipline that had physical models first and adopted yours second.

## The European Union Artificial Intelligence Act, two articles

**Regulation (EU) 2024/1689 laying down harmonised rules on artificial
intelligence (the Artificial Intelligence Act), Article 12 and Article 14.**
<http://data.europa.eu/eli/reg/2024/1689/oj> — free, and read only those two
articles: they are a page each. Article 12 requires a high-risk system to allow
the automatic recording of events over its lifetime so that its functioning is
traceable; Article 14 requires that a person can oversee it, including the
ability to intervene in it or to interrupt it. Read them after block one and
notice that the tracking store, the signature and the alias you have just built
are those two obligations, arrived at from the engineering rather than from the
law.

**Regulation (EU) 2026/1744, the Digital Omnibus on artificial intelligence.**
<http://data.europa.eu/eli/reg/2026/1744/oj> — the instrument that moved the
Annex III high-risk obligations from 2 August 2026 to 2 December 2027. Worth a
glance only for the date, and for the habit of checking one: a compliance date
quoted from a slide is the kind of fact that goes stale between one teaching
year and the next.

## The exam question for Module 3

> **A service you deployed six months ago has been returning status 200 to every
> request the whole time, with unchanged latency and no errors in the logs. A
> colleague believes its answers have been wrong for weeks. Describe how that is
> possible, how you would establish whether it is true, and what you would have
> had to build beforehand to answer in an afternoon rather than a fortnight.**

A strong answer reaches for training–serving skew — a preparation that changed
on one side only, with nothing to raise an error — and names at least one other
silent failure, such as an upstream field changing meaning while keeping its
name. It distinguishes what can be checked without labels (the distribution of
inputs, the distribution of outputs, the rate of decisions) from what cannot
(accuracy). And on the "beforehand": provenance in every response, so a change
in behaviour attaches to a version rather than to a guess; a registry, so
"what would it have said yesterday" is a lookup; and a stored contract, so an
input monitor knows what valid looked like. Module 4 supplies the statistics for
the comparison and Module 5 assembles them into a monitor.

Twelve further questions, for the oral examination, are in `Module 3/EXAM.md` —
four of them answerable only by somebody who ran these labs, because they turn on
a number the student measured rather than one the deck printed.
