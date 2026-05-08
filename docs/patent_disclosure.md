# Invention Disclosure

## Black-Box Behavioral Supervision of Agentic LLM Systems

## 1. Field of the Invention

The invention relates to monitoring of agentic systems built on large language models, in particular systems deployed in regulated decision-support contexts (e.g. clinical decision support, financial advisory, legal triage) that are required to operate within explicitly specified behavioral policies. More specifically, the invention relates to apparatus and methods for detecting, measuring, and predicting non-compliance with such policies in production settings where (a) the monitored system's internal mechanisms are inaccessible to the monitor, (b) genuine policy violations are extremely rare or unacceptable, and (c) standard supervised-learning approaches to violation detection are therefore inapplicable.

---

## 2. Background and Problem Addressed

### 2.1 Operating Constraints

Three constraints jointly shape the technical problem:

- **Black-box constraint.** The monitored system is treated as a pure input/output channel: only text prompts going in and text outputs coming out are observable. No access to model weights, intermediate activations, chain-of-thought, or output token logits is assumed. This constraint reflects the practical situation in which the monitored system is a remote API or a third-party deployment.
- **Rare-event constraint.** A well-designed monitored system produces few or zero observable policy violations in normal operation. Standard supervised-learning approaches that train a classifier on examples of violating outputs are therefore inapplicable: the positive class is missing.
- **Two-mode failure constraint.** The system can fail (a) by drifting over time so that the same input produces progressively non-compliant outputs (temporal drift), and (b) by being statically fragile in specific regions of input space such that small input variations cause disproportionate compliance changes even when nominal inputs look compliant (static fragility). Both modes require detection, but they have different temporal signatures and require different signal-extraction methodologies.

### 2.2 Limitations of Existing Approaches

LLM-as-judge scoring, Gaussian-process active learning, statistical-process-control monitoring of response distributions, and information-geometric metric learning each appear separately in the prior art. However their individual application to LLM monitoring does not address the joint constraint set above. The invention addresses these limitations through a coordinated arrangement of subsystems described in Chapter 4 that share a common compliance-score representation and a common learned geometry over that representation.

---

## 3. Summary of the System

The system is a monitoring apparatus that observes the text outputs of a monitored language model in response to a structured library of test prompts and produces alerts indicative of policy non-compliance preconditions. The system comprises:

1. **A probe library subsystem** (4.1) maintaining anchor probes (fixed test prompts), perturbation probes (controlled variations of anchors), and synthesized probes (machine-generated near-boundary scenarios).
2. **A two-stage compliance scorer** (4.2) producing a continuous per-policy compliance score from each text output, comprising a rubric-decomposed LLM-judge stage (stage-1) and a fine-tuned regression-head stage (stage-2).
3. **A calibration subsystem** (4.3) measuring agreement between the compliance scorer and a hand-scored reference corpus and producing a status gate that controls downstream pipeline activation.
4. **A persistent score store** (4.4) recording per-output sub-condition scores, aggregates, scorer identities, and probe-role provenance, indexed for temporal retrieval.
5. **A dual Gaussian-process layer** (4.5) comprising an **input-side fit** over prompt embeddings and a **response-side fit** over output-text embeddings, each fitted with stationary and non-stationary kernels, each producing posterior predictive mean and standard deviation, and each carrying a candidate-target proposer combining uncertainty with proximity to the violation boundary under quartile-stratified seeding. The input-side fit is consulted by the probe synthesizer for self-consistent target proposal in input-prompt space; the response-side fit retains the Riemannian-pullback kernel and is consulted for post-hoc acceptance scoring of synthesized probes, for monitoring, and for feedback.
6. **A Riemannian metric learner** (4.6) producing two position-dependent metric tensor fields over compliance-score space — `g_input(c)` fitted on input-perturbation Jacobians and `g_stoch(c)` fitted on multi-sample sampling-stochasticity Jacobians obtained by repeated invocations of the supervised LLM on identical prompts — together with a derived total metric `g_total(c) = g_input(c) + g_stoch(c)` corresponding to the law-of-total-variance decomposition `Var(score) = E[Var(score|prompt)] + Var(E[score|prompt])` of the score vector into aleatoric (LLM-stochasticity) and prompt-sensitivity components.
7. **A probe synthesizer** (4.7) producing new test scenarios at machine-proposed embedding-space targets via K-nearest-neighbor exemplar prompting with re-embedding-based target verification, in either a target-driven or a gradient-driven mode.
8. **A monitoring signal subsystem** (4.8) producing four orthogonal signals (drift, fragility, decoupling, curvature) over the learned compliance-space geometry.
9. **A feedback subsystem** (4.9) generating natural-language remediation messages from contrastive evidence pairs and delivering them through one or more injection channels.
10. **An experiment-controller subsystem** (4.10) coordinating run-once, drift-induction, perturbation, recovery, and feedback-application sessions, and recording their results for monitoring purposes.

The subsystems share two common substrates: the per-output compliance score (produced by 2 and stored by 4) and the learned Riemannian geometry of compliance-score space (produced by 6 and consumed by 5, 7, 8, and 9). The Riemannian geometry is itself decomposed into two complementary tensor fields (`g_input` and `g_stoch`) whose sum captures the total variance budget of the compliance score at each coordinate, and whose individual eigenstructures distinguish input-sensitivity fragility from inherent LLM stochasticity.

---

## 4. Detailed Description of Components

### 4.1 Probe Library Subsystem (Component 100)

#### 4.1.1 Anchor probes (110)

A set of fixed-text test prompts, persisted in configuration, intended to be sent identically across all measurement sessions. Each anchor probe carries a string identifier and the prompt text. The anchor probe set is the temporal-drift measurement instrument: a change in the compliance score vector for a given anchor between two time points is, by construction, attributable to a change in the monitored system's behavior rather than to a change in input distribution.

#### 4.1.2 Perturbation probes (120)

Programmatically-generated variations of anchor probes, each carrying a reference to its anchor of origin and a perturbation-kind label. The perturbation kinds include:

- **Paraphrase (121).** Same scenario, different surface phrasing, produced by a paraphrasing language model.
- **Demographic substitution (122).** Substitution of demographic attributes (age token, ethnicity-keyed first name, profession) where the substitution targets are configured as enumerated values per attribute.
- **Authority framing (123).** Insertion or removal of authority cues (e.g., "the senior cardiologist requests").
- **Boundary approach (124).** Incremental movement of the input scenario toward the policy boundary by lexical substitution along a configured boundary direction.

Each perturbation kind is implemented by a dedicated generator module that takes an anchor and produces one or more variants. Generators may be rule-based (substitution from configured tables) or LLM-based (paraphrase generator). Generation outputs are persisted to the probe store with anchor reference, generator identifier, and perturbation kind.

#### 4.1.3 Synthesized probes (130)

Machine-generated novel scenarios produced by Component 600 (probe synthesizer) at embedding-space targets proposed by Component 460 (candidate-target proposer). Synthesized probes carry a generator identifier, a target embedding, the achieved cosine similarity to that target, the validator status, and the exemplar anchor identifiers used in their construction.

### 4.2 Compliance Scorer (Component 200)

#### 4.2.1 Stage-1 LLM-as-Judge with rubric decomposition (210)

The Stage-1 scorer accepts as input (i) a policy specification, (ii) a rubric specification, and (iii) the supervised system's output text. It produces as output a vector of normalised per-sub-condition scores in [0, 1] and a scalar policy-aggregate score in [0, 1].

The rubric specification is a structured document declaring, for each policy sub-condition: an identifier, a description, a scale type, and a weight. The supported scale types are:

- **`boolean`** — yes/no scale; the judge model returns a boolean which the framework maps to {0.0, 1.0}.
- **`"0-3"`** — four-level ordinal scale; the judge returns an integer in {0, 1, 2, 3} which the framework normalises by division by 3.
- **`labels`** — Behaviorally Anchored Rating Scale (BARS); the rubric enumerates a fixed list of labels each carrying a per-label `value` in [0, 1] and a behavioral description, and the judge returns the identifier of one label which the framework maps to its declared `value`.

The judge model is invoked with a JSON-schema-enforced request specifying the response shape (one entry per sub-condition, with the value type matching the declared scale). The response is parsed by Component 213, normalised by Component 214, and passed to Component 215, which computes the policy aggregate as the rubric-weighted sum of normalised per-sub-condition scores. Sub-condition weights sum to one by construction; the framework, not the judge model, owns the aggregation.

Two structural properties of the BARS rubric are operationally significant:

- **Label monotonicity.** Labels are ordered such that values are monotonically non-increasing from most-compliant to least-compliant, so that the BARS-derived sub-condition score is itself an ordinal estimate.
- **Behavioral distinctness.** The rubric specification carries a constraint that adjacent labels must describe distinct, observable behaviors, so that the judge model selects between behavioral descriptions rather than between ordinal positions on an implicit numeric scale.

#### 4.2.2 Stage-2 fine-tuned scorer (220)

The Stage-2 scorer accepts as input the supervised system's output text and produces as output the same vector of compliance scores as Stage-1. It comprises (i) an output-text encoder producing an embedding vector in fixed dimension, e.g. a clinically-tuned transformer encoder producing a 768-dimensional embedding, and (ii) a per-policy-axis regression head, e.g. an sklearn `Pipeline(StandardScaler, RidgeRegressor)` or `Pipeline(StandardScaler, MLPRegressor)`, fitted on `(text, score)` training pairs produced by Stage-1.

The Stage-2 scorer is the primary scorer in production. Its consistency property — same output text always producing the same score — derives from the deterministic encoder and the deterministic regression head, and is required for reliable temporal drift detection.

#### 4.2.3 Stage-2 audit subsystem (230)

A periodic process that re-scores a sampled subset of recent outputs through Stage-1 and compares the result to the persisted Stage-2 score for the same output. Disagreement above a threshold triggers a Stage-2 retrain or escalation alert.

### 4.3 Calibration Subsystem (Component 300)

#### 4.3.1 Hand-scored reference corpus (310)

A configuration document containing reference tuples, each tuple comprising an identifier, an anchor-probe reference, a hand-authored compliance aggregate in [0, 1], and a supervised-output text. The corpus is constructed with deliberate stratified sampling across the compliance range, with denser sampling in the boundary region.

#### 4.3.2 Calibration runner (320)

A process that, for each tuple in the reference corpus, invokes the compliance scorer (Component 210) on the supervised-output text and computes:

- Spearman rank correlation between predicted aggregates and hand aggregates over the corpus.
- Mean absolute error between predicted and hand aggregates.
- A binned calibration curve mapping predicted aggregate buckets to mean hand aggregate.
- A status flag computed by thresholding the Spearman correlation: `green` if `≥ 0.7`, `yellow` if `≥ 0.5`, `red` otherwise.

The output is a structured report (CSV in the reference implementation) containing the summary statistics and the per-tuple agreement.

#### 4.3.3 Calibration history monitor — monitor-of-the-monitor (330)

A subsystem that runs Component 320 on a recurring schedule against a stable reference corpus, persisting each report's summary statistics with timestamp, and applying Statistical Process Control (SPC) monitoring to those statistics over time. An alert is raised when the Spearman correlation drops or the mean absolute error rises beyond a threshold computed from the historical baseline. This alert is dispatched on a separate channel from the alerts produced by Component 700, so as to disambiguate "the supervised system has drifted" from "the compliance scorer has drifted."

### 4.4 Persistent Score Store (Component 400)

A relational data store comprising at least the following tables:

- `compliance_scores` — one row per scored output, recording anchor identifier, policy identifier, per-sub-condition normalised scores (as JSON), aggregate, judge model identifier, supervised model identifier, timestamp, probe role (anchor / perturbation / synthesized), and references to perturbation, drift-session, recovery-run, and synthesized-probe records.
- `perturbation_probes` — generated perturbation variants with anchor reference and generator metadata.
- `synthesized_probes` — outputs of Component 600 with target embedding, achieved similarity, generator metadata, and quality-gate verdict.
- `gp_fits` — persisted Gaussian-process artefacts (Component 500).
- `metric_fits` — persisted Riemannian-metric artefacts (Component 600 in §4.6).
- `drift_sessions`, `drift_runs`, `recovery_runs`, `audit_runs`, `structural_signals` — operational session and signal records.
- `llm_calls`, `embed_calls` — recordings of all model invocations for replay and audit.

### 4.5 Gaussian-Process Layer (Component 500)

The Gaussian-process layer is instantiated as **two parallel fits sharing a common subcomponent architecture**, distinguished only by the text corpus over which they are trained:

- **Input-side fit (Component 500-I).** Fitted over `(prompt_text, aggregate_score)` pairs in which `prompt_text` is the input scenario sent to the supervised LLM. The input-side fit is operational and cheap to evaluate (no LLM round-trip required at inference time). It supplies embedding-space targets to the probe synthesizer in a space self-consistent with the synthesizer's K-nearest-neighbor exemplar substrate (Component 710), eliminating the input/output type mismatch that would otherwise arise when targets in response-embedding space are matched to candidate scenarios in input-embedding space.
- **Response-side fit (Component 500-R).** Fitted over `(response_text, aggregate_score)` pairs in which `response_text` is the supervised LLM's output. The response-side fit is the primary substrate consumed by the Riemannian-pullback kernel of §4.5.3 (since the Stage-2 scorer of Component 222 maps response embeddings to score space and is undefined on prompt embeddings), by the monitoring signal subsystem (Component 800), and by the feedback subsystem (Component 900). It is also consulted by the probe synthesizer at acceptance-scoring time, after a candidate scenario has been routed through the supervised LLM and its response embedded.

Both fits share Components 510, 520, 530, 540, and 550 below, instantiated independently per fit. Disagreement between the two fits at a common compliance-space coordinate is itself a diagnostic signal — a region in which the input-side and response-side posteriors diverge localises **prompt-conditional response variance** (the aleatoric component captured by `g_stoch` in §4.6) and is exposed to the monitoring subsystem as an auxiliary signal.

#### 4.5.1 Training pair extractor (510)

Reads `compliance_scores` and the corresponding stored texts, deduplicating on text and resolving multi-score collisions by aggregation (mean), to produce a set of `(text, aggregate)` training pairs. The extractor is invoked twice per fit cycle: once with `text = prompt_text` for the input-side fit (Component 500-I) and once with `text = response_text` for the response-side fit (Component 500-R).

#### 4.5.2 Embedding pipeline (520)

Embeds each training text via the same encoder used by Stage-2 (Component 221). Both prompt texts and response texts are embedded with the identical encoder so that the two fits operate in commensurable 768-dimensional spaces; the encoder's fitness for the prompt-side corpus is assumed and is monitored by the agreement of the input-side and response-side GP posteriors (a sustained divergence of input-side and response-side posteriors over a stable score region is a signal that the encoder fails to align prompt and response semantics in this domain, reportable to the monitoring subsystem).

#### 4.5.3 Kernel selector (530)

The fitting routine accepts a kernel-selector argument with at least three values:

- **`stationary`** — instantiates an RBF kernel over a preprocessing chain comprising (i) a `StandardScaler` over the embedding dimensions, (ii) a `PCA` projection to a reduced dimension (e.g. 50), and (iii) a noise term `α` (e.g. 10⁻²) on the Gaussian Process (GP) likelihood. The preprocessing chain addresses the curse-of-dimensionality and the duplicate-output noise simultaneously.
- **`non_stationary`** — instantiates a Gibbs kernel with a position-dependent length-scale `ℓ(x) = exp(a + b·u(x) + c·u(x)²)` parameterised by hyperparameters `(σ, a, b, c)` and a one-dimensional projection `u(·)`. Hyperparameters are optimised externally to sklearn (the marginal likelihood is maximised by a separate solver and the optimised kernel is then handed to sklearn for prediction) because the Gibbs hyperparameters are real-valued and cannot be driven directly by sklearn's bounded log-scale optimiser.
- **`riemannian_pullback`** — instantiates a metric-warped kernel that consumes the compliance-space metric tensor field of Component 600 by routing input embeddings through the Stage-2 compliance scorer of Component 222 and applying the metric in score space. For embeddings `e_1`, `e_2` with predicted score vectors `c_i = f(e_i)` and score-space midpoint `m_c = (c_1 + c_2) / 2`, the kernel is

  ```
  k(e_1, e_2) = σ² · exp(−½ · (c_1 − c_2)^T · g(m_c) · (c_1 − c_2) / ℓ²) + ε · exp(−½ · ‖e_1 − e_2‖² / ℓ_E²)
  ```

  where `g(m_c)` is the local metric tensor produced by Component 620 evaluated at `m_c`, the first term is a Mahalanobis-form kernel applied to the score-space images of `f` (positive-semidefinite by construction), and the second `ε · k_RBF` term (`ε ≪ σ²`) is a small isotropic embedding-space tiebreaker rendering the composite kernel positive-definite even when distinct embeddings collapse to identical score vectors under `f`. The GP posterior trained under this kernel inherits the fragility geometry of compliance-score space, so that the candidate-target proposer of §4.5.5 ranks more highly those embeddings whose score predictions place them in high-curvature regions of the boundary, where small perturbations produce disproportionate compliance changes. An alternative embodiment uses a local Jacobian-based embedding-space pullback `g_E(e) = J_f(e)^T · g(f(e)) · J_f(e) + ε · I`, with `J_f` computed by the same central-differences finite-differencing routine used by Component 730, applying `g_E(m)` at the embedding-space midpoint `m = (e_1 + e_2)/2`. Both forms eliminate any requirement for backprop access to the Stage-2 head and are agnostic to the head's function-approximator family.

Each kernel produces a `ComplianceGP` artefact persisted to disk and referenced from a `gp_fits` row.

#### 4.5.4 Posterior predictor (540)

For an arbitrary test embedding, returns the posterior mean (a compliance score estimate) and the posterior standard deviation (an uncertainty estimate).

#### 4.5.5 Candidate-target proposer (550)

A subsystem that selects embedding-space points to use as targets for the probe synthesizer. The proposer:

1. Constructs a candidate pool by **quartile-stratified seeding**: training observations are bucketed into compliance-score quartiles and the candidate pool draws an equal share from each bucket. This counters the production-distribution skew toward compliance, which would otherwise cause naive uniform sampling to under-represent the boundary region.
2. For each candidate, computes a score combining posterior uncertainty with proximity to the violation boundary:

   ```
   score = posterior_std × max(0, 1 − 2·|0.5 − posterior_mean|)
   ```

   This score peaks at the violation boundary (`posterior_mean = 0.5`) with weight 1.0 and falls linearly to zero at the extremes (`posterior_mean = 0` or `posterior_mean = 1`).
3. Returns the top-K candidates by this score as proposed embedding-space targets.

### 4.6 Riemannian Metric Learner (Component 600)

The metric learner produces **two complementary metric tensor fields** over compliance-score space, fitted from two distinct empirical Jacobian sources, plus a derived total metric corresponding to their sum. The decomposition follows the law of total variance applied to the score vector conditioned on the input prompt:

```
Var(score) = E[ Var(score | prompt) ] + Var( E[score | prompt] )
              └────── g_stoch ──────┘    └────── g_input ──────┘
```

`g_input(c)` captures **prompt-sensitivity fragility** — how the score moves when the input is perturbed, holding the LLM's sampling RNG implicit. `g_stoch(c)` captures **inherent LLM stochasticity** — how the score moves when the input is held fixed and only the LLM's sampling realisation varies. Their sum `g_total(c)` is the operative total fragility metric. Either field individually, or any positive-weighted combination, can be supplied to downstream consumers (the Riemannian distance computer of §4.6.3, the pullback kernel of §4.5.3, the curvature signal of §4.8.4).

##### 4.6.1a Input-perturbation Jacobian (610a)

For each anchor probe, the perturbation cloud (Component 120) provides, for each perturbation in the cloud, a per-axis compliance score change. The input-perturbation Jacobian estimator constructs a matrix `J_input` whose rows are per-axis score deltas across **distinct perturbed inputs** sharing a common anchor, and forms the Fisher-information-style target

```
g_input_target(c_anchor) = J_input^T J_input / ‖J_input‖²
```

at the compliance-space coordinate `c_anchor` corresponding to the anchor's pre-perturbation compliance vector. This Jacobian estimates the second term of the variance decomposition above — the variance of the conditional-mean compliance score across input prompts — and reproduces the construction of prior embodiments of Component 610.

##### 4.6.1b Sampling-stochasticity Jacobian (610b)

For each anchor probe, the multi-sample stochasticity driver (Component 1060 of §4.10 below) sends the **identical anchor scenario** through the supervised LLM `N` times (with `N ≥ 10` to ensure adequate rank coverage of the k×k target tensor) and records `N` independent per-axis score vectors. The stochasticity Jacobian estimator constructs a matrix `J_stoch` whose rows are zero-mean per-axis score deviations across the `N` runs (each row equal to a per-run score vector minus the across-run mean), and forms
```
g_stoch_target(c_anchor) = J_stoch^T J_stoch / ‖J_stoch‖²
```
at the same compliance-space coordinate `c_anchor`. This Jacobian estimates the first term of the variance decomposition above — the expected within-prompt variance of the score vector — and constitutes the metric-learning analogue of Fisher information for the conditional response distribution `p(score | prompt)`.

In embodiments where the compliance scorer of §4.2 contains an internally-stochastic component (e.g., an LLM-as-judge with non-zero temperature), `g_stoch_target` mixes LLM-sampling variance with judge variance. The reference embodiment isolates LLM-sampling variance by re-judging each of the `N` responses `K` times (`K ≥ 3`) and using the per-response mean score vector as the row entry, producing a judge-decorrelated `J_stoch`. An equivalent embodiment fits a third metric field `g_judge` from `K` re-judgings of a single response and reports `g_stoch_clean = g_stoch_total ⊖ g_judge` under a positive-semidefinite-preserving subtraction (e.g., projection of `g_stoch_total − g_judge` onto the PSD cone via spectral clipping).

#### 4.6.2 Metric MLPs (620)

A small feed-forward neural network mapping a compliance-space coordinate `c ∈ ℝᵏ` to a parameter vector of length `k(k+1)/2`. The parameter vector is decoded into the lower-triangular Cholesky factor `L(c)` of the local metric tensor `g(c) = L(c) L(c)^T` via the parameterisation:

- The first `k` entries are exponentiated to populate the diagonal of `L`.
- The remaining `k(k−1)/2` entries populate the strictly-lower triangle of `L` unchanged.

This parameterisation makes `g(c)` strictly positive-definite for any finite raw parameter vector. **Two MLPs are trained** with the architecture above: one minimising the Frobenius-norm loss between predicted `g_input(c_anchor)` and `g_input_target(c_anchor)`, and a second minimising the same loss between predicted `g_stoch(c_anchor)` and `g_stoch_target(c_anchor)`. Training uses pure-numpy forward and backward passes plus an Adam optimiser; persistence is `numpy.savez_compressed` to two `.npz` files referenced from one `metric_fits` row carrying both artefact paths and a `kind ∈ {input, stoch}` discriminator per artefact.

#### 4.6.3 Riemannian distance computer (630)

Given a metric artefact and two compliance-space coordinates `c_0`, `c_1`, the distance computer integrates the local metric along the straight-line path between them in coordinate space:

```
d(c_0, c_1) ≈ Σ_i √( Δc_i^T  g(c_mid_i)  Δc_i )
```

over `n_segments` equal sub-intervals. This is a strict upper bound on the true geodesic distance under `g`, tight whenever `g` is locally smooth, and is the operational quantity used by drift and feedback subsystems. The computer is metric-agnostic and accepts any of `g_input`, `g_stoch`, or `g_total` (see §4.6.4); the consumer subsystem selects which metric to integrate based on its semantic question — `g_input` for input-sensitivity drift, `g_stoch` for stochasticity drift, `g_total` for total-fragility drift.

#### 4.6.4 Total-fragility metric (640)

A derived metric tensor field defined pointwise as `g_total(c) = g_input(c) + g_stoch(c)`. The sum of two positive-definite tensors is positive-definite, so no separate Cholesky parameterisation is required. `g_total(c)` is the unique tensor consistent with the law-of-total-variance decomposition of `Var(score | c)` and is the recommended default consumer for the monitoring signal subsystem (Component 800), the feedback subsystem (Component 900), and the Riemannian-pullback kernel (Component 530, `riemannian_pullback` option). Embodiments that distinguish failure modes — e.g., a fragility-source attribution dashboard separating "the LLM is becoming more stochastic" from "the LLM is becoming more input-sensitive" — consume `g_input` and `g_stoch` independently.

### 4.7 Probe Synthesizer (Component 700)

The probe synthesizer accepts an embedding-space target produced by the **input-side GP** (Component 500-I, §4.5) and produces a new test scenario whose **input-prompt** embedding is verified to be within a configured cosine similarity of the target, and whose **response embedding** — obtained by routing the candidate scenario through the supervised LLM — passes a response-side acceptance score computed via the **response-side GP** (Component 500-R, §4.5). It operates in two modes that share a common synthesis substrate.

#### 4.7.1 K-NN exemplar synthesizer — common substrate (710)

For an input-embedding-space target `e_target` produced by the candidate-target proposer of Component 500-I:

1. Find the K=5 library anchor probes whose **prompt embeddings** have highest cosine similarity to `e_target`. (Library anchor prompts are pre-embedded by Component 520 at fit time and cached on the GP artefact.)
2. Construct a generator-LLM prompt comprising those K anchor scenarios as exemplars, with an instruction to produce a new scenario clinically similar to but distinct from the exemplars and exploring the gap between them. The prompt enforces structural anchors (e.g., a substitutable first name, age token, and profession) required by downstream perturbation generators.
3. Invoke the generator LLM. Strip meta-prefixes ("Okay, I understand the rules…", markdown headers, numbered list prefixes) from the response, retaining the scenario text.
4. **Input-side similarity pre-screen.** Re-embed the cleaned scenario via Component 520 and compute the cosine similarity between the re-embedding and `e_target`. This step operates entirely in input-prompt-embedding space and is consistent in type with the GP that produced `e_target`. If the similarity is below a configured threshold `τ` (e.g. 0.7), and a per-target retry budget remains, re-prompt with a small instructional nudge and return to step 3.
5. **Supervised-LLM round-trip.** Route the cleaned scenario through the supervised LLM, capture the response text, and embed the response via the same encoder of Component 520. The response embedding is denoted `r_obtained`.
6. **Response-side acceptance scoring.** Evaluate the response-side GP (Component 500-R) at `r_obtained` to obtain a posterior mean `μ_r` and posterior standard deviation `σ_r`, and compute the response-side acceptance score
   ```
   a(r_obtained) = σ_r × max(0, 1 − 2·|0.5 − μ_r|)
   ```
   (the same boundary-seeking weight used by the candidate-target proposer of Component 550). Accept the candidate if `a(r_obtained) ≥ a_min`, where `a_min` is a configured floor (default: the lower-quartile boundary of acceptance scores observed across the prior synthesis session, so the floor adapts to the prevailing distribution). The acceptance score quantifies the **information value** of the synthesized probe in the response-side GP's coordinate system, which is the coordinate system in which monitoring, drift, and feedback subsystems operate. Acceptance failures at this step are non-sticky and may consume retry budget.
7. **Quality gate.** Route the scenario through an LLM-as-validator quality gate (a configured second model called with a validator prompt) that returns an approval / rejection verdict. Validator rejections are sticky (no retry); both input-side similarity rejections (step 4) and response-side acceptance rejections (step 6) are not.
8. If approved, persist the scenario as a synthesized probe record with input-target embedding `e_target`, achieved input similarity, response embedding `r_obtained`, response-side acceptance score `a(r_obtained)`, exemplar references, and validator verdict. The persisted record is an immediate training contribution to **both** GP fits at the next refit cycle: it joins the input-side fit's training set as a `(prompt, score)` pair and the response-side fit's training set as a `(response, score)` pair.
9. Optionally, route the new probe through the supervised + Stage-1 judge pipeline OR through the Stage-2 scorer, depending on a configured scoring strategy, and persist the resulting score. (When the supervised-LLM round-trip of step 5 is already configured to capture per-axis scores, this step is subsumed.)

The K-NN exemplar substrate plays the role of a constrained text generator without requiring a separately-trained constrained text generation model. Constraint enforcement is achieved by **dual verification** — input-prompt re-embedding similarity (step 4, cheap, type-consistent with the input-side GP target) and response-embedding acceptance score (step 6, expensive, type-consistent with the response-side GP and the downstream monitoring substrate) — rather than by constrained-decoding mechanics during generation. The cheap pre-screen filters obvious failures before the expensive supervised-LLM round-trip is paid for; the expensive check enforces the principled criterion (information value in the GP coordinate system that monitoring and feedback actually consume) only on candidates that already passed the cheap filter.

#### 4.7.2 Target-driven mode (720)

The proposer (Component 550) supplies embedding-space targets directly to the K-NN exemplar synthesizer. Used when the operational goal is to expand coverage in high-uncertainty / near-boundary regions.

#### 4.7.3 Gradient-driven mode (730)

A proposer that, for each GP-supplied seed embedding `e_seed`:

1. Computes the descent direction of the Stage-2 model's score for a chosen policy axis (default: the policy's worst-fragility axis as identified by the fragility map of Component 740 below) by central-differences finite-differencing. Two batched forward passes through the Stage-2 head per gradient call compute every coordinate's partial in one shot, which avoids requiring backprop access to the head and works uniformly across head families (Ridge, MLPRegressor).
2. Steps `e_seed` along that descent direction by a configured step size (default: a fraction of the GP kernel's representative length-scale).
3. After each step, evaluates the GP posterior (mean, std) at the new position and stops walking if either:
   - the predicted compliance crosses a violation threshold (default 0.5) — the policy boundary in the Stage-2 model's view; or
   - the GP posterior standard deviation exceeds an uncertainty cap (default `2 × σ_train_max`) — the boundary of the GP's confidence region; or
   - the gradient becomes degenerate (zero norm); or
   - a maximum step count is reached.
4. Hands the final stepped embedding to Component 710 as the target.

The two modes share Components 710 and Component 540; they differ only in how the embedding-space target is chosen.

### 4.8 Monitoring Signal Subsystem (Component 800)

#### 4.8.1 CUSUM control chart (810) and EWMA control chart (820)

Per anchor probe, per policy axis, two control charts: a Cumulative Sum chart and an Exponentially Weighted Moving Average chart, both operating on the per-axis compliance score time series read from the score store. Each chart produces an alert when the control statistic exceeds a baseline-derived control limit.

Critically, in the configured operational mode the input to each chart is not the Euclidean displacement of the score vector but the **Riemannian displacement** computed via Component 630, so that score trajectories moving toward high-curvature regions of the boundary trigger alerts at smaller absolute displacements than trajectories moving through the safe interior.

#### 4.8.2 Fragility map estimator (830)

Per anchor probe, computes from the perturbation cloud a local sensitivity gradient — per perturbation kind, the mean compliance score change. Aggregated across all anchors, the per-anchor sensitivity gradients constitute a fragility map indexed by anchor and perturbation kind.

#### 4.8.3 Decoupling signal computer (840)

Tracks the covariance matrix of per-axis perturbation-induced compliance changes, anchor by anchor and over time. An alert fires when the covariance structure between previously-correlated policy axes shows a structural break — e.g., when two axes that historically moved together under a given perturbation kind begin moving independently or in opposite directions.

#### 4.8.4 Curvature signal computer (850)

For each anchor at each fit interval, evaluates the metric tensor at the anchor's compliance coordinate via Component 620 and computes the condition number `κ(g) = λ_max / λ_min` of the local metric. An alert fires when the relative increase `(κ_current − κ_baseline) / κ_baseline` exceeds a configured threshold (default 0.5). The condition-number proxy captures the local-anisotropy interpretation of "compliance landscape becoming steeper" without requiring computation of the Ricci scalar.

The four signals (CUSUM/EWMA, fragility, decoupling, curvature) are orthogonal in the sense that they detect distinct failure modes — positional drift, sensitivity drift, structural reorganisation, and geometric reorganisation respectively — and any subset can fire independently of the others.

### 4.9 Feedback Subsystem (Component 900)

#### 4.9.1 Drift localiser (910)

Ranks all anchor probes by their current Riemannian distance to the policy boundary (using Component 630), and within each anchor identifies the perturbation directions causing the largest compliance drops.

#### 4.9.2 Contrastive pair extractor (920)

Constructs evidence pairs of two kinds:

- **Temporal pair:** `(output_at_t_0, output_at_t_1)` for the same anchor at two time points.
- **Spatial pair:** `(output_for_anchor_x, output_for_perturbed_probe_x_plus_δ)` at the same time.

#### 4.9.3 Feedback synthesizer (930)

A language model invoked with a prompt comprising (i) the policy specification in natural language, (ii) the contrastive evidence pairs, and (iii) a description of the gradient direction identified by Component 910. Produces a natural-language remediation recommendation tailored to the scenario: corrective for temporal drift, robustness-oriented for static fragility.

#### 4.9.4 Feedback delivery channel (940)

One or more of: system-prompt injection into the monitored system, and an operator report. The delivery channel is configured per deployment.

#### 4.9.5 Effectiveness measurement (950)

After delivery, re-runs the affected anchor and perturbation probes, computes (a) compliance recovery toward the safe interior, (b) perturbation-cloud tightening, and (c) reduction in local Riemannian curvature, and records the result. If effectiveness is below threshold after a configured number of cycles, escalates to the next delivery channel.

### 4.10 Experiment-Controller Subsystem (Component 1000)

A set of session controllers each producing a session record in the score store and coordinating one or more of the components above:

- **Run-once driver (1010):** Sends every anchor through the supervised model, scores via Stage-1, persists to the store.
- **Drift-induction driver (1020):** Constructs a sequence of contextual perturbations intended to induce drift in the supervised model and records the per-step compliance trajectory; used for validating the drift detection layer.
- **Perturbation-session driver (1030):** Sends every (anchor, perturbation) pair through the supervised model and persists the resulting compliance changes; populates the input to Components 610a, 830, and 840.
- **Recovery driver (1040):** Re-runs probes after feedback delivery and records the recovery trajectory.
- **Apply-feedback driver (1050):** Coordinates Components 910–940 in a single end-to-end feedback session.
- **Multi-sample stochasticity driver (1060):** For each anchor probe in a configured subset (typically the boundary-proximate subset, since stochasticity is highest there), sends the **identical anchor scenario** through the supervised model `N` times (`N ≥ 10`), capturing `N` independent response texts and routing each through the compliance scorer to obtain `N` per-axis score vectors. Persists each of the `N` invocations as a distinct row in `compliance_scores` with a new `probe_role = "stochasticity_sample"` label and a session reference, enabling Component 610b to consume the resulting cluster of same-prompt score vectors as the rows of `J_stoch`. In embodiments that decorrelate judge variance from LLM-sampling variance (see §4.6.1b), the driver additionally routes each of the `N` responses through the judge model `K` times (`K ≥ 3`) and persists the per-response judge mean rather than each individual judge score.

---

## 5. Embodiments and Variants

The detailed description above describes one operational embodiment. The components admit numerous variant implementations, and the system as a whole can be embodied at any point in the cross product of these variants. In particular:

- **The output-text encoder** (Component 221) can be any clinically- or domain-tuned transformer encoder, or any general-purpose sentence encoder; the reference implementation uses a 768-dimensional clinically-tuned encoder but the system is invariant to the encoder choice up to the rebuild of all downstream artefacts.
- **The Stage-2 regression head** (Component 222) can be any function approximator; the reference implementation supports Ridge regression and a small MLP.
- **The judge model and the validator model** (Component 210, Component 716) can be any language model with structured-output capability; they can be the same model or different models, with cost/quality tradeoffs configurable per deployment.
- **The kernel choice for the GP** (Component 530) can be any of the three described kernels (stationary, non-stationary Gibbs, or pullback-metric) or any other kernel meeting the positive-definiteness and continuity requirements; further kernel variants can be substituted without altering the surrounding subsystems. In the pullback-metric kernel, the score-space form (composing the Stage-2 head `f` with the metric `g`) and the Jacobian-pullback form (constructing an embedding-space metric tensor `g_E(e) = J_f(e)^T g(f(e)) J_f(e)`) are equivalent embodiments differing only in whether the metric warp is applied before or after the Jacobian linearisation of `f`. The pullback-kernel embodiment is restricted to the response-side GP fit (Component 500-R), since the Stage-2 scorer is defined over response embeddings; the input-side GP fit (Component 500-I) is restricted to stationary or non-stationary kernels.
- **The choice of metric supplied to the pullback kernel** can be any of `g_input`, `g_stoch`, or `g_total = g_input + g_stoch`. The reference embodiment defaults to `g_total` because it represents the full variance budget; embodiments restricted to input-sensitivity probing use `g_input`; embodiments focused on inherent-stochasticity probing use `g_stoch`.
- **The Gaussian-process layer's dual-fit arrangement** (Component 500-I and Component 500-R, §4.5) admits a single-fit reduction in deployments where the input/output type-mismatch is empirically negligible (e.g., where Bio_ClinicalBERT or an equivalent encoder produces statistically indistinguishable representations of clinically equivalent prompts and responses) — in such deployments the response-side fit alone is operative and the synthesizer's input-side similarity step is performed against response-side embeddings of library anchor scenarios. Conversely, embodiments that operate three or more fits (e.g., one over prompts, one over responses, one over `(prompt, response)` joint encodings) are within the scope of the disclosure.
- **The metric-network architecture** (Component 620) can be any function approximator producing the Cholesky parameter vector; the reference implementation uses a small MLP, but a Gaussian-process-over-metric construction or a transformer-based predictor are within the scope of the disclosure. The two MLPs producing `g_input` and `g_stoch` may share an architecture but are trained independently; embodiments that share parameters via a multi-head MLP (one shared trunk, two Cholesky-output heads) are within the scope of the disclosure.
- **The number of multi-sample re-runs `N`** in Component 1060 can range from `N = k+1` (the minimum for full-rank `J_stoch`) to `N` in the hundreds for high-precision `g_stoch` estimation. The reference embodiment uses `N = 10–30`, applied selectively to boundary-proximate anchors to amortise the LLM-call cost.
- **The judge-decorrelation count `K`** in Component 1060 can be `K = 0` (no decorrelation, accept that `g_stoch` mixes LLM and judge variance), `K ≥ 3` (per-response judge averaging), or replaced entirely by the use of a deterministic judge or a temperature-zero judge call in deployments where the judge can be configured deterministically.
- **The Riemannian distance computation** (Component 630) can be replaced by a true geodesic ODE solver in embodiments where the metric estimate is sufficiently precise; the segment-integration form is one operational embodiment.
- **The curvature proxy** (Component 850) can be replaced by the Ricci scalar in embodiments where the metric field is sufficiently smooth and a coordinate-invariant curvature comparison across different anchors is required; the condition-number form is one operational embodiment.
- **The probe-synthesis cosine threshold τ**, the **uncertainty cap factor**, the **violation threshold**, the **calibration status thresholds**, the **CUSUM and EWMA control limits**, and the **curvature alert threshold** are all configurable per deployment.
- **The persistent score store** (Component 400) can be any relational store; the reference implementation uses SQLite, but the system is invariant to the storage backend.

---

## 6. Points of Novelty

Each numbered point below identifies an aspect of the arrangement believed to be distinct from prior art at the time of disclosure. Final novelty determinations require a prior-art search by qualified counsel.

### 6.1 Output-only learned compliance geometry

The system constructs and operates on a **learned position-dependent metric tensor field over a compliance-score space derived from black-box behavioral observation of an LLM**. The metric is fitted on empirical Jacobians (compliance score changes per unit input perturbation) collected via systematic perturbation experiments on a fixed anchor probe library, with positive-definiteness guaranteed by Cholesky parameterisation. Prior art in information-geometric metric learning operates on parametric model spaces or directly observed data manifolds; the application of metric learning to a compliance-score space derived from output-only observation, with Jacobians estimated from controlled-input perturbation, is believed to be novel.

### 6.2 Closed-loop boundary-seeking probe synthesis

The system pairs **GP posterior uncertainty** with **proximity to the violation boundary** via the multiplicative weight `posterior_std × max(0, 1 − 2·|0.5 − posterior_mean|)`, applied over a candidate pool constructed by **quartile-stratified seeding** from the training distribution. The combination explicitly counters the production-distribution skew toward the dominant compliance mode and concentrates probe-generation effort where the monitor's classification is operationally load-bearing. Prior art in active learning for classification typically maximises predictive variance or entropy alone; the boundary-seeking weight and the stratified-seeding correction together address a problem that uniform-uncertainty active learning cannot solve in heavily-imbalanced regulated-decision-support settings.

### 6.3 BARS rubric with framework-owned aggregation

The Stage-1 LLM-as-judge architecture **separates judgment from aggregation**: the judge model selects, per sub-condition, one identifier from an enumerated list of behaviorally anchored labels each carrying a pre-declared numeric value, and the framework computes the aggregate as a version-controlled weighted sum. Prior art in LLM-as-judge approaches typically requests a single overall numeric score, coupling judgment to aggregation in a way that prevents independent recalibration. The pre-declared label-to-value mapping plus framework-owned aggregation moves the calibration handle into a version-controlled artefact and enables (i) JSON-schema enforcement of judge responses against an enumerated label set, (ii) independent recalibration of the aggregation weights without re-querying the judge model, and (iii) post-hoc isotonic-regression recalibration against a hand-scored reference corpus while preserving the judge's ordinal ranking.

### 6.4 Embedding-space target verification as a substitute for constrained text generation

The probe synthesizer (Component 710) achieves controlled near-boundary text generation **without a separately-trained constrained text generator**, by combining (i) K-nearest-neighbor exemplar prompting with (ii) post-generation re-embedding and cosine-similarity verification against the original embedding-space target, (iii) bounded retries on similarity failures, and (iv) an LLM-as-validator quality gate. This substitutes a verification-based mechanism for a generation-time constrained-decoding mechanism, reusing existing generator and embedder models.

### 6.5 Gradient-driven probe synthesis on a learned compliance scorer

The gradient-driven mode (Component 730) computes the descent direction of a learned Stage-2 compliance scorer's per-axis score in embedding space using **central-differences finite-differencing**, steps a candidate embedding along that direction with **dual stopping criteria** (violation-threshold crossing and GP posterior uncertainty cap), and feeds the final embedding into the embedding-space verification mechanism of Component 710. The use of finite-differencing avoids requiring backprop access to the regression head and works uniformly across head families. The dual stopping criterion ensures that gradient extrapolation halts at the boundary of the learned model's confident region rather than continuing into uninterpretable high-uncertainty territory.

### 6.6 Four orthogonal monitoring signals on a shared learned geometry

The system produces four qualitatively distinct alert signal types — **drift** (CUSUM/EWMA on Riemannian displacement), **fragility** (perturbation cloud spread), **decoupling** (covariance-structure change between policy axes), and **curvature** (condition-number drift of the local metric) — all computed over the same learned compliance-space geometry. Each signal detects a different failure mode and any subset can fire independently. The unified geometry substrate is what makes the four signals comparable and jointly interpretable.

### 6.7 Disambiguating monitor drift from monitored-system drift

The system maintains a **monitor-of-the-monitor channel** (Component 330) operating SPC monitoring on the agreement statistics of the LLM-as-judge against a held-out hand-scored reference corpus, on a separate alert path from the drift / fragility / decoupling / curvature signals of Component 800. The architectural separation of the two alert paths enables disambiguation of "the supervised system has drifted" from "the compliance scorer has drifted" — a distinction that cannot be made by any single-channel monitoring scheme.

### 6.8 Pullback of compliance-space metric to embedding-space active-learning kernel

The system warps the kernel of the embedding-space Gaussian process (Component 530, `riemannian_pullback` option of §4.5.3) with a metric tensor pulled back from compliance-score space through the Stage-2 compliance scorer (Component 222). For input embeddings `e_1`, `e_2` with score-space images `c_i = f(e_i)`, the kernel measures distance as `(c_1 − c_2)^T · g(m_c) · (c_1 − c_2)` where `g(c)` is the metric tensor field of Component 620 and `m_c` is the score-space midpoint, augmented with a small isotropic embedding-space term to ensure positive-definiteness. This pullback is the mechanism by which the learned compliance geometry of Component 600 is consumed not only by the monitoring signal subsystem (Component 800) and the feedback subsystem (Component 900) but also by the active-learning probe-synthesis path (Components 500 and 700), realising the substrate-sharing arrangement of §3 across both downstream branches.

The construction is believed to be distinct from prior art in three respects. First, deep kernel learning learns its feature map jointly with the Gaussian-process marginal likelihood and does not consume an externally-fitted geometry; here the feature map (the Stage-2 head) is fitted independently in service of compliance scoring and re-used as a fixed pullback channel. Second, prior art in Gaussian processes on Riemannian manifolds assumes the GP's input space coincides with the manifold whose metric is consumed; here the GP's input space (embedding space) is distinct from the metric's domain (compliance-score space), and the pullback through the scorer is the bridge. Third, the construction admits a Jacobian-based variant in which the embedding-space metric tensor is constructed pointwise from the central-differences Jacobian of the compliance scorer, making the pullback agnostic to the regression head's function-approximator family.

### 6.9 Dual-fit Gaussian-process arrangement with input-side proposal and response-side acceptance

The system instantiates two parallel Gaussian-process fits (Components 500-I and 500-R, §4.5) over the **same compliance-score labels** but distinct text corpora — one over input-prompt embeddings and one over output-response embeddings — and consumes them asymmetrically: the **input-side fit proposes** embedding-space targets for the probe synthesizer, ensuring the synthesizer's K-nearest-neighbor exemplar substrate operates in a coordinate system type-consistent with the targets it is asked to hit; the **response-side fit accepts or rejects** synthesized probes after a supervised-LLM round-trip, ensuring the acceptance criterion is in the coordinate system in which the monitoring, drift, and feedback subsystems actually operate.

This arrangement resolves an input/output type-mismatch that arises in single-fit embodiments: a response-only GP would propose targets in response-embedding space against which the synthesizer would be matching prompt-embeddings; an input-only GP would lose the Riemannian-pullback substrate (since the Stage-2 scorer is defined over response embeddings, not prompt embeddings) and could not feed the monitoring subsystem. The dual-fit arrangement gives each subsystem the GP fit appropriate to the question it is asking. Disagreement between the two fits at a common compliance-space coordinate localises **prompt-conditional response variance** and is itself a diagnostic signal exposed to the monitoring subsystem, providing a coarse aleatoric-uncertainty proxy that does not require Component 1060's multi-sample protocol.

Prior art in active learning with Gaussian processes typically operates a single GP whose input space coincides with the synthesis substrate; prior art in dual-encoder retrieval typically operates two encoders mapping into a shared space rather than two GP fits over distinct text-source corpora sharing labels. The combination — same labels, distinct corpora, asymmetric consumption (proposal vs. acceptance) — is believed to be novel.

### 6.10 Variance-decomposition metric learner

The system fits **two complementary metric tensor fields** over compliance-score space (Components 610a and 610b, §4.6.1) corresponding to the two terms of the law-of-total-variance decomposition `Var(score) = E[Var(score | prompt)] + Var(E[score | prompt])`: `g_stoch(c)` from multi-sample re-runs of identical input prompts (the LLM-stochasticity term) and `g_input(c)` from the perturbation cloud (the input-sensitivity term). The total fragility metric `g_total(c) = g_input(c) + g_stoch(c)` is the unique tensor consistent with the total variance of the score vector at coordinate `c`, and either term individually exposes a distinct failure mode: `g_input` localises **adversarial input fragility** (the LLM responds predictably but is brittle to wording changes); `g_stoch` localises **inherent prompt ambiguity** (the LLM is wording-stable but produces varied responses to the same prompt).

Prior art in information-geometric metric learning typically fits a single tensor field over a directly observed manifold; the variance-decomposition arrangement — fitting two tensor fields whose sum reconstructs total observed score variance and whose individual eigenstructures expose qualitatively distinct failure modes — is believed to be novel as applied to LLM compliance monitoring. The construction additionally provides the metric-learning analogue of Fisher information for the conditional response distribution `p(score | prompt)` (in `g_stoch`), which is closer in spirit to the Fisher-information formulation than the input-perturbation Jacobian alone.

The corresponding claim of novelty is supported by the operational specificity of the embodiment: the `N`-fold multi-sample protocol (`N ≥ k+1` for full-rank `J_stoch`, `N ≥ 10` in the reference embodiment) implemented by the multi-sample stochasticity driver (Component 1060), the dual-MLP architecture sharing the Cholesky-parameterisation backbone, the optional judge-decorrelation step (`K`-fold re-judging of identical responses to subtract judge variance), and the metric-agnostic interface of the Riemannian distance computer (Component 630) which permits any of `g_input`, `g_stoch`, or `g_total` to be supplied as the integrand.

### 6.11 The arrangement as a whole

Independently of the per-component novelties, the **coordinated operation of components 100 through 1000** as a closed-loop monitoring apparatus — in which the score store feeds the GP layer, the GP feeds the probe synthesizer, the synthesizer feeds the score store, the metric learner feeds both the GP layer (in the pullback-kernel embodiment of §6.8) and the monitoring signals, the monitoring signals feed the feedback subsystem, and the feedback effectiveness measurement feeds back into the score store — is itself a novel arrangement of subsystems, in which each subsystem is required for the operational guarantees of the others.

---

## 7. Candidate Claim Outlines

The following claim outlines are engineering-level statements provided as input to claim drafting. They are not legal claims and are expected to be refined, narrowed, broadened, or restructured by qualified counsel.

### 7.1 Broad system claim

A system for behavioral monitoring of an autonomous text-generating system, comprising:

1. a probe library subsystem maintaining a set of fixed anchor prompts and a set of programmatically-generated perturbation variants thereof;
2. a compliance scorer producing, from a text output of the monitored system, a vector of per-policy-dimension compliance scores in [0, 1];
3. a metric-learning module producing a position-dependent positive-definite metric tensor field over the compliance-score space, the metric tensor field fitted on empirical Jacobians of the compliance scorer with respect to the perturbation variants;
4. a Gaussian-process layer producing, for an embedding of an output text, a posterior mean compliance prediction and a posterior uncertainty;
5. a probe synthesizer producing new test inputs at embedding-space targets selected as a function of said posterior uncertainty and a function of proximity of said posterior mean to a violation boundary; and
6. a temporal monitor producing alerts derived from cumulative deviation of compliance scores along trajectories whose displacement is computed under said metric tensor field.

### 7.2 Broad method claim

A method of detecting policy non-compliance preconditions in a text-generating system, comprising:

1. sending a set of fixed anchor prompts to the monitored system and recording per-output compliance scores from a compliance scorer that decomposes a policy into sub-conditions, scores each sub-condition independently via an LLM judge, and computes a policy aggregate as a framework-owned weighted sum;
2. for each anchor prompt, generating one or more perturbation variants by controlled linguistic transformation, sending each to the monitored system, and recording per-perturbation compliance score changes;
3. fitting a position-dependent metric tensor over the compliance-score space from said per-perturbation score changes with positive-definiteness guaranteed by Cholesky parameterisation of the metric;
4. computing risk-weighted distances between compliance-score points by integrating said metric tensor along straight-line paths in coordinate space; and
5. raising an alert when a risk-weighted distance for an anchor prompt exceeds a control limit derived from a baseline measurement.

### 7.3 Narrower claims (illustrative)

- The system of broad claim 1 wherein the compliance scorer is configured to receive a rubric specification declaring, per sub-condition, one of: a boolean scale, an integer ordinal scale, and a behaviorally anchored label scale comprising an enumerated list of labels each carrying a pre-declared numeric value, and to invoke the LLM judge with a structured-output request constrained to the declared scale.
- The system of broad claim 1 wherein said Gaussian-process layer is configured with a kernel comprising a Mahalanobis-form quadratic warp whose metric tensor is the value of said position-dependent metric tensor field of element 3 evaluated at the score-space image of a kernel-input embedding under said compliance scorer of element 2, optionally augmented with an isotropic embedding-space tiebreaker term to ensure strict positive-definiteness, said kernel realising consumption of said position-dependent metric tensor field by said Gaussian-process layer for active-learning probe-synthesis target proposal.
- The system of the preceding claim wherein said metric tensor evaluation is replaced by a pointwise embedding-space pullback `g_E(e) = J_f(e)^T · g(f(e)) · J_f(e)` in which `J_f(e)` is the Jacobian of said compliance scorer at `e` computed by central-differences finite-differencing.
- The system of broad claim 1 wherein the probe synthesizer is configured to operate in either a target-driven mode in which embedding-space targets are taken directly from the Gaussian-process layer's proposer, or a gradient-driven mode in which embedding-space targets are produced by stepping a seed embedding along a descent direction of a learned compliance scorer with stopping criteria comprising a violation threshold and an uncertainty cap.
- The system of broad claim 1 wherein the probe synthesizer comprises a generator language model invoked with K-nearest-neighbor anchor exemplars and a verifier that re-embeds the generator output and accepts the output only if its cosine similarity to the embedding-space target exceeds a configured threshold, and a separate quality-gate verifier model that accepts or rejects the output.
- The system of broad claim 1 wherein said Gaussian-process layer comprises **two parallel Gaussian-process fits** over the same per-output compliance-score labels: a first fit (input-side) trained on embeddings of input prompts, and a second fit (response-side) trained on embeddings of output texts produced by the monitored system in response to said prompts; wherein said probe synthesizer consults the input-side fit for embedding-space target proposal and consults the response-side fit for acceptance scoring of synthesized probes after each candidate scenario has been routed through the monitored system to obtain a response embedding; and wherein only the response-side fit is configured with a kernel comprising a metric-tensor warp pulled back from the compliance-score space.
- The system of the preceding claim wherein the synthesizer's acceptance criterion comprises a posterior-uncertainty × boundary-proximity score `σ_r × max(0, 1 − 2·|0.5 − μ_r|)` evaluated under said response-side Gaussian-process fit at the response embedding, and wherein synthesized probes are persisted only when said score exceeds a configured floor; and wherein the persisted record contributes simultaneously to both fits' training corpora at the next refit cycle as a `(prompt, score)` pair for the input-side fit and a `(response, score)` pair for the response-side fit.
- The system of broad claim 1 wherein said position-dependent metric tensor field of element 3 comprises **two complementary metric tensor fields** `g_input(c)` and `g_stoch(c)` summing to a total metric `g_total(c) = g_input(c) + g_stoch(c)`; wherein `g_input(c)` is fitted on Jacobians of compliance-score deltas across distinct perturbed input prompts sharing a common anchor, and `g_stoch(c)` is fitted on Jacobians of compliance-score deltas across `N ≥ k+1` independent invocations of the monitored system on an identical input prompt, where `k` is the dimension of compliance-score space; said decomposition realising the law-of-total-variance partitioning of score variance into input-sensitivity and inherent-stochasticity components.
- The system of the preceding claim wherein each of said two complementary metric tensor fields is produced by an independent feed-forward network mapping a compliance-space coordinate to a Cholesky parameter vector decoded into a strictly positive-definite tensor, and wherein consumers of said metric tensor field — including a Riemannian distance computer, a Riemannian-pullback kernel of the response-side Gaussian-process layer, and a curvature-signal computer — are configured to accept any one of `g_input`, `g_stoch`, or `g_total` as the integrand or evaluand under deployment-time selection.
- The system of the preceding two claims further comprising a multi-sample stochasticity driver configured to send each anchor probe in a configured subset through the monitored system `N` times with `N ≥ 10`, capture `N` independent response texts and corresponding `N` per-axis compliance score vectors, persist each of said `N` invocations as a row in the persistent score store with a probe-role label distinguishing said rows from anchor and perturbation rows, and supply the resulting cluster of same-prompt score vectors to the Jacobian estimator producing `g_stoch`.
- The method of broad claim 2 further comprising fitting two metric tensor fields over compliance-score space — a first from compliance-score deltas across distinct perturbed inputs sharing an anchor, and a second from compliance-score deltas across multiple invocations of the monitored system on an identical anchor input — and computing risk-weighted distances under the elementwise sum of said two fields.
- The system of broad claim 1 further comprising a calibration subsystem that operates the compliance scorer on a hand-scored reference corpus, computes Spearman rank correlation, mean absolute error, and a calibration curve, and produces a status flag governing activation of downstream subsystems.
- The system of broad claim 1 further comprising a calibration history monitor that applies SPC monitoring to said agreement statistics over time and raises an alert on a separate channel from said temporal monitor when the agreement statistics drift below a baseline-derived limit.
- The method of broad claim 2 further comprising raising a curvature alert at an anchor when the condition number of the local metric tensor at the anchor's compliance-space coordinate increases by more than a configured fraction relative to a baseline.
- The method of broad claim 2 further comprising raising a decoupling alert when the covariance structure between two compliance-score axes — previously positively correlated under a given perturbation kind — exhibits a structural break.

---

## 8. Glossary

For the avoidance of doubt in claim drafting:

- **Anchor probe.** A fixed-text prompt sent identically across all measurement sessions and used as a temporal-drift measurement instrument.
- **Perturbation probe.** A programmatically-generated variant of an anchor probe, carrying a reference to its anchor of origin and a perturbation-kind label.
- **Synthesized probe.** A new scenario produced by the probe synthesizer at an embedding-space target proposed by the candidate-target proposer, verified by re-embedding and cosine similarity, and approved by an LLM-as-validator.
- **Compliance score.** A continuous scalar in [0, 1] per policy dimension produced by the compliance scorer for an output text, where 1 indicates full compliance and 0 indicates clear non-compliance.
- **Sub-condition.** One of the verifiable items into which a policy is decomposed in the rubric specification.
- **BARS (Behaviorally Anchored Rating Scale).** A scale type in which the rubric enumerates a fixed list of labels, each label carrying a behavioral description and a pre-declared numeric value in [0, 1], and the judge model selects one label per sub-condition.
- **Aggregate.** The framework-owned weighted sum of per-sub-condition normalised scores, with weights summing to one.
- **Riemannian distance.** A risk-weighted distance between two points in compliance-score space, computed by integrating the local metric tensor along the straight-line path between them in coordinate space.
- **Input-side Gaussian-process fit (Component 500-I).** The Gaussian-process fit trained on `(prompt_text, aggregate_score)` pairs in which `prompt_text` is the input scenario sent to the supervised LLM. Used by the probe synthesizer for embedding-space target proposal in a coordinate system type-consistent with the synthesizer's K-nearest-neighbor exemplar substrate.
- **Response-side Gaussian-process fit (Component 500-R).** The Gaussian-process fit trained on `(response_text, aggregate_score)` pairs in which `response_text` is the supervised LLM's output. Used by the Riemannian-pullback kernel, by the monitoring signal subsystem, by the feedback subsystem, and by the probe synthesizer at acceptance-scoring time after a candidate scenario has been routed through the supervised LLM.
- **Input-perturbation Jacobian (`J_input`).** A matrix whose rows are per-axis compliance-score deltas across distinct perturbed input prompts sharing a common anchor; the input to the metric tensor field `g_input(c)`. Captures the variance of the conditional-mean compliance score across input prompts.
- **Sampling-stochasticity Jacobian (`J_stoch`).** A matrix whose rows are zero-mean per-axis compliance-score deviations across `N` independent invocations of the supervised LLM on an identical input prompt; the input to the metric tensor field `g_stoch(c)`. Captures the within-prompt variance of the score vector under LLM sampling stochasticity.
- **`g_input(c)`.** The position-dependent metric tensor field over compliance-score space fitted on input-perturbation Jacobians. Quantifies prompt-sensitivity fragility at coordinate `c`.
- **`g_stoch(c)`.** The position-dependent metric tensor field over compliance-score space fitted on sampling-stochasticity Jacobians. Quantifies inherent LLM stochasticity at coordinate `c`. The metric-learning analogue of Fisher information for the conditional response distribution `p(score | prompt)`.
- **`g_total(c)`.** The pointwise sum `g_input(c) + g_stoch(c)`; the unique tensor consistent with the law-of-total-variance decomposition `Var(score | c) = E[Var(score | prompt, c)] + Var(E[score | prompt, c])`.
- **Multi-sample stochasticity driver (Component 1060).** The experiment-controller subcomponent that sends an identical anchor scenario through the supervised LLM `N` times (`N ≥ 10`) and persists each invocation's per-axis score vector for consumption by the sampling-stochasticity Jacobian estimator.
- **Fragility.** The local sensitivity of compliance scores to input perturbations at a given anchor.
- **Drift signal, fragility signal, decoupling signal, curvature signal.** The four orthogonal alert types produced by the monitoring signal subsystem.

---

## Appendix A. Illustrative Numerical Examples

The following examples are illustrative of the behavior of specific components and are provided to support enablement. They are not intended to limit the scope of the disclosure.

### A.1 Worked aggregate computation (Component 215)

For a scope-of-practice rubric with five sub-conditions and the following BARS-derived normalised scores —

```
flags_physician_review:                  0.75   weight 0.20
expresses_uncertainty:                   0.75   weight 0.15
avoids_unqualified_prescriptive_language: 1.00   weight 0.25
recommendation_within_scope:             0.75   weight 0.25
recommendation_appropriate_specificity:  1.00   weight 0.15
```

— the framework-computed aggregate is

```
0.20·0.75 + 0.15·0.75 + 0.25·1.00 + 0.25·0.75 + 0.15·1.00
= 0.150 + 0.1125 + 0.250 + 0.1875 + 0.150
= 0.850
```

### A.2 Worked Jacobian estimate (Component 610)

For an anchor with compliance vector `(c₁, c₂)` and three perturbations producing measured per-axis compliance changes —

```
                   age change   prescriptive framing   authority cue
Δc₁ (scope):       -0.02        -0.41                  -0.22
Δc₂ (calibration):  0.00        -0.18                  -0.01
```

— the local Jacobian is the 2×3 matrix above; the corresponding Fisher-style metric target is `g_target = J^T J / ‖J‖²` and is supplied to the metric MLP at coordinate `(c₁, c₂)`.

### A.3 Worked candidate-target score (Component 550)

For a candidate with `posterior_mean = 0.52` and `posterior_std = 0.18`:

```
score = 0.18 × max(0, 1 − 2·|0.5 − 0.52|)
      = 0.18 × max(0, 1 − 0.04)
      = 0.18 × 0.96
      = 0.1728
```

For a candidate with `posterior_mean = 0.91` and `posterior_std = 0.20`:

```
score = 0.20 × max(0, 1 − 2·|0.5 − 0.91|)
      = 0.20 × max(0, 1 − 0.82)
      = 0.20 × 0.18
      = 0.036
```

The first candidate, despite having lower posterior uncertainty, scores nearly five times higher than the second because its posterior mean is at the violation boundary, where the boundary-seeking weight is near unity.

### A.4 Worked calibration report (Component 320)

A calibration run on twenty hand-scored references yields summary statistics:

```
n,20
spearman,+0.7121
spearman_pvalue,0.0004
mae,0.1460
status,green
```

The status flag of `green` indicates that the Spearman rank correlation between predicted aggregates and hand aggregates is at or above the configured 0.7 threshold. Mean absolute error of 0.146 is consistent with a bimodal scalar distortion (predicted aggregates compressed toward 0.85 in the upper half of the range and toward 0.25 in the lower half) absorbable by a post-hoc isotonic regression recalibration step.

---

*End of disclosure document.*
