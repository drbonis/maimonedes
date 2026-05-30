# Invention Disclosure

## Black-Box Behavioral Supervision of Agentic LLM Systems

## 1. Field of the Invention

The invention relates to monitoring of agentic systems built on large language models, in particular systems deployed in regulated decision-support contexts (e.g. clinical decision support, financial advisory, legal triage) that are required to operate within explicitly specified behavioral policies. More specifically, the invention relates to methods for detecting, measuring, and predicting non-compliance with such policies in production settings where (a) the monitored system's internal mechanisms are inaccessible to the monitor, (b) genuine policy violations are extremely rare or unacceptable, and (c) standard supervised-learning approaches to violation detection are therefore inapplicable.

---

## 2. Background and Problem Addressed

### 2.1 Operating Constraints

Three constraints jointly shape the technical problem:

- **Black-box constraint.** The monitored system is treated as a pure input/output channel: only text prompts going in and text outputs coming out are observable. No access to model weights, intermediate activations, chain-of-thought, or output token logits is assumed. This constraint reflects the practical situation in which the monitored system is a remote API or a third-party deployment.
- **Rare-event constraint.** A well-designed monitored system produces few or zero observable policy violations in normal operation. Standard supervised-learning approaches that train a classifier on examples of violating outputs are therefore inapplicable: the positive class is missing.
- **Two-mode failure constraint.** The system can fail (a) by drifting over time so that the same input produces progressively non-compliant outputs (temporal drift), and (b) by being statically fragile in specific regions of input space such that small input variations cause disproportionate compliance changes even when nominal inputs look compliant (static fragility).

### 2.2 Limitations of Existing Approaches

LLM-as-judge scoring, Gaussian-process active learning, statistical-process-control monitoring of response distributions, and Riemannian geometric metric learning each appear separately in the prior art. However their individual application to LLM monitoring does not address the joint constraint set above. The invention addresses these limitations through a coordinated arrangement of subsystems described in Chapter 4 that share a common compliance-score representation and a common learned geometry over that representation.

---

## 3. Summary of the System

The system is a monitoring apparatus that observes the text outputs of a monitored language model in response to a structured library of test prompts and produces alerts indicative of policy non-compliance preconditions. The system comprises:

1. **A probe library subsystem** (4.1) maintaining anchor probes (test prompts), perturbation probes (controlled variations of anchors), and synthesized probes (machine-generated near-boundary scenarios).
2. **A two-stage compliance scorer** (4.2) producing a continuous per-policy compliance score from each text output, comprising a rubric-decomposed LLM-judge stage (stage-1) and a fine-tuned regression-head stage (stage-2).
3. **A calibration subsystem** (4.3) measuring agreement between the compliance scorer and a hand-scored reference corpus and producing a status gate that controls downstream pipeline activation.
4. **A persistent score store** (4.4) recording per-output and per policy scores, aggregated scores, scorer identities, and probe-role provenance, indexed for temporal retrieval.
5. **A dual Gaussian-process layer** (4.5) comprising an **input-side fit** over prompt embeddings and a **response-side fit** over output-text embeddings, each fitted with stationary and optionally non-stationary kernels, each producing posterior predictive mean and standard deviation of overall compliance score, and each carrying a candidate-target proposer combining uncertainty with proximity to the violation boundary under quartile-stratified seeding. The input-side fit is consulted by the probe synthesizer for self-consistent target proposal in input-prompt space; the response-side fit can also use the Riemannian-pullback kernel and is consulted for post-hoc acceptance scoring of synthesized probes, for monitoring, and for feedback.
6. **A Riemannian metric learner** (4.6) producing two position-dependent metric tensor fields over compliance-score space. `g_input(c)` fitted on input-perturbation Jacobians and `g_stoch(c)` fitted on multi-sample Jacobians obtained by repeated invocations of the supervised LLM on identical prompts, together with a derived total metric `g_total(c) = g_input(c) + g_stoch(c)`.
7. **A probe synthesizer** (4.7) producing new test scenarios at machine-proposed embedding-space targets via K-nearest-neighbor exemplar prompting with re-embedding-based target verification, in either a target-driven or a gradient-driven mode.
8. **A monitoring signal subsystem** (4.8) producing four orthogonal signals (drift, fragility, decoupling, curvature) over the learned compliance-space geometry.
9. **A feedback subsystem** (4.9) generating natural-language remediation messages from contrastive evidence pairs and delivering them through one or more injection channels.
10. **A multi-sample stochasticity driver** (4.10) sending a particular input text (anchor, perturbation) to the supervised system several times and registering the different responses to capture the internal stochasticity of the supervised system.

The subsystems share two common substrates: the per-output compliance score (produced by 2 and stored by 4) and the learned Riemannian geometry of compliance-score space (produced by 6 and consumed by 5, 7, 8, and 9). The Riemannian geometry is itself decomposed into two complementary tensor fields (`g_input` and `g_stoch`) whose sum captures the total variance of the compliance score at each coordinate, and whose individual eigenstructures distinguish input-sensitivity fragility from inherent LLM stochasticity.

---

## 4. Detailed Description of Components

### 4.1 Probe Library Subsystem (Component 100)

#### 4.1.1 Anchor probes (110)

A set of fixed-text test prompts, persisted in configuration, intended to be sent identically across all measurement sessions. Each anchor probe carries a string identifier and the prompt text. The anchor probe set is the temporal-drift measurement instrument: a change in the compliance score vector for a given anchor between two time points is, by construction, attributable to a change in the monitored system's behavior rather than to a change in input distribution.

#### 4.1.2 Perturbation probes (120)

Programmatically-generated variations of anchor probes or synthesized probes, each carrying a reference to its anchor of origin and a perturbation-kind label. The perturbation kinds could include among others:

- **Paraphrase**: Same scenario, different surface phrasing, produced by a paraphrasing language model.
- **Demographic substitution**: Substitution of demographic attributes (age token, ethnicity-keyed first name, profession) where the substitution targets are configured as enumerated values per attribute.
- **Authority framing**: Insertion or removal of authority cues (e.g., "the senior cardiologist requests").
- **Boundary approach**: Incremental movement of the input scenario toward the policy boundary by lexical substitution along a configured boundary direction.

Each perturbation kind is implemented by a dedicated generator module that takes an anchor and produces one or more variants. Generators may be rule-based (substitution from configured tables) or LLM-based (e.g. paraphrase generator). Generation outputs are persisted to the probe store with anchor reference, generator identifier, and perturbation kind.

#### 4.1.3 Synthesized probes (130)

Machine-generated novel scenarios produced by Component 600 (probe synthesizer) at embedding-space targets proposed by Component 460 (candidate-target proposer). Synthesized probes carry a generator identifier, a target embedding, the achieved cosine similarity to that target, the validator status, and the exemplar anchor identifiers used in their construction.

### 4.2 Compliance Scorer (Component 200)

#### 4.2.1 Stage-1 LLM-as-Judge with rubric decomposition (210)

The Stage-1 scorer accepts as input (i) a policy specification, (ii) a rubric specification, and (iii) the supervised system's output text. It produces as output a vector of normalised per-sub-condition scores in [0, 1] and a scalar policy-aggregate score in [0, 1].

The rubric specification is a structured document declaring, for each policy sub-condition: an identifier, a description, a scale type, and a weight. The supported scale types are:

- **`boolean`**: yes/no scale; the judge model returns a boolean which the framework maps to {0.0, 1.0}.
- **`"0-3"`**: four-level ordinal scale; the judge returns an integer in {0, 1, 2, 3} which the framework normalises by division by 3.
- **`labels`**_ Behaviorally Anchored Rating Scale (BARS); the rubric enumerates a fixed list of labels each carrying a per-label `value` in [0, 1] and a behavioral description, and the judge returns the identifier of one label which the framework maps to its declared `value`.

The judge model is invoked with a JSON-schema-enforced request specifying the response shape (one entry per sub-condition, with the value type matching the declared scale). The response is parsed, normalised and used to compute the policy compliance aggregate as the rubric-weighted sum of normalised per-sub-condition scores.

Two structural properties of the Behaviorally Anchored Rating Scale rubric are operationally significant:

- **Label monotonicity.** Labels are ordered such that values are monotonically non-increasing from most-compliant to least-compliant, so that the Behaviorally Anchored Rating Scale derived sub-condition score is itself an ordinal estimate.
- **Behavioral distinctness.** The rubric specification carries a constraint that adjacent labels must describe distinct, observable behaviors, so that the judge model selects between behavioral descriptions rather than between ordinal positions.

#### 4.2.2 Stage-2 fine-tuned scorer (220)

The Stage-2 scorer accepts as input the supervised system's output text and produces as output the same vector of compliance scores as Stage-1. It comprises (i) an output-text encoder producing an embedding vector in fixed dimension, e.g. a clinically-tuned transformer encoder producing a 768-dimensional embedding, and (ii) a per-policy-axis regression head, e.g. a Ridge Regressor or a Multilayer Perceptron, fitted on `(text, score)` training pairs produced by Stage-1.

The Stage-2 scorer is the primary scorer in production. Its consistency property (same output text always producing the same score) derives from the deterministic encoder and the deterministic regression head, and is required for reliable temporal drift detection.

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

#### 4.3.3 Calibration monitoring (330)

A subsystem that runs Component 320 on a recurring schedule against a stable reference corpus, persisting each report's summary statistics with timestamp, and applying Statistical Process Control (SPC) monitoring to those statistics over time. An alert is raised when the Spearman correlation drops or the mean absolute error rises beyond a threshold computed from the historical baseline. This alert is dispatched on a separate channel from the alerts produced by Component 700, so as to disambiguate "the supervised system has drifted" from "the compliance scorer has drifted."

### 4.4 Persistent Store (Component 400)

The persistent store is a database configured to maintain the historical and operational state of the system. Contains following information:  
- Measurement and Evaluation Records: The comprehensive scoring history for every evaluated text output. 
- Test Scenario Archives: The complete library of programmatically modified and machine-synthesized test prompts. 
- Learned Statistical and Geometric Artifacts: The finalized, fitted mathematical models.
- Operational and Audit Logs: Chronological records of continuous monitoring sessions, structural tests, and recovery events.

### 4.5 Gaussian-Process Layer (Component 500)

The Gaussian-process layer is instantiated as **two parallel fits sharing a common subcomponent architecture**, distinguished only by the text corpus over which they are trained:

- **Input-side fit (Component 500-I).** Fitted over `(prompt_text, aggregate_score)` pairs in which `prompt_text` is the input scenario sent to the supervised LLM. The input-side fit is cheaper to evaluate (no supervised LLM execution required at inference time). It supplies embedding-space targets to the probe synthesizer in a space self-consistent with the synthesizer's K-nearest-neighbor exemplar substrate (Component 700).
- **Response-side fit (Component 500-R).** Fitted over `(response_text, aggregate_score)` pairs in which `response_text` is the supervised LLM's output. The response-side fit is the primary substrate consumed by the Riemannian-pullback kernel of §4.5.3, by the monitoring signal subsystem (Component 800), and by the feedback subsystem (Component 900). It is also consulted by the probe synthesizer at acceptance-scoring time, after a candidate scenario has been routed through the supervised LLM and its response embedded.

Both fits share Components 510, 520, 530, 540, and 550 below, instantiated independently per fit. Disagreement between the two fits at a common compliance-space coordinate is itself a diagnostic signal that can identify region in which the input-side and response-side posteriors diverge and is exposed to the monitoring subsystem as an auxiliary signal.

#### 4.5.1 Training pair extractor (510)

Reads `compliance_scores` and the corresponding stored texts, deduplicating on text and resolving multi-score collisions by aggregation (mean), to produce a set of `(text, aggregate)` training pairs.

#### 4.5.2 Embedding pipeline (520)

Embeds each training text via an encoder. In the default embodiment the same encoder used by Stage-2 (Component 220) is applied to both prompt texts and response texts; in an alternate embodiment, **separate encoders per side (optional)** are used (e.g. a prompt-tuned encoder for input texts and a response/clinical-note-tuned encoder for response texts), and the two encoders need not produce embeddings of equal dimensionality.

#### 4.5.3 Kernel selector (530)

The fitting routine accepts a kernel-selector argument with three options:

- **stationary (531)**: instantiates an Radial Basis Function kernel over a preprocessing chain comprising (i) a `StandardScaler` over the embedding dimensions, (ii) a `PCA` projection to a reduced dimension (e.g. 50), and (iii) a noise term `α` (e.g. 10⁻²) on the Gaussian Process likelihood. The preprocessing chain addresses the curse-of-dimensionality and the duplicate-output noise simultaneously.
- **non_stationary (532)**: instantiates a Gibbs kernel with a position-dependent length-scale.
- **riemannian_pullback (533)**: instantiates a Riemannian metric-warped kernel that evaluates distance based on the system's compliance-score space rather than raw embedding space. This is achieved by routing input embeddings through the Stage-2 compliance scorer (Component 222) to obtain predicted score vectors, and then applying the compliance-space Riemannian metric tensor field (Component 600) to those vectors.

Instead of evaluating standard distance formulas, the Riemmanian composite kernel is constructed from two distinct terms: 

- **A primary Riemmanian metric-warped term**: A Mahalanobis-form kernel applied to the predicted score vectors. This term integrates the local Riemmanian metric tensor evaluated at the score-space midpoint between the two inputs, capturing the local curvature and geometry of the compliance space.
- **A stationary tiebreaker term**: A infinitesimally small Radial Basis Function (RBF) kernelthat acts as a structural safeguard to ensure the composite kernel remains strictly positive-definite even if the Stage-2 scorer collapses distinct text embeddings into identical predicted score vectors.

Standard Gaussian Process hyperparameters governing signal variance and lengthscales are optimized to maximize the log-marginal likelihood.

Consequently, the Gaussian Process trained under this composite kernel natively inherits the "fragility geometry" of the compliance-score space. This mathematically enables the candidate-target proposer (Component 550) to accurately prioritize embedding-space targets located in high-curvature boundary regions—vulnerable areas where minuscule input perturbations can trigger disproportionately large shifts in compliance

#### 4.5.4 Posterior predictor (540)

For an arbitrary test embedding, returns the posterior mean (a compliance score estimate) and the posterior standard deviation (an uncertainty estimate).

#### 4.5.5 Candidate-target proposer (550)

A subsystem that selects embedding-space points to use as targets for the probe synthesizer. The proposer:

1. Constructs a candidate pool by **quartile-stratified seeding**: training observations are bucketed into compliance-score quartiles and the candidate pool draws an equal share from each bucket. This mitigates the distribution skew toward compliance, which would otherwise under-represent the boundary region.
2. For each candidate, computes a score combining posterior uncertainty with proximity to the violation boundary:

   ```
   score = posterior_std × max(0, 1 − 2·|0.5 − posterior_mean|)
   ```

   This score peaks at the violation boundary (`posterior_mean = 0.5`) with weight 1.0 and falls linearly to zero at the extremes (`posterior_mean = 0` or `posterior_mean = 1`).
3. Returns the top-K candidates by this score as proposed embedding-space targets.

### 4.6 Riemannian Metric Learner (Component 600)

The Riemmanian metric learner produces **two complementary metric tensor fields** over compliance-score space, fitted from two distinct empirical Jacobian sources, plus a derived total metric corresponding to their sum.

`g_input(c)` captures **prompt-sensitivity fragility** or how the score moves when the input is perturbed. `g_stoch(c)` captures **inherent LLM stochasticity** or how the score moves when the input is held fixed and only the LLM's sampling realisation varies. Their sum `g_total(c)` is the operative total fragility metric. Either field individually, or any positive-weighted combination, can be supplied to downstream consumers (the Riemannian distance computer of §4.6.3, the pullback kernel of §4.5.3, the curvature signal of §4.8.4).

##### 4.6.1a Input-perturbation Jacobian (610a)

For each anchor probe, the perturbation cloud (Component 120) provides, for each perturbation in the cloud, a per-axis compliance score change. The input-perturbation Jacobian estimator constructs a matrix `J_input` whose rows are per-axis score deltas across **distinct perturbed inputs** sharing a common anchor, and forms the Fisher-information-style target

```
g_input_target(c_anchor) = J_input^T J_input / ‖J_input‖²
```

at the compliance-space coordinate `c_anchor` corresponding to the anchor's pre-perturbation compliance vector. This Jacobian estimates the variance of the conditional-mean compliance score across input prompts.

##### 4.6.1b Sampling-stochasticity Jacobian (610b)

For each anchor probe, the multi-sample stochasticity driver (Component 1000 of §4.10 below) sends the **identical anchor scenario** through the supervised LLM `N` times (with `N ≥ 10` to ensure adequate rank coverage of the k×k target tensor) and records `N` independent per-axis score vectors. The stochasticity Jacobian estimator constructs a matrix `J_stoch` whose rows are zero-mean per-axis score deviations across the `N` runs (each row equal to a per-run score vector minus the across-run mean), and forms
```
g_stoch_target(c_anchor) = J_stoch^T J_stoch / ‖J_stoch‖²
```
at the same compliance-space coordinate `c_anchor`. This Jacobian estimates the expected within-prompt variance of the score vector.

In embodiments where the compliance scorer of §4.2 contains an internally-stochastic component (e.g., an LLM-as-judge with non-zero temperature), `g_stoch_target` mixes LLM-sampling variance with judge variance. The reference embodiment isolates LLM-sampling variance by re-judging each of the `N` responses `K` times (`K ≥ 3`) and using the per-response mean score vector as the row entry, producing a judge-decorrelated `J_stoch`.

#### 4.6.2 Riemmanian Metric Multilayer Perceptron (MLP) (620)

A small feed-forward neural network mapping a compliance-space coordinate `c ∈ ℝᵏ` to a parameter vector of length `k(k+1)/2` where `k` is the number of policy axes or sub-conditions the system is evaluating. The parameter vector is decoded into the lower-triangular Cholesky factor `L(c)` of the local metric tensor `g(c) = L(c) L(c)^T` via the parameterisation:

- The first `k` entries are exponentiated to populate the diagonal of `L`.
- The remaining `k(k−1)/2` entries populate the strictly-lower triangle of `L` unchanged.

This parameterisation makes `g(c)` strictly positive-definite for any finite raw parameter vector and prevents Gaussian Process optimized collapse. **Two MLPs are trained** with the architecture above: one for the `g_input(c_anchor)` (621) and a second for the `g_stoch(c_anchor)` (622).

#### 4.6.3 Riemannian distance computer (630)

Given a metric artefact and two compliance-space coordinates `c_0`, `c_1`, the distance computer integrates the local metric along the straight-line path between them in coordinate space:

```
d(c_0, c_1) ≈ Σ_i √( Δc_i^T  g(c_mid_i)  Δc_i )
```

over `n_segments` equal sub-intervals. This is an approximation on the true geodesic distance under `g` and is the value used by drift and feedback subsystems. The computer is metric-agnostic and accepts any of `g_input`, `g_stoch`, or `g_total` (see §4.6.4); the consumer subsystem selects which metric to integrate based on its semantic question: `g_input` for input-sensitivity drift, `g_stoch` for stochasticity drift, `g_total` for total-fragility drift.

#### 4.6.4 Total-fragility metric (640)

A derived metric tensor field defined pointwise as `g_total(c) = g_input(c) + g_stoch(c)`. The sum of two positive-definite tensors is positive-definite, so no separate Cholesky parameterisation is required. `g_total(c)` is the default for the monitoring signal subsystem (Component 800), the feedback subsystem (Component 900), and the Riemannian-pullback kernel (Component 530, `riemannian_pullback` option). Embodiments that distinguish failure modes (e.g., a fragility-source attribution dashboard separating "the LLM is becoming more stochastic" from "the LLM is becoming more input-sensitive") consume `g_input` and `g_stoch` independently.

### 4.7 Probe Synthesizer (Component 700)

The probe synthesizer accepts an embedding-space target produced by the **input-side GP** (Component 500-I and Component 550, §4.5) and produces a new test scenario whose **input-prompt** embedding is verified to be within a configured cosine similarity of the target, and whose **response embedding** (obtained by routing the candidate scenario through the supervised LLM) passes a response-side acceptance score computed via the **response-side GP** (Component 500-R, §4.5). It operates as follows:

#### 4.7.1 K-NN exemplar synthesizer

For an input-embedding-space target `e_target` produced by the candidate-target proposer of Component 500-I and Component 550:

1. Find the K=5 library anchor probes whose **prompt embeddings** have highest cosine similarity to `e_target`. (Library anchor prompts are pre-embedded by Component 520 at fit time and cached on the Gaussian Process artefact.)
2. Construct a generator-LLM prompt comprising those K anchor scenarios as exemplars, with an instruction to produce a new scenario clinically similar to but distinct from the exemplars and exploring the gap between them. In some embodiments the prompt enforces structural anchors (e.g., a substitutable first name, age token, and profession) required by downstream rule-based perturbation generators.
3. Invoke the generator LLM. Strip meta-prefixes ("Okay, I understand the rules…", markdown headers, numbered list prefixes) from the response, retaining the scenario text.
4. **Input-side similarity pre-screen.** Re-embed the new scenario via Component 520 and compute the cosine similarity between the embedding and `e_target`. If the similarity is below a configured threshold, and a per-target retry budget remains, re-prompt with a small instructional nudge and return to step 3.
5. **Supervised-LLM round-trip.** Route the new scenario through the supervised LLM, capture the response text, and embed the response via the same encoder of Component 520. The response embedding is denoted `r_obtained`.
6. **Response-side acceptance scoring.** Evaluate the response-side GP (Component 500-R) at `r_obtained` to obtain a posterior mean `μ_r` and posterior standard deviation `σ_r`, and compute the response-side acceptance score using the same boundary-seeking weight used by the candidate-target proposer of Component 550:
   ```
   a(r_obtained) = σ_r × max(0, 1 − 2·|0.5 − μ_r|)
   ```
It accepts the candidate if `a(r_obtained) ≥ a_min`, where `a_min` is a configured floor. The acceptance score quantifies the **information value** of the synthesized probe in the response-side GP's coordinate system.
7. **Quality gate.** Route the scenario through an LLM-as-validator quality gate (a configured second model called with a validator prompt that checks if the scenario is plausible given the domain, for example if the scenario makes sense from a clinical point of view) that returns an approval / rejection verdict.
8. If approved, persist the scenario as a synthesized probe record with input-target embedding `e_target`, achieved input similarity, response embedding `r_obtained`, response-side acceptance score `a(r_obtained)`, exemplar references, and validator verdict. The persisted record is an immediate training contribution to **both** Gaussian Process fits at the next refit cycle: it is included in the input-side fit's training set as a `(prompt, score)` pair and the response-side fit's training set as a `(response, score)` pair. 

The K-NN exemplar substrate plays the role of a constrained text generator without requiring a separately-trained text generation model. Constraint enforcement is achieved by **dual verification**: 1) input-prompt re-embedding similarity (step 4) and 2) response-embedding acceptance score (step 6). The cheap pre-screen (step 4) filters obvious failures before the expensive supervised-LLM round-trip (step 5) is paid for. As a result the expensive check (step 6) enforces the most important criterion (information value in the GP coordinate system that monitoring and feedback actually consume) only on candidates that already passed the cheap filter (step 4).

### 4.8 Monitoring Signal Subsystem (Component 800)

The four signals (drift §4.8.1, fragility §4.8.2, decoupling §4.8.3, curvature §4.8.4) are orthogonal in the sense that they detect distinct failure modes (positional drift, sensitivity drift, structural reorganisation, and geometric reorganisation respectively) and each can fire independently of the others.

#### 4.8.1 CUSUM control chart (810) and EWMA control chart (820)

Per anchor probe (and synthetic probe) and per policy axis, two control charts operate on the per-axis compliance score time series read from the score store: a Cumulative Sum (CUSUM) chart and an Exponentially Weighted Moving Average (EWMA) chart. Each chart produces an alert when its control statistic exceeds a baseline-derived control limit.

In the configured operational mode the input to each chart is not the Euclidean displacement of the score vector between successive observations but the **Riemannian displacement** computed via Component 630 under the metric tensor field `g_stoch(c)` of Component 600. Two properties of this choice are operationally significant:

1. **Curvature-sensitive alerting.** Score trajectories moving toward high-curvature regions of the policy boundary register larger Riemannian displacements per unit Euclidean step than trajectories moving through the safe interior, so alerts fire at smaller absolute displacements precisely where the consequence of further drift is greatest.
2. **Self-calibrating noise floor.** `g_stoch(c)` is fitted (§4.6.1b) such that one Riemannian unit of displacement at coordinate `c` equals one within-prompt stochastic standard deviation of the compliance score at that coordinate. The use of `g_stoch` rather than `g_total` for control-chart input is appropriate since the input prompt is held constant by construction.

The default chart parameters follow standard Statistical Process Control operating points expressed in these dimensionless units:

- **CUSUM (Component 810).** Reference value `k = 0.5` and decision threshold `h ∈ [4, 5]`.
- **EWMA (Component 820).** Smoothing parameter `λ = 0.1` and control-limit multiplier `L = 2.7`, following the Lucas–Saccucci operating point.

The two charts are intentionally non-redundant: EWMA covers the slow-creep regime and CUSUM covers the medium step-change regime, jointly spanning the changes relevant to behavioral-policy drift. All four parameters (`k`, `h`, `λ`, `L`) are configurable per deployment but the defaults above require no per-deployment retuning by virtue of the dimensionless unit system, which is itself a consequence of the variance-decomposition metric learner of §4.6 supplying the noise floor empirically rather than as an externally-supplied constant.

#### 4.8.2 Fragility map estimator (830)

Per anchor probe `A` and per perturbation kind `p`, the fragility map estimator computes the **mean per-axis compliance score delta** across the perturbation cloud entries of kind `p` at anchor `A`. The row of the fragility map at anchor `A`is the **local sensitivity gradient at `A`**: its largest-magnitude entry identifies the perturbation kind to which the monitored system is most fragile at that scenario, and the sign of that entry identifies whether the fragility manifests as compliance collapse (negative) or increase on compliance (positive).

The per-anchor sensitivity gradients constitute a two-dimensional fragility map indexed by (anchor, perturbation kind). The map is a complementary view of the same Δ data consumed by the input-perturbation Jacobian estimator of §4.6.1a, retaining the per-kind labels that the metric-tensor construction collapses. The map is recomputed at each measurement session and each cell's time series is independently subject to the SPC monitoring of §4.8.1, producing the fragility-signal alerts of §4.8 when a cell's value drifts beyond its historical control limit. The map is consumed by the drift localiser (Component 910) to identify the worst-offending perturbation directions per anchor for downstream contrastive-pair extraction (§4.9.2).

#### 4.8.3 Decoupling signal computer (840)

The decoupling signal is complementary to the drift signal of §4.8.1 (which tracks per-axis means) and the fragility signal of §4.8.2 (which tracks mean per-perturbation Δs): it exposes failure modes in which per-axis means and mean perturbation impacts are stable but the cross-axis correlation structure has shifted, indicating that the monitored system's previously coherent multi-axis behavior has changed even though no per-axis signal has drifted enough to alert.

For each anchor probe A, each perturbation kind p, and each pair of policy axes (i, j), the decoupling signal computer maintains a time series of per-cell Pearson correlation coefficients. At each measurement session, the value of cell (A, p, i, j) is the Pearson correlation between the per-variant compliance-score deltas on axis i and the per-variant compliance-score deltas on axis j, computed across the variants of the perturbation cloud associated with (A, p) at that session. Cells (ahchors) whose perturbation cloud contains fewer than ten variants are reported as unavailable.

Over an initial baseline window of T_baseline measurement sessions of stable operation (default ten sessions), the computer accumulates two per-cell summary statistics from the correlation values observed during the window: the mean correlation across the baseline sessions, and the standard deviation of the correlation across those same sessions. At each subsequent session, the computer compares the cell's current correlation against its baseline mean. An alert fires when the absolute difference between the current correlation and the baseline mean exceeds three times an effective standard deviation, where the effective standard deviation is the larger of (a) the per-cell baseline standard deviation accumulated during the baseline window, and (b) a configured floor value (default 0.05). The floor prevents spurious alerts in cases where the baseline window happens to produce a near-zero standard deviation. The alert payload carries the anchor identifier, the perturbation kind, the axis pair, the baseline mean correlation, and the current correlation value, so that downstream consumers can interpret both the magnitude and the direction of the structural change. Some embodiments may apply the Fisher z-transform `arctanh(r)` before the SPC test to stabilize the variance at correlations near +1 or -1.

In embodiments where the multi-sample stochasticity driver (Component 1000) is configured to apply at perturbation-cloud variants in addition to anchor probes, the per-cell baseline mean and standard deviation may be established from a single measurement session via resampling of the per-variant Δ vector distribution, rather than requiring an initial baseline window of T_baseline sessions. This embodiment trades supervised-LLM call cost for baseline-establishment latency: monitoring activates after one session rather than after T_baseline sessions, but at the cost of more supervised-LLM calls per session.

#### 4.8.4 Curvature signal computer (850)

For each anchor at each fit interval of Component 620, the computer evaluates the Reimmanian metric tensor at the anchor's compliance-space coordinate and computes a scalar summary of how anisotropic that local metric has become, generating an alert when the anisotropy has grown materially relative to the anchor's historical baseline.

##### Inputs

The local metric tensor `g(c_A)` at the anchor's compliance-space coordinate `c_A` is supplied by Component 620 (the metric MLPs of §4.6.2). Here:

- `c_A` is the `k`-dimensional compliance-score vector produced by the supervised LLM at the anchor (one entry per policy axis), and identifies the point in score space at which the metric is evaluated.
- `g(c_A)` is the `k × k` symmetric positive-definite matrix output by Component 620 at that point. By construction (§4.6), it captures the local "stretch" of compliance-score space at `c_A`: how much the compliance-score vector moves under unit perturbations of the input prompt.
- The default choice is `g_total(c) = g_input(c) + g_stoch(c)`, the pointwise sum of the input-sensitivity and stochasticity metric fields, which captures the full variance budget at the anchor. Alternative embodiments use `g_input(c)` (input-sensitivity geometry only) or `g_stoch(c)` (sampling-stochasticity geometry only).

##### Anisotropy via the condition number

Any `k × k` symmetric positive-definite matrix `g` admits an eigendecomposition that produces `k` real positive eigenvalues `λ_1 ≥ λ_2 ≥ ... ≥ λ_k > 0` and corresponding orthogonal eigenvectors `v_1, v_2, ..., v_k`. Each eigenvalue measures how strongly the metric "stretches" space along the corresponding eigenvector direction:

- `λ_max := λ_1` is the largest eigenvalue, corresponding to the direction of greatest stretching, that is, the direction in score space along which input perturbations produce the largest compliance-score movements at this anchor. The corresponding eigenvector `v_1` identifies *which combination of policy axes* this direction is.
- `λ_min := λ_k` is the smallest eigenvalue, corresponding to the direction of least stretching, that is, the direction along which the model is most stable to input perturbation.

The **condition number** of the metric is the ratio `κ(g) = λ_max / λ_min` which is a single dimensionless scalar summarising the metric's anisotropy.

##### Baseline establishment and alarm rule

A baseline condition number `κ_baseline(A)` is established per anchor over an initial stable-operation window, it is calculated as the mean of the `κ` values observed at that anchor during baseline. At each subsequent fit interval `t`, the **relative increase** `Δκ_rel(A, t)` of the condition number relative to baseline is calculated. An alert fires when the relative increase is higher than a κ_threshold with `κ_threshold` being a configured parameter (default 0.5, meaning the condition number has grown by 50% relative to baseline).

When an alert fires, the payload carries the anchor identifier `A`, the baseline and current condition numbers `κ_baseline(A)` and `κ(g(c_A, t))`, the relative increase `Δκ_rel(A, t)`, the **dominant eigenvector** `v_1` of `g(c_A, t)` corresponding to `λ_max`, expressed in score-axis coordinates.

The dominant eigenvector is operationally significant because it identifies *which combination of policy axes* the anchor has become disproportionately fragile along. For instance an alert that says "anchor `Chest Pain in a 65 year-old male' has become 70% more anisotropic, with the dominant fragility direction now aligned 80% with the uncertainty-expression policy axis" is consumed by the contrastive pair extractor (Component 920) to construct evidence pairs along the identified fragility direction, and by the drift localiser (Component 910) to prioritise this anchor for remediation ahead of anchors with lower anisotropy.

### 4.9 Feedback Subsystem (Component 900)

#### 4.9.1 Drift localiser (910)

Per measurement session, the drift localiser ranks all anchor probes by their current Riemannian distance to the policy boundary, computed via Component 630 under the metric `g_total(c)` (alternative embodiments use `g_input` or `g_stoch`). The Riemannian formulation, rather than Euclidean, ensures that anchors in high-fragility regions are flagged as more dangerous than equidistant anchors in low-fragility regions, since a unit Euclidean step under a high-eigenvalue metric covers more boundary-crossing risk than the same step under an isotropic metric.

Within each top-ranked anchor, the drift localiser then identifies the top-n worst-offending perturbation kinds by calculating the **Riemannian distance** using the Riemmanian distance computer (Component 830). This makes perturbation kinds whose mean impact aligns with the anchor's dominant fragility direction (the high-eigenvalue directions of `g_total(c_A)`) are surfaced as highest-severity, while perturbations of equivalent Euclidean magnitude along low-eigenvalue (stable) directions are correctly ranked as less severe.

#### 4.9.2 Contrastive pair extractor (920)

For each anchor surfaced by Component 910, the pair extractor constructs contrastive evidence pairs of two kinds drawn from output texts persisted in the score store:

- **Temporal pair**: the supervised-LLM output text for anchor `A` at a baseline session `t_0` and at the current session `t_1`, with their   compliance scores and the Riemannian drift between them. Captures temporal drift evidence and drives corrective remediation.

- **Spatial pair**: the supervised-LLM output text for anchor `A` at the current session and the output text for the top-n  perturbed variants, with the perturbation-kind label, the textual description of the perturbation applied, and the per-axis score delta. This captures static fragility evidence and drives robustness-oriented remediation. The perturbed variants are selected by Component 910 favouring perturbation kinds whose Δs direction aligns with the most fragile direction (dominant eigenvector) in the Riemmanian spacer for that anchor.

Pair extraction is bounded: the top K anchors from Component 910's ranked list contribute one temporal pair each and one spatial pair per top worst-offending perturbation kind, producing an evidence bundle of typically 12-20 pairs per session. The bundle is the input to the feedback synthesizer of §4.9.3.

#### 4.9.3 Feedback synthesizer (930)

It is based on a language model, instantiated separately from the supervised LLM and optionally drawn from a different model family, invoked each time Component 920 produces an evidence bundle. The Feedback synthesizer's input prompt comprises two categories of content:

- **(i)** the natural-language specification of the policy
- **(ii)** the contrastive evidence pairs produced by Component 920 with their associated output texts, compliance score vectors, perturbation-kind labels, and textual perturbation descriptions

The synthesizer's output is a natural-language remediation message tailored to the failure mode evidenced by the bundle: corrective for temporal-drift evidence (instructing the monitored system to align future responses with the earlier compliant behavior, citing the temporal pair contrasts), robustness-oriented for static-fragility evidence (instructing the monitored system to respond consistently across the identified perturbation kind, citing the spatial pair contrasts), or combined when both kinds of evidence are present at the same anchor.

The separation of the synthesizer model from the supervised model is important because a synthesizer sharing the supervised model's biases would produce remediations aligned with those biases and fail to correct them.

#### 4.9.4 Feedback delivery channel (940)

One or more delivery channels selected per deployment configuration, each routing the synthesized remediation message to its destination:

- **System-prompt injection.** The remediation message is appended to or merged with the monitored system's system prompt, taking effect at the next supervised-LLM invocation. This channel is appropriate for deployment contexts permitting autonomous closed-loop behavior   modification, such as research environments and explicitly self-correcting systems.

- **Operator report.** The remediation message is formatted as a structured report and delivered to a human operator via the deployment's configured notification channel. The operator's decision (apply, modify, defer, dismiss) is logged as part of the audit trail. This channel is appropriate for deployment contexts requiring human review of behavior modifications, such as clinical decision support under regulatory oversight.

A deployment may activate one channel exclusively or both with configured escalation rules, for example, low-severity remediations
auto-injected and high-severity remediations routed to operator review. The channel selection is part of the deployment configuration alongside the policy specification and rubric, and varies independently of the monitoring system's codebase.

#### 4.9.5 Effectiveness measurement (950)

After a remediation has been delivered, the effectiveness measurement subsystem measures whether the remediation has produced the intended behavioral change. The subsystem re-runs the affected anchor probes and their perturbation clouds through the supervised LLM and obtains new compliance scores. Three complementary effectiveness metrics are computed from the re-measurement:

- **(a) Compliance recovery toward the safe interior.** The signed Riemannian distance, evaluated under `g_total(c)` via Component 630, from each affected anchor's pre-remediation coordinate to its post-remediation coordinate, projected onto the direction away from the policy boundary. Positive values indicate movement back toward compliance; negative values indicate further drift. Aggregated across affected anchors as mean and worst-case values.

- **(b) Reduction in local Riemannian curvature.** The change in the condition number `κ(g(c_A))` at each affected anchor, evaluated post-remediation against the pre-remediation baseline. A reduction indicates that the local fragility geometry has become less anisotropic, in other words, that the model is no longer concentrating its sensitivity along a single dominant direction.

Each metric is compared against a configured improvement threshold. The remediation is classified as **effective** when it meets thresholds and **ineffective** otherwise. The classification, per-metric values, affected anchors, and original remediation message are persisted to the score store, indexed by the session of remediation delivery, building a longitudinal record of remediation outcomes.

### 4.10 Multi-sample stochasticity driver (Component 1000)

For each anchor probe it sends the **identical anchor scenario** through the supervised model `N` times (`N ≥ 10`), capturing `N` independent response texts and routing each through the compliance scorer to obtain `N` per-axis score vectors. Persists each of the `N` invocations as a distinct row in `compliance_scores` with a new `probe_role = "stochasticity_sample"` label and a session reference, enabling Component 610b to consume the resulting cluster of same-prompt score vectors as the rows of `J_stoch`. In embodiments that decorrelate LLM-as-judge stochasticity from supervised LLM-sampling stochasticity (see §4.6.1b), the driver additionally routes each of the `N` responses through the judge model `K` times (`K ≥ 3`) and persists the per-response judge mean rather than each individual judge score.

---

## 5. Embodiments and Variants

The detailed description above describes one operational embodiment. The components admit numerous variant implementations, and the system as a whole can be embodied at any point in the cross product of these variants. In particular:

- **The output-text encoder** (Component 221) can be any clinically or domain-tuned transformer encoder, or any general-purpose sentence encoder; the reference implementation uses a 768-dimensional clinically-tuned encoder but the system is invariant to the encoder choice up to the rebuild of all downstream artefacts.
- **The Stage-2 regression head** (Component 222) can be any function approximator; the reference implementation supports Ridge regression and a small Multilayer Perceptron (MLP).
- **The judge model and the validator model** (Component 210, Component 700) can be any language model with structured-output capability; they can be the same model or different models, with cost/quality tradeoffs configurable per deployment.
- **The kernel choice for the Gaussian Process** (Component 530) can be any of the three described kernels (stationary, non-stationary Gibbs, or Riemmanian pullback-metric) or any other kernel meeting the positive-definiteness and continuity requirements; further kernel variants can be substituted without altering the surrounding subsystems. The pullback-kernel embodiment is restricted to the response-side Gaussian Process fit (Component 500-R), since the Stage-2 scorer is defined over response embeddings; the input-side Gaussian Process fit (Component 500-I) is restricted to stationary or non-stationary kernels.
- **The choice of metric supplied to the Riemannian pullback kernel** can be any of `g_input`, `g_stoch`, or `g_total = g_input + g_stoch`. The reference embodiment defaults to `g_total` because it represents the full variance; embodiments restricted to input-sensitivity probing use `g_input`; embodiments focused on inherent-stochasticity probing use `g_stoch`.
- **The Gaussian-process layer's dual-fit arrangement** (Component 500-I and Component 500-R, §4.5) admits a single-fit reduction in deployments where the input/output type-mismatch is empirically negligible (e.g., where Bio_ClinicalBERT or an equivalent encoder produces statistically indistinguishable representations of clinically equivalent prompts and responses). In such deployments the response-side fit alone is operative and the synthesizer's input-side similarity step is performed against response-side embeddings of library anchor scenarios. Conversely, embodiments that operate three or more fits (e.g., one over prompts, one over responses, one over `(prompt, response)` joint encodings) are within the scope of the disclosure.
- **The Riemmanian metric learner architecture** (Component 620) can be any function approximator producing a Cholesky parameter vector; the reference implementation uses a small Multilayer Perceptron, but a Gaussian-process-over-metric construction or a transformer-based predictor are within the scope of the disclosure. The two MLPs producing `g_input` and `g_stoch` may share an architecture but are trained independently; embodiments that share parameters via a multi-head MLP (one shared trunk, two Cholesky-output heads) are within the scope of the disclosure.
- **The Riemannian distance computation** (Component 630) can be replaced by a true geodesic ODE solver in some embodiments; the segment-integration form is one operational embodiment.
- **The curvature proxy** (Component 850) can be replaced by the Ricci scalar in embodiments where the metric field is sufficiently smooth; the condition-number form is one operational embodiment.
- The **Stage-2 scorer audit threshold**, the **scorer calibration Spearman correlation threshold**, the **scorer calibration Mean Absolute Error threshold**, the **probe-synthesis cosine similarity threshold**, the **CUSUM and EWMA control limits**, and the **curvature signal condition-number change threshold**, the **feedback effectiveness signed Riemannian distance threshold**, and the **feedback effectiveness local Riemannian curvature threshold** are all configurable per deployment.
- **The persistent score store** (Component 400) can be any relational database store; the system is invariant to the storage backend.

---

## 6. Points of Novelty

Each numbered point below identifies an aspect of the arrangement believed to be distinct from prior art at the time of disclosure. 

### 6.1 Output-only learned compliance geometry

The system constructs and operates on a **learned position-dependent metric tensor field over a compliance-score space derived from black-box behavioral observation of an LLM**. The metric is fitted on empirical Jacobians (compliance score changes per unit input perturbation and LLM-internal stochastic response variability) collected via systematic perturbation experiments on a fixed anchor probe library, with positive-definiteness guaranteed by Cholesky parameterisation. Prior art in Riemmanian metric learning operates on directly observed data manifolds; the application of Riemmanian metric learning to a compliance-score space derived from output-only observation, with Jacobians estimated from controlled-input perturbation, is believed to be novel.

### 6.2 Closed-loop boundary-seeking probe synthesis

The system combines **Gaussian Process posterior uncertainty** with **proximity to the violation boundary** via the multiplicative weight `posterior_std × max(0, 1 − 2·|0.5 − posterior_mean|)`, applied over a candidate pool constructed by **quartile-stratified seeding** from the training distribution. The combination explicitly counters the production-distribution skew toward the dominant compliance mode and concentrates probe-generation effort where the monitor's classification is operationally more critical. Prior art in active learning for classification typically maximises predictive variance or entropy alone; the boundary-seeking weight and the stratified-seeding correction together address a problem that uniform-uncertainty active learning cannot solve in heavily-imbalanced regulated-decision-support settings.

### 6.3 Behaviorally Anchored Rating Scale rubric with framework-owned aggregation

The Stage-1 LLM-as-judge architecture **separates judgment from aggregation**: the judge model selects, per sub-condition, one identifier from an enumerated list of behaviorally anchored labels each carrying a pre-declared numeric value, and the framework computes the aggregate as a version-controlled weighted sum. Prior art in LLM-as-judge approaches typically requests a single overall numeric score, coupling judgment to aggregation in a way that prevents independent recalibration. The pre-declared label-to-value mapping plus framework-owned aggregation moves the calibration handle into a version-controlled artefact and enables (i) JSON-schema enforcement of judge responses against an enumerated label set, (ii) independent recalibration of the aggregation weights without re-querying the judge model, and (iii) post-hoc isotonic-regression recalibration against a hand-scored reference corpus while preserving the judge's ordinal ranking.

### 6.4 Embedding-space target verification as a substitute for constrained text generation

The probe synthesizer (Component 700) achieves controlled near-boundary text generation **without a separately-trained constrained text generator**, by combining (i) K-nearest-neighbor exemplar prompting with (ii) post-generation re-embedding and cosine-similarity verification against the original embedding-space target, (iii) bounded retries on similarity failures, and (iv) an LLM-as-validator quality gate. This allows to reuse pre-existing language generator and embedding models.

### 6.5 Four orthogonal monitoring signals on a shared learned geometry

The system produces four qualitatively distinct alert signal types: **drift** (CUSUM/EWMA on Riemannian displacement), **fragility** (perturbation cloud spread), **decoupling** (covariance-structure change between policy axes), and **curvature** (condition-number drift of the local metric). All them are computed over the same learned Riemmanian compliance-space geometry. Each signal detects a different failure mode and any subset can fire independently. The unified Riemmanian geometry substrate is what makes the four signals jointly interpretable.

### 6.6 Disambiguating monitor drift from monitored-system drift

The system maintains a **monitor-of-the-monitor channel** (Component 330) operating Statistical Process Control monitoring on the agreement statistics of the LLM-as-judge against a held-out hand-scored reference corpus, on a separate alert path from the drift / fragility / decoupling / curvature signals of Component 800. The architectural separation of the two alert paths enables disambiguation of "the supervised system has drifted" from "the compliance scorer has drifted", a distinction that cannot be made by any single-channel monitoring scheme.

### 6.8 Pullback of compliance-space metric to embedding-space active-learning kernel

The system warps the kernel of the embedding-space Gaussian process (Component 530, `riemannian_pullback` option of §4.5.3) with a metric tensor pulled back from compliance-score space through the Stage-2 compliance scorer (Component 222). For input embeddings `e_1`, `e_2` with score-space images `c_i = f(e_i)`, the kernel measures distance as `(c_1 − c_2)^T · g(m_c) · (c_1 − c_2)` where `g(c)` is the Riemannian metric tensor field of Component 620 and `m_c` is the score-space midpoint, augmented with a small isotropic embedding-space term to ensure positive-definiteness. This pullback is the mechanism by which the learned compliance Riemannian geometry of Component 600 is consumed not only by the monitoring signal subsystem (Component 800) and the feedback subsystem (Component 900) but also by the active-learning probe-synthesis path (Components 500 and 700).

The construction is believed to be distinct from prior art in Gaussian processes, because in our approach the Gaussian Process's input space (embedding space) is distinct from the Riemannian metric's domain (compliance-score space), and the pullback through the Stage-2 scorer is the bridge, contrary to previous Gaussian Proceses on Riemannian manifolds, where the Gaussian Porcess's input space coincides with the Riemannian manifold whose metric is consumed.

### 6.9 Dual-fit Gaussian-process arrangement with input-side proposal and response-side acceptance

The system instantiates two parallel Gaussian-process fits (Components 500-I and 500-R, §4.5) over the **same compliance-score labels** but distinct text corpora  (one over input-prompt embeddings and one over output-response embeddings) and consumes them asymmetrically: the **input-side fit proposes** embedding-space targets for the probe synthesizer; the **response-side fit accepts or rejects** synthesized probes after a supervised-LLM round-trip. The dual-fit arrangement gives each subsystem the Gaussian Process fit appropriate to the question it is asking.

Prior art in active learning with Gaussian processes typically operates a single Gaussian Process whose input space coincides with the synthesis substrate. The combination of same labels, distinct corpora, asymmetric consumption (proposal vs. acceptance) is believed to be novel.

### 6.10 Variance-decomposition metric learner

The system fits **two complementary metric tensor fields** over compliance-score space (Components 610a and 610b, §4.6.1): `g_stoch(c)` from multi-sample re-runs of identical input prompts (the LLM-stochasticity term) and `g_input(c)` from the perturbation cloud (the input-sensitivity term). The total fragility metric `g_total(c) = g_input(c) + g_stoch(c)` is the unique tensor consistent with the total variance of the score vector at coordinate `c`, and either term individually exposes a distinct failure mode: `g_input` localises **adversarial input fragility** (the LLM responds predictably but is brittle to wording changes); `g_stoch` localises **inherent prompt ambiguity** (the LLM is wording-stable but produces varied responses to the same prompt).

Prior art in information-geometric metric learning typically fits a single tensor field over a directly observed manifold; fitting two tensor fields whose sum reconstructs total observed score variance and whose individual eigenstructures expose qualitatively distinct failure modes is believed to be novel as applied to LLM compliance monitoring.

### 6.11 The arrangement as a whole

Independently of the per-component novelties, the **coordinated operation of components 100 through 1000** as a closed-loop monitoring apparatus in which the score store feeds the GP layer, the GP feeds the probe synthesizer, the synthesizer feeds the score store, the metric learner feeds both the GP layer (in the pullback-kernel embodiment of §6.8) and the monitoring signals, the monitoring signals feed the feedback subsystem, and the feedback effectiveness measurement feeds back into the score store is itself a novel arrangement of subsystems, in which each subsystem is required for the operational guarantees of the others.

---

## 7. Glossary

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
- **`g_total(c)`.** The pointwise sum `g_input(c) + g_stoch(c)`
- **Multi-sample stochasticity driver (Component 1000).** The experiment-controller subcomponent that sends an identical anchor scenario through the supervised LLM `N` times (`N ≥ 10`) and persists each invocation's per-axis score vector for consumption by the sampling-stochasticity Jacobian estimator.
- **Fragility.** The local sensitivity of compliance scores to input perturbations or LLM stochasticity at a given anchor.
- **Drift signal, fragility signal, decoupling signal, curvature signal.** The four orthogonal alert types produced by the monitoring signal subsystem.


