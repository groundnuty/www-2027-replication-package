# Deviations from the pre-registration

This document lists where the work differs from the registered design. It covers everything after the freeze. It has three parts: registered items that were not run, registered items that were changed, and items reported provisionally.

## How to read it

- Each bullet is tagged **[confirmatory]** if it touches the six registered claims, their runs or their analysis call. It is tagged **[exploratory]** if it touches only exploratory cells. A bullet that touches both says so.
- The registered claims, tests, calibration, multiplicity procedure, scorer, question sets, fault banks and model set were not changed after the freeze. The changes below concern how runs were served, which exploratory cells were run, and how some exploratory analyses are computed.
- Each change records the evidence it used. Most rest on schedule, capacity, cost or infrastructure counts (stop reasons, time-outs, run completeness), never on accuracy or on any outcome of the compared arms. One decision, to complete the closed-book probes, followed a logged look at their results. The record states that the look changed no rule. No other change consulted a test outcome.
- No result values appear here. They are in the results.
- **Not run:** registered, and no run or computation was made. **Not computed:** the data exist, but no analysis was made. **Not recorded:** the quantity was not logged during the runs. **Not reported:** a computation may exist in the analysis output, but the results do not tabulate it.
- Where a report is only in the analysis output files, it is not listed here. This applies to the registered strata and to several secondary columns.
- Interval labels: the registered tests are unstratified cluster bootstraps. Exploratory intervals, the fault tables and H6 use the unstratified cluster bootstrap. The descriptive intervals in the confirmatory call's output resample complete question vectors stratified by template and tier. Pooled-over-items differences in the fault tables are percentile intervals.

## 1. Registered items not run, not computed, not recorded or not reported

- **OBQC on the fault bank.** [exploratory] Not run. The baseline work after the freeze covered real outputs. The bank evaluation of this rule set was not scheduled.
- **The selective verifier.** [exploratory] Not run. It was not scheduled before the data cutoff. No estimate of a cheapest check subset is reported.
- **Source-layer ablations outside UniProt.** [exploratory] The layer-by-layer ablation (section 2) was run on UniProt only. It was not run on Rhea, opaque Rhea, the corporate graph or the private graph. It was scoped to UniProt first, and the other graphs were not reached before the cutoff.
- **The census at thresholds 2 and 20 on the private graph.** [exploratory] Not run there. It was replayed as a sensitivity on every other graph. The replay was not scheduled for the private graph.
- **SIB memorisation probes.** [exploratory] Not run. The probe scorer needs exact answers that most SIB gold answers cannot supply. Building probe records would also copy the held-out gold answers to more machines.
- **Closed-book probes, coverage gaps.** [exploratory]
  - Probes on an opaque UniProt graph were not run, because no agent evaluation ran on that graph.
  - Probe (b) was not run on the private graph.
  - GLM ran a seeded subset of 40 base questions per graph, not all 210. On the private graph it ran one attempt per question for that subset.
  - Replies cut at the output cap were re-run at a higher cap in some cells only. The other cells are reported with their cut counts.
  - The reason is capacity.
- **The LLM judge on the other graphs.** [exploratory] The judge was registered for every model's natural outputs on every graph. It ran on UniProt for all seven models, and on Sonnet's SIB runs. It was not run on Rhea, opaque Rhea, the corporate graph or the private graph. Each judged output needs a model call, and the cost and time did not fit.
- **The LLM judge on Qwen3.8's SIB outputs.** [exploratory] Not run, on either of its SIB runs. The judge was planned for Sonnet's SIB answers only. Under a cost cap it covers Sonnet's two reported SIB runs. Sonnet's companion runs were not judged.
- **The named baselines outside UniProt and SIB.** [exploratory] Not reported. Retry-on-empty, SPARQL-LLM's published validator and OBQC were built for UniProt and applied to SIB. Their results on Rhea, opaque Rhea, the corporate graph and the private graph are not in the results.
- **Baseline retry cells.** [exploratory] The retry cells of retry-on-empty, the judge and SPARQL-LLM ran on UniProt only (retry-on-empty and the judge for all seven models). They were not run on any other graph. OBQC as a retry trigger was not run. The scope was set to UniProt by a capacity decision.
- **Retry cells on the private graph.** [exploratory] Retries after any source's flags, the generic re-check and the deployment row were not run for any model there. The frozen runner cannot run a retry on that graph.
- **Prompted arms on the private graph.** [exploratory] Prompted arms for Haiku, GLM, DeepSeek and Qwen3-Coder were not run there. Gemma's prompted arms were not run either. These are Tier 3 cells, and capacity did not reach them.
- **GLM's opaque Rhea prompted and feedback phases.** [exploratory] These Tier 3 phases ran only in part, as seeded partitions, and are not in an analysis package.
- **Registered sensitivities not run.** [exploratory] The curated-plus-documentation, lossless-against-projected-drafts and superseded-drafts sensitivities were not run on any graph. They were not scheduled before the cutoff.
- **Registered sensitivities reported only in part.** [exploratory] The no-common-subclass-off variant is reported for H6 and for the census on the UniProt fault bank. The branch-local-flags-off variant is reported for H6. Other registered sensitivities are not reported unless a results table shows them. The adverse-assignment and refusal-excluded readings are computed in the confirmatory call and reported there. They are not reported for other graphs.
- **The per-model rate of invented identifiers.** [exploratory] Not computed. No analysis tool for it was built before the cutoff. The fault bank's catch rates for invented identifiers are reported.
- **H5's person-hours per mapped term.** [exploratory] Not reported. The annotator's hours were logged by hand, outside the harness, and no analysis reads them.
- **Checking time per answer and census build time.** [exploratory] Not recorded. The harness did not log them, so they cannot be rebuilt. H3's costs are reported in other terms: retry tokens and latency per flagged answer, triples read per checked answer, and the drafts' generation usage. The cost of the further retry cells per source was not recorded either.
- **Depth baselines (stretch goal).** [exploratory] Not run. They come first in the proposal's drop order.
- **H6's third model family (stretch goal).** [exploratory] Not run. The crossing uses the registered pair, plus Qwen3.8 on UniProt.
- **The exploratory models (Llama, Qwen3.6).** [exploratory] Not run. They were registered to run only if capacity remained.
- **H6's bank reference outside UniProt, and its mechanism witnesses.** [exploratory] The Rhea fault bank was not supplied to H6, and opaque Rhea has no bank. Generator recall on the bank is therefore UniProt only. The recorded witness for a premise shared by agent and checker was not built beyond the audit's shared-premise items (section 3).
- **H2 on the fault bank outside UniProt.** [exploratory] Not reported. H2 was registered on Rhea, opaque Rhea, the corporate graph and the private graph (recall per fault class, conformance apart from answer errors). The results do not report it.
- **Registered descriptive reports not computed on UniProt.** [confirmatory] Not computed: precision split by whether the mutated query returns rows; the natural drafts' per-question false-flag rates on the bank; the realisability descriptor; the split of untouched-gold false flags by whether a violation is witnessed; the mix of real agents' wrong constants beside the bank's. The conformance label is computed only as a flag-level proxy. No analysis tool for these was built.
- **Invented constants in a class position.** [confirmatory] The registered prediction (recall near 1 in a class position) cannot be tested. The test bank has no answer-changing invented constant of that kind. The cell is reported as empty. A separate labelled row reports the invented class and predicate operators as the closest testable analogue.
- **Re-runs for time-out and cut-call sensitivity, six models.** [confirmatory beside the registered result] A rule fixed after the freeze called for re-running every touched attempt. It was completed for GLM only. For the other six models the results report the original attempts, with the exclusion sensitivity beside them. Per-arm values and intervals for the exclusion were not computed.
- **The conditional deployment sensitivity.** [both] A rule fixed after the freeze restricts a contrast to the model's reference deployment when its arms' deployment shares differ by more than 10 points. It was computed only for one model's UniProt contrast, as a descriptive table. It was not computed elsewhere. Deployment shares are reported.
- **Per-deployment rate tables.** [both] The rates of capped, looping, cut-call and store time-out endings per deployment and arm, and each malfunction check's result, are not tabulated in the results.
- **The cap sensitivity on opaque Rhea.** [exploratory] Re-running every attempt with a turn at its output cap at 81,920 tokens was run for DeepSeek, and its results are not reported. It was not run for Qwen3.8.
- **Re-runs of opaque Rhea attempts that ran on a slower store copy.** [exploratory] Not done for GLM, whose no-ontology cell mixes copy classes. Not done for Qwen3.8 either, after an outcome-free replay of its time-outs on a fast copy. The contrasts are reported pooled and restricted to the partitions whose arms share one copy class.

## 2. Registered items changed after the freeze

### Serving and infrastructure

- **One model's runs on several serving deployments.** [both]
  - Registered: replicas must report the same served identity. A deployment that differs is a new stack, and it serves only graphs not yet started.
  - Changed: a model's running runs may take endpoints from other deployments when the engine release, the checkpoint and the served name match. A deployment that joins a running run first passes a short malfunction check on development questions. The check reads the share of capped and looping attempts and never accuracy. Candidates that failed the check were kept out of running runs.
  - Why: capacity became available while the runs were in progress.
  - A matching release and checkpoint is not behavioural equivalence. The contrasts are read under the serving conditions actually used. Each turn records its deployment, and the results report each arm's share per deployment.
- **Deployments that render prompts differently.** [both] A provider's gateway appears to add a short fixed text segment to every prompt. A gateway deployment and a direct deployment of the same checkpoint therefore differ in prompt tokens. Such deployments are never mixed within a run. Each run is served by one deployment, and the deployment per graph is disclosed. The same holds for one model's deployments that differ in chat-template revision.
- **Exceptions to one deployment per run.** [exploratory]
  - One model's private-graph arms continued on a second deployment of the same weights after the first stopped answering. Completed attempts were not re-run, and each arm switched once.
  - Retry cells and time-out re-runs for some models (GLM, Gemma, DeepSeek) ran on another deployment than their initial outputs. The initial deployment's fingerprints were admitted for the replayed turns.
  - GLM's attention implementation differs between its deployments (dense in some layers on some, sparse on others). Its cross-graph comparisons carry that difference.
  - The reason is capacity and outages. The decisions were made before any outcome of the affected arms was read.
- **Request time-outs per serving stack.** [exploratory]
  - Registered: a 600 s request time-out per model call.
  - Changed: on one model's dedicated stack the time-out was raised to 1,800 s. Long contexts filled the cache and the registered limit cut calls that were still progressing. The model's young exploratory runs on that stack were voided and restarted. This was decided before any outcome was read.
  - Later new runs use per-model time-outs sized to a full-length reply.
- **Run-time serving controls added.** [both] These change speed only, and each change is recorded.
  - Router policies were changed where they piled sessions onto one server.
  - A health monitor ejects a backend whose share of calls ending at the output cap exceeds the other backends' by more than 15 points. The backend returns only after a restart, an identity check and a fresh malfunction check.
  - The concurrency governor's latency baseline was pinned per model where the registered baseline misfired on long contexts.
  - Session counts changed during runs as capacity changed.
  - A released fault hold triggers one recorded restart of the run.
- **Store copies, speed classes and a second CPU architecture.** [both]
  - Registered: one store endpoint per model and run.
  - Changed for exploratory runs: several identity-checked store copies per graph, each listed with its identity proof. Confirmatory runs never changed copy before the confirmatory call.
  - On one graph, copies built for a second CPU architecture matched the others in results and row order, up to the engine's own run-to-run variation. On two graphs they differed in some unordered results. There, those copies serve only whole new runs, never mixed with the others. Unordered results with a row limit can differ between any two copies.
  - Copies are classed by timing a fixed heavy query set. The rule is that the arms of a run share one class. Where they did not, one model's affected natural attempts were re-run on a fast copy. In other cases (see section 1) a re-run was not done. The results then report the contrasts pooled and restricted to the partitions whose arms share a class, with time-out counts per arm and copy.
  - A few copies were killed by the operating system on very heavy queries because the memory limit of the allocation also held the index. Each such event is listed. The server flags were not changed.
- **Fault windows and re-runs.** [both]
  - Registered: a hold is confirmed by a frozen probe, and a confirmed window sets aside and re-runs the attempts in it, in every arm.
  - Added: the window rule was extended to store-server process deaths and to one file-system quota event. Only infrastructure logs were used, and the re-runs were outcome-blind.
  - The extension for store deaths was later retired for new deaths. The frozen store client already waits out a death and re-sends once, so no scored record can have run through one. Windows already applied stand.
  - A death triggered by an agent's own query opens no window.
  - Every run was audited for infrastructure faults before analysis, by a reviewed tool that reads infrastructure fields only.
  - Attempts holding a cut model call stand as scored.
- **Harness code after the freeze.** [both] The harness was re-tagged several times. The changes allowed serving through a front over several deployments, per-model request time-outs and turn caps, the baselines' cells, raised output caps, a window clamp (below) and a per-run selection for the probes. For each tag, the analysis files, scorer and inference digests were shown to be unchanged. Replays of existing records were identical. The confirmatory feedback phase was allowed to move to a re-tagged build only after a reviewed proof that every file on its path was unchanged or inert and that existing records replay identically.
- **Voided and superseded runs.** [exploratory] Some early runs were voided or superseded for capacity or infrastructure reasons before any outcome was read. They are kept sealed and are never analysed. This includes runs on a first deployment that was replaced, runs restarted under a changed time-out, an earlier run superseded by a re-planned one, SIB runs set aside after a cap change, and GLM's superseded judge runs. GLM's judge run was redone on one deployment because the frozen reader requires one route per run.

### Caps and clamps

- **Output caps for new runs.** [exploratory]
  - Registered: 16k output tokens per call, and measured limits for Qwen3.8, GLM and Qwen3.6.
  - Changed for new runs only: SIB runs and the cap re-runs use higher caps. Claude's cap rose to 32,000 and then to 128,000. The caps of Qwen3.8, DeepSeek, GLM and Qwen3-Coder rose to 81,920, with request time-outs sized to a full turn. Every existing run keeps its registered cap.
  - Why: a cap that ends more than 1% of a run's attempts is raised. The cap bound in SIB and in four running exploratory runs.
  - Opaque Rhea's raised turn cap was built after the freeze, as registered. Two Claude models' opaque runs started earlier and stayed at 120 turns.
- **Claude's SIB runs: substitution instead of a full repeat.** [exploratory] The rule fixed before the cap test called for a full repeat at 128,000 tokens if more than 1% of attempts were cap-bound. Both graphs met the condition. The full repeat was not run, because of the API budget and the deadline. The reported runs are the 32,000-token runs with each at-cap attempt replaced by its re-run at 128,000. The plain 32,000 runs are reported beside them.
- **A window clamp for Qwen3.8's SIB runs.** [exploratory]
  - At an 81,920-token cap the prompt window shrinks, because the server refuses a call whose prompt plus requested output exceeds the context window.
  - The fix: on exactly that refusal, the same call is sent once more without an output cap, and the server grants the smaller of the cap and the room left. Every call that fits is unchanged.
  - Attempts that ended at a cap were re-run with the clamp and substituted in the reported run. The originals are reported beside it. The substitution rule was fixed before the re-runs.
  - For the UniProt SIB run, the attempts not yet reached also ran under the clamp, in seeded partitions.
- **Closed-book probes.** [exploratory]
  - Registered: three repetitions for every graph and model.
  - Changed: probes became a must-complete item. They were filled by seeded top-ups that keep every finished call. Qwen3.8's and Gemma's full designs were re-run at raised caps. Qwen3.8's earlier runs at the lower cap are kept sealed as a labelled companion. Gemma's earlier calls are sealed and not reported.
  - Cut replies re-run at a higher cap are reported beside the originals and never pooled with them.
  - Output caps and request time-outs differ by model and are listed with the runs.

### Partitions, prefixes and cutoffs

- **Partitioned runs.** [exploratory]
  - Registered: one run directory per model, graph and cell, with one store endpoint.
  - Changed: several exploratory runs were split into seeded question partitions. The partition is a fixed hash of the question id with a fixed seed. Each partition ran on an identity-checked store copy, and its questions' earlier records were copied unchanged. This applies to several models' runs on Rhea, opaque Rhea and the private graph, and to Qwen3.8's UniProt SIB run.
  - Why: one store copy could not carry the sessions that model capacity allowed.
  - A pooled estimate comes from one clustered bootstrap over the union of the partitions' per-question vectors. Each question is counted once, and per-partition intervals are never combined.
- **The clean-prefix rule.** [exploratory]
  - Registered: cells not finished by the analysis are reported as not run.
  - Added: an unfinished partition or run enters the analysis as a clean seeded prefix. This is its attempts in their seeded order up to the first attempt with no final record. It enters only where the seeded order interleaves the arms being compared.
  - The prefix is fixed by the order alone and never by outcome. Its size is disclosed.
  - A sensitivity with every final record at the snapshot is reported beside the primary for the private graph's Gemma run.
- **A data cutoff for exploratory cells.** [exploratory]
  - Registered: no time cutoff. The confirmatory runs finished in full and were analysed after they finished.
  - Added for exploratory cells: a fixed data cutoff on the last day. Cells complete by then are analysed. A cell still running is analysed on its clean prefix. The rest is reported as not run.
  - The cutoff was moved once, to a later time on the same day, because of the schedule and before the affected cells' outcomes were read.
  - One run's snapshot was taken earlier than the cutoff so its checks could finish.
- **Early looks and a sanity monitor.** [both]
  - Registered: no interim looks.
  - Added: an hourly monitor ran over every cell with records, the confirmatory cells included. It reports only pass or flag with the criterion, never numbers or arm contrasts. The criteria are scorer errors, no submitted answer, unparseable answers, checker errors, implausible pooled accuracy and accuracy far from the model's development value. Exploratory results were looked at per completed phase, and each look was logged.
  - The confirmatory call ran on the frozen directories before any look at the confirmatory runs' results, other than the monitor's pass or flag. A planned interim look at them was dropped.
  - No re-run, setting, capacity or scheduling decision may cite a seen result. The audit's annotators do not see results or models.
- **Incidental exposures.** [both] Registered: exploratory components consult no test outcome before they are pinned, and operational decisions read infrastructure signals only.
  - On several occasions, builders, operators or the harness's own tools displayed infrastructure counts, per-directory progress counts, one result line or a few attempt lines.
  - On other occasions, held-out file names, listings or aggregate counts were displayed. Held-out gold files were also found outside the held-out tree and removed. In one case held-out blobs passed through a scanning tool's memory.
  - Each was reported when found. No held-out content or test outcome informed any decision, setting or selection.
  - The models under evaluation have no file access, so none could see held-out content.
  - The rules were tightened. Tools print allowlisted infrastructure fields only. Logs that could show counts are redacted. Scans exclude held-out paths. Builders may not list the held-out tree.

### The confirmatory call

- **The call's required arms.** [confirmatory]
  - The frozen analysis configuration required 37 arms. Four retry arms (two models, with the census and union retries) exist only in the separate exploratory run for UniProt's other feedback cells. The call would have refused to run.
  - Those four arms left the call's required set, and its descriptive retry table is empty for them. The rows come from the separate exploratory call.
  - The calibrated inference code was untouched. The grouping digest and the configuration were re-pinned.
  - A synthetic 37-arm development input showed identical outputs before and after, except the four rows and the declared digests. Two independent reviewers checked the change. The design was fixed before any outcome of the confirmatory runs was seen.
  - The call ran on the frozen directories. A confirming run on the completed directories reproduced every number macro. No estimand changed.
- **Time-out and cut-call diagnostics and sensitivities.** [confirmatory beside the registered result]
  - Added after the freeze: every store query that timed out is replayed alone on a quiet copy. A query that answers alone within the limit is load-induced. One that times out again is intrinsic.
  - Attempts touched by a load-induced time-out or a cut model call enter the sensitivities in every arm, whatever their result. The registered result is unchanged and is reported first.
  - For GLM the touched units were re-run in two rounds, on another deployment and, in round 2, at lower concurrency than registered. Round 2's units replace round 1's. There is no third round.
  - For all seven models the touched attempts are also excluded in a second sensitivity.

### Exploratory analyses

- **The layer-by-layer ablation.** [exploratory]
  - Registered: source-layer ablations, scoped before the freeze to vocabulary on and off and to the census's type checks against its join and pattern checks.
  - Changed: the layer-by-layer design the proposal describes was added. Layers are class checks, domain and range, disjointness, vocabulary existence, and the census's join and pattern checks. Rungs are cumulative, with and without vocabulary, and each source gets every layer it contains.
  - It is computed offline from the replay matrices, which record only the fired check kinds. "Flagged under a layer set" means one of its kinds fired. Pass and unknown per layer are not computed.
  - Filtering by check kind is not the same as removing an axiom kind. Domain and range axioms also feed the implied classes that the disjointness and no-common-subclass checks read.
  - A second grouping of the no-common-subclass check, outside the class layer, is reported for the census beside the frozen code's grouping.
- **H6 analysis rules.** [exploratory] These were fixed after the freeze and before any H6 output existed.
  - The interaction is computed only for two agents with one generator each. On UniProt's 3 × 3, every pairwise 2 × 2 is reported, plus each generator's own-minus-others recall. There is no pooled interaction.
  - Opaque Rhea uses two of DeepSeek's three drafts. Masked-name DeepSeek drafts were not run.
  - Every draft is replayed as drawn. A sensitivity drops any draft that declares no property. Every draft uses the frozen first-block extraction.
  - The within-run part has a pooled headline over the curated and census cohorts for all seven models, and a three-cohort reading for two models. A cell with fewer than 20 wrong outputs is called uninformative. There are no p-values and no mechanism claim.
- **Baselines: scope and implementation.** [exploratory]
  - The judge is each agent's own model. On the fault bank, the judge is Sonnet and Haiku.
  - SPARQL-LLM's validator is the published code with its one retry. Where its inference tries only the first class an index lists, list order decides the verdict. The order was fixed by IRI. One adapted, detection-only row is labelled as ours.
  - Retry cells are new run directories built from copied parents.
  - The judge on Sonnet's SIB answers ran through the provider's batch interface under a cost cap.
  - "Matched-information replay" is read as the replay of the generated drafts against the census on the same outputs.
- **SIB scoring and sets.** [exploratory]
  - Scoring rules were fixed before any SIB attempt was scored. SELECT golds use the main scorer. ASK golds are correct when the booleans match, or when a SELECT answer's non-emptiness matches. CONSTRUCT and DESCRIBE golds ask for a graph, so they are excluded from accuracy and counted.
  - The Rhea set's size differs from the proposal's by one: the proposal counted a prefixes file. An exclusion rule stated before the gold run removed examples with a SERVICE clause, a graph outside the store, or a non-SELECT query. Examples with an empty gold answer were removed afterwards by the UniProt rule.
  - Each question is its own cluster, because SIB's only anchors would come from held-out gold. SIB runs the no-ontology cohort only.
- **The bank replay was extended.** [exploratory] The frozen test-bank replay covered the curated source, the census at threshold 5 and the three masked drafts. After the freeze it also replays the natural drafts, the census at thresholds 2 and 20, and the descriptive type-consistent constant swaps. Conformance is derived per source as a flagged mutant whose untouched gold is not flagged. The confirmatory outputs are unchanged.
- **H5's replay on the private graph.** [exploratory] The replay path for the private graph was built after the freeze, under the frozen rules. A check the store cannot answer counts as unknown in both arms. H5 is also reported on the records where neither arm has such a check, as a paired sensitivity fixed before any test replay.
- **The audit's population and appendices.** [exploratory]
  - Registered: two annotators, a seeded stratified sample and a codebook.
  - Changed: the population is fixed by completion time, not by the end of all runs. The draw was moved to a later fixed time, before any package was handed out or any item seen. It is a new draw with the same seed, order, quotas and protocol.
  - Units that complete after the main draw and before a second fixed time get an appendix package. It uses the same seed, order and protocol, an allocation rule fixed before any appendix unit completed, disjoint numbering and its own key.
  - Each annotator gets one package with all items shuffled together. Cells that completed after a draw are excluded from it and reported.

### Analysis-side changes

- **The partition and prefix reader.** [exploratory] The analysis reads seeded partitions and clean prefixes. The rule fixed after the freeze takes per-question means over the repetitions present, with only questions with no attempt counted as "not run". It replaces a rule that kept only questions with all three repetitions, which would have discarded most questions. The subset is fixed by the order alone. For the private graph's Gemma run, the analysis as run instead counted each attempt outside the population as missing and scored it wrong. This is disclosed with the result, and that run's accuracy under the per-question rule was not computed. The reader also accepts the weighted partition assignment that run used, with the producer's own rule.
- **The store allowance in exploratory analyses.** [exploratory] The frozen analysis refuses a model whose records name two store endpoints. The exploratory calls accept several identity-checked copies per graph and model where each copy is listed with its identity proof. They skip the one-endpoint check for sensitivity calls and for partitions with copied parents. For plain partition unions, a recorded store change on a gated store is accepted when both sides name one endpoint and carry identical gate settings. Each copy's share of attempts per arm is reported.
- **Updates to the serving-identity tool.** [both] The admitted-fingerprints file lets a run accept several fingerprints. The probe's summary check relaxed from equality to "expected is a subset of recorded". Admission compares the public engine release and records the build label. Tier 1's fingerprints are admitted for replayed parent turns in the further retry cells. The stack-name check accepts partitions that share one assignment and one fingerprint. For the private graph's partitions, the identity tool reads each partition's own launch record, records the earlier starts as inherited from the source run, and names the parents' store from the single recorded store change.
- **The set-aside exclusion in Gemma's further retry cell.** [exploratory] The frozen analysis needs every scored output's row in every retry arm, and a set-aside retry has none. The parent attempt of each checker-divergence set-aside is dropped from every arm of that cell, so the arms keep one population. One attempt of 630 is dropped. Outputs without a set-aside are byte-identical to the frozen code's. The change had one independent review.
- **Fixes to the pooled SIB union analysis.** [exploratory] These apply to Qwen3.8's pooled SIB UniProt analysis, and were checked mechanically but not independently reviewed.
  - A guard that refused grid-layout batches was narrowed to refuse only real retry cells and ambiguous grids.
  - The stack-name check accepts the five partitions' differently named stacks because they share one assignment and one fingerprint.
  - A failing offline replay pass is repeated at the same seed up to 8 times, only on the reproduction failure, with every try logged.
  - The reason is the checker's run-to-run nondeterminism (below).
- **The store-proof bypass.** [exploratory] The analysis's store-proof bookkeeping check was bypassed for this one union run, with every other check on. The kept records' copy is proven for the source run only, and every copy was identity-checked when used. The decision was made so that these results could be part of the submission.

### The checker's nondeterminism and the reproduction retries

- **Nondeterminism in the frozen checker.** [exploratory]
  - The offline replay of one SIB record does not ask the same store queries on every run. A footprint query's choice of one term varies from run to run, even at a fixed hash seed. A code reading puts the likely cause in collections ordered by memory address while query text is built.
  - The reported verdict is that of the staged online replay. A failing offline pass writes nothing.
  - The analysis accepts a run only if its matrix and store log equal the staged ones. If more than one run passes, their results must be byte-identical. The submission count and the pass and fail counts of the record's reproduction are disclosed.
  - A checker fix applied to one partition after the freeze was rejected as the larger deviation. The frozen checker is unchanged.
- **Reproduction retries in Gemma's further retry cell.** [exploratory] The runner's exit check held records whose online decisions did not match the offline recompute, probably for the same reason, and the held set changed from one full check to the next. Each held record was re-checked alone in a fresh process, up to 8 times, with no store reads. A record that reproduced in at least one try was kept. One record that failed all 8 tries of its series was set aside under the registered checker-divergence rule; it had reproduced once in an earlier trial try outside that series. The cell was completed once every held record of a full check had reproduced. Every check, held set and try is logged.

## 3. Items reported provisionally

- **The two-annotator audit.** [exploratory] The two annotators coded independently. The registered adjudication by discussion was not completed before submission. The results report each annotator's codes and the codes both gave, with Cohen's κ per field on the primary code. No intervals are given. The tables built on adjudicated codes are not reported. Its numbers were checked by one independent verifier; it stays provisional until adjudication.
- **H4's residue and H6's mechanism.** [exploratory] Both rest on this audit alone. The residue of errors of meaning (H4) is read from its error-type codes. The audit's shared-premise items are the only evidence on mechanism. No mechanism is claimed. Both are provisional until adjudication.
- **Qwen3.8's pooled SIB UniProt results.** [exploratory] These are reported with the analysis-side changes in section 2. Their analysis code was not independently reviewed.
