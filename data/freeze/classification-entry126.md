# Freeze-registry classification (change-log entry 126, items 5 and 6)

Generated from the freeze check of `data/freeze/registry.json` (sha256 `04e638b4b0e3c2b9179c34b6830d92f56c81e9aef481eeb9e0c9706a7a171478`) against the science repo at `4e84c92180251ade68449c830f1a2273e38b4e97` (preregistration blob `adbd45e68efc`, change log blob `d0c6c0270dc3`), bindings at experiments `9e34f6839f33`.

Counts: match 100, mismatch 0, unresolvable 3, unresolved 10, ignored 1, historical 5 (the bindings included: check-contract match, frozen-config unresolved).

Classes: **binding**, the check recomputes the artefact and compares it with the cited value; **historical**, cited for the record, bound to nothing; **placeholder**, a [FILL] that fills at the freeze; **unregistered**, no full value in git to bind to.

## Pre-registration (sections 1-7): every hash binding

| Line | Cited | Class | Registry entry | Reason |
|---|---|---|---|---|
| 9 | `[FILL]` | placeholder | analysis-code-commit | fills at the freeze |
| 18 | `[FILL]` | prose about the marker | fill-marker-mention | Prereg preamble: 'to fill in the items marked [FILL]' describes the marker; it is not a placeholder. |
| 28 | `17eef284` | binding | scope-uniprot-frozen-set | Prereg section 1, the question population: the frozen UniProt round-2 set (the 210 agent-facing base questions), the file on the UniProt generator branch, the same artefact as the round-2 set in section 5. The … |
| 28 | `f11dff07` | binding | scope-extension-questions | Prereg section 1, the question population: the bank-only extension questions (entry 41), a separate file under the UniProt questions directory on the extension branch. |
| 33 | `[FILL: whether C1a is pooled over the panel, as C5 is; decided before the freeze from its power by the rule in §4.2, entry 44]` | placeholder | – | fills at the freeze |
| 62 | `87c7126` | binding | controls-calibrated-code-commit | Prereg section 2, C4's controls: the calibrated code's commit, which must be in the analysis branch's history (prereg.c4 is defined at it). The same commit the calibration entries cite. |
| 109 | `3d6b1e6afe70…` | binding | scorer-digest | Prereg section 3, accuracy: the sha256 of the scorer's closed module set (harness/score/digest.py; since 05492e6 it includes harness/score/gold.py), by the scorer's own scorer_digest() at integration/freeze 7df3699, … |
| 109 | `[FILL: tag]` | placeholder | – | fills at the freeze |
| 155 | `20,000` | binding | calibration-n1 | Prereg 4.2 and 5: N1, from harness/analysis/calibration.py at the calibrated tree. Read in the pre-registration only: the change log records earlier values (entry 35's N1 = 10,000) as history. |
| 156 | `100,000` | binding | calibration-n2 | Prereg 4.2 and 5: N2, likewise. Read in the pre-registration only: the change log records earlier values (entry 35's N1 = 10,000) as history. |
| 163 | `b805acd9` | binding | calibration-round1-questions | Prereg 4.2: the questions file calibration step 1's fingerprint names, the default --questions on the calibrated tree 87c7126 (round 1). |
| 163 | `17eef284` | binding | calibration-round2-questions | Prereg 4.2: the round-2 set the calibration's structure is compared with. |
| 164 | `47` | binding | calibration-configuration-count | The number of calibration configurations, from step 1's own output (its configuration list, which the fingerprint binds). |
| 165 | `[FILL]` | placeholder | – | fills at the freeze |
| 193 | `[FILL: the weighted-Holm replay on the final bank structure]` | placeholder | – | fills at the freeze |
| 196 | `[FILL]` | placeholder | – | fills at the freeze |
| 205 | `[FILL]` | placeholder | – | fills at the freeze |
| 214 | `17eef284d7c8…` | binding | uniprot-round2-questions | Prereg section 5, question sets: the UniProt round-2 questions file on the UniProt generator branch. The FROZEN marker on feat/e0.1-skeleton names the round-1 file and is not used. |
| 214 | `f11dff07f69d…` | binding | question-sets-extension | Prereg section 5, question sets: the bank-only extension questions, the same file as in section 1. |
| 214 | `b204846e7c20…` | binding | question-sets-rhea | Prereg section 5, question sets: the Rhea 0.4.3 questions file on the Rhea generator branch (change-log entry 75). |
| 214 | `36556c92858a…` | binding | question-sets-corporate | Prereg section 5, question sets: the corporate question set (the text2sparql-ck26 questions file) on the corporate generator branch. |
| 214 | `4153ad29e7d4…` | binding | question-sets-opaque-rhea | Prereg section 5, question sets: the opaque-Rhea questions file, the re-map of the Rhea set, on the remap branch. |
| 214 | `b204846e` | binding | question-sets-rhea-remap | Prereg section 5, question sets: the Rhea set named again as the source of the opaque-Rhea re-map (the same file as question-sets-rhea). |
| 214 | `e22bc7c63f1b…` | binding | question-sets-a15 | Prereg section 5, question sets: the a15 set, which is private; bound to the questions sha256 the committed a15 build record names (a build input, bound to that file and field). |
| 216 | `21a971d2` | binding | sources-request-uniprot-natural | Prereg section 5, sources: the generator's request hash, the request_sha256 of data/sources/uniprot-sp-2026_03/generated/input/bundle.json; the generator refuses a bundle whose prompt files do not hash to it … |
| 216 | `4775905b` | binding | sources-request-uniprot-masked | Prereg section 5, sources: the generator's request hash, the request_sha256 of data/sources/uniprot-sp-2026_03/generated-dc/input/bundle.json; the generator refuses a bundle whose prompt files do not hash to it … |
| 216 | `46b6958e` | binding | sources-request-rhea | Prereg section 5, sources: the generator's request hash, the request_sha256 of data/sources/rhea-142/generated/input/bundle.json; the generator refuses a bundle whose prompt files do not hash to it … |
| 216 | `abc79d4c` | binding | sources-request-opaque-rhea | Prereg section 5, sources: the generator's request hash, the request_sha256 of data/sources/opaque-rhea-142/generated/input/bundle.json; the generator refuses a bundle whose prompt files do not hash to it … |
| 216 | `6c25a55f` | binding | sources-request-corporate | Prereg section 5, sources: the generator's request hash, the request_sha256 of data/sources/text2sparql-ck26/generated/input/bundle.json; the generator refuses a bundle whose prompt files do not hash to it … |
| 217 | `3b389cf7e52d…` | binding | prompt-uniprot-curated-prereg | Prereg section 5, prompt renders (UniProt): cited as renders; these are the prompt files. The hash is of curated.prompt.txt (framing and rendered source, the bytes a run's prompt carries), not of the render's own … |
| 217 | `7d1b3299e21e…` | binding | prompt-uniprot-census-prereg | Prereg section 5, prompt renders (UniProt): cited as renders; these are the prompt files. The hash is of census.prompt.txt (framing and rendered source, the bytes a run's prompt carries), not of the render's own … |
| 217 | `950f366c438e…` | binding | prompt-uniprot-generated-1-prereg | Prereg section 5, prompt renders (UniProt): cited as renders; these are the prompt files. The hash is of generated.prompt.txt (framing and rendered source, the bytes a run's prompt carries), not of the render's own … |
| 217 | `10279bb3f156…` | binding | prompt-uniprot-none-prereg | Prereg section 5, prompt renders (UniProt): cited as renders; these are the prompt files. The hash is of none.prompt.txt, the base prompt with no ontology block. The path is on the prompt-conditions branch. Since … |
| 217 | `76f85b7d` | binding | prompt-rhea-curated-prereg | Prereg section 5, prompt renders: the Rhea curated render at B = 27,000 (data/prompts/rhea-142/curated.prompt.txt), on integration/freeze (change-log entries 88 and 109). |
| 217 | `4d016357` | binding | prompt-rhea-census-prereg | Prereg section 5, prompt renders: the Rhea census render at B = 27,000 (data/prompts/rhea-142/census.prompt.txt), on integration/freeze (change-log entries 88 and 109). |
| 217 | `4f29e596` | binding | prompt-rhea-generated-prereg | Prereg section 5, prompt renders: the Rhea generated render at B = 27,000 (data/prompts/rhea-142/generated.prompt.txt), on integration/freeze (change-log entries 88 and 109). |
| 217 | `10279bb3` | binding | prompt-rhea-none-prereg | Prereg section 5, prompt renders: the Rhea none render at B = 27,000 (data/prompts/rhea-142/none.prompt.txt), on integration/freeze (change-log entries 88 and 109). |
| 217 | `f6886580` | binding | prompt-rhea-manifest-prereg | Prereg section 5, prompt renders: the Rhea manifest render at B = 27,000 (data/prompts/rhea-142/manifest.json), on integration/freeze (change-log entries 88 and 109). |
| 217 | `192c834d9af5…` | binding | prompt-opaque-rhea-curated-prereg | Prereg section 5, prompt renders: the opaque-Rhea curated render at the same B (data/prompts/opaque-rhea-142/curated.prompt.txt), on integration/freeze. |
| 217 | `4e8e6ee7579c…` | binding | prompt-opaque-rhea-census-prereg | Prereg section 5, prompt renders: the opaque-Rhea census render at the same B (data/prompts/opaque-rhea-142/census.prompt.txt), on integration/freeze. |
| 217 | `d5d8698efaff…` | binding | prompt-opaque-rhea-generated-prereg | Prereg section 5, prompt renders: the opaque-Rhea generated render at the same B (data/prompts/opaque-rhea-142/generated.prompt.txt), on integration/freeze. |
| 217 | `10279bb3` | binding | prompt-opaque-rhea-none-prereg | Prereg section 5, prompt renders: the opaque-Rhea none render at the same B (data/prompts/opaque-rhea-142/none.prompt.txt), on integration/freeze. |
| 218 | `[FILL]` | placeholder | contract-commit | fills at the freeze |
| 218 | `[FILL]` | placeholder | compiler-commit | fills at the freeze |
| 219 | `5ca011cb8fe7…` | binding | feedback-format-v3 | Prereg section 5: the checker-feedback-v3 format hash (canonical JSON of FORMAT_SPEC), by the checker loop's own format_sha256(). |
| 219 | `ae3150da` | binding | generic-retry-text | Prereg section 5, online retry settings: the generic retry text. Recomputed by generic_sha256() from GENERIC_RETRY_TEXT in the checker loop's code, never from the text quoted in the document. |
| 222 | `bd17577eeea5…` | binding | fault-bank-uniprot-manifest | Prereg 5: the frozen UniProt fault bank's manifest (design log, change-log entry 83), frozen before any checker ran on it. |
| 222 | `8b05bff44dd1…` | binding | fault-bank-rhea-manifest | Prereg 5: the frozen Rhea fault bank's manifest (design log, change-log entry 83); exposure marked by query. |
| 223 | `20,000` | binding | calibration-n1 | Prereg 4.2 and 5: N1, from harness/analysis/calibration.py at the calibrated tree. Read in the pre-registration only: the change log records earlier values (entry 35's N1 = 10,000) as history. |
| 223 | `100,000` | binding | calibration-n2 | Prereg 4.2 and 5: N2, likewise. Read in the pre-registration only: the change log records earlier values (entry 35's N1 = 10,000) as history. |
| 223 | `0e9232e9f3d0…` | binding | calibration-step1-json | Calibration step 1's output (alpha' per claim and level), committed on feat/analysis (6237ca1). |
| 223 | `47` | binding | calibration-configuration-count | The number of calibration configurations, from step 1's own output (its configuration list, which the fingerprint binds). |
| 223 | `numpy defaul…` | binding | calibration-seed-scheme | The seed scheme string, from calibration.SEED_SCHEME. Read in the pre-registration only: the change log records earlier values (entry 35's N1 = 10,000) as history. |
| 223 | `20261101 and…` | binding | calibration-seeds | The two step seeds (steps 1 and 2, in that order), from calibration.SEEDS. A single seed (the analysis's 20261026) is not read. Read in the pre-registration only: the change log records earlier values (entry 35's N1 … |
| 223 | `770f902551a2…` | binding | calibration-fingerprint | The calibration fingerprint, by scripts/calibrate.py's own setup() and fingerprint() at the calibrated tree 87c7126, with step 1's inputs: the bank items on SiteA (not in git; read with ssh + tar, read-only, and … |
| 223 | `81708e646509…` | binding | calibration-code-digest | The plan-D code digest of the calibrated tree 87c7126 (harness/ and scripts/), by harness.analysis.digest.code_digest() there. Only on a line about the calibration: the final main's code digest is another artefact. |
| 223 | `87c7126` | binding | calibration-commit | Prereg 5: the calibrated commit beside the calibration's code digest; must be reachable from the analysis branch (main at the freeze). |
| 224 | `17eef284` | binding | calibration-table-uniprot-frozen-set | Prereg section 5, calibration: the UniProt set frozen by change-log entry 68, the round-2 file on the UniProt generator branch. |
| 224 | `b204846e` | binding | calibration-table-rhea-frozen-set | Prereg section 5, calibration: the Rhea set frozen by change-log entry 75, the questions file on the Rhea generator branch. |
| 224 | `36556c92` | binding | calibration-corporate-set | Prereg section 5, calibration: the corporate set (the text2sparql-ck26 questions file), frozen without a calibration round (change-log entry 109). |
| 224 | `7a3454f` | binding | calibration-corporate-commit | Prereg section 5, calibration: the commit the corporate set was frozen at, reachable from the corporate generator branch. |
| 225 | `dc93f8df62de…` | binding | store-qlever-image | Prereg section 5, stores: the QLever image digest, the sha256 in the 'image' field of the Rhea store's committed build manifest. The build record is the only place it exists in git; the check binds the document to … |
| 225 | `9bebc60` | binding | store-qlever-git | Prereg section 5, stores: the QLever git commit, the hash after 'qlever-index' in the 'qlever_version' field of the Rhea store's committed build manifest. The opaque UniProt manifest records the same commit in a … |
| 225 | `9484989e47e8…` | binding | store-identity-uniprot | Prereg section 5, stores: the UniProt subset. Identity = sha256 of the store's committed build/index.sha256 list (uniprot-sp-2026_03); the leading backtick keeps the opaque names from also naming the plain ones. |
| 225 | `c0174e728db6…` | binding | store-identity-rhea | Prereg section 5, stores: Rhea. Identity = sha256 of the store's committed build/index.sha256 list (rhea-142); the leading backtick keeps the opaque names from also naming the plain ones. |
| 225 | `c454dc75b2d5…` | binding | store-identity-opaque-rhea | Prereg section 5, stores: opaque Rhea. Identity = sha256 of the store's committed build/index.sha256 list (opaque-rhea-142); the leading backtick keeps the opaque names from also naming the plain ones. |
| 225 | `37898ec62716…` | binding | store-identity-opaque-uniprot | Prereg section 5, stores: opaque UniProt. Identity = sha256 of the store's committed build/index.sha256 list (opaque-uniprot-sp-2026_03); the leading backtick keeps the opaque names from also naming the plain ones. |
| 225 | `c51a9638d28f…` | binding | store-identity-corporate | Prereg section 5, stores: the corporate graph. Identity = sha256 of the store's committed build/index.sha256 list (text2sparql-ck26); the leading backtick keeps the opaque names from also naming the plain ones. |
| 291 | `7f804abe` | binding | c3-census-input | Prereg section 7, C3's reading: the census input of the bank and the prompts, census-t5 (data/sources/uniprot-sp-2026_03/census-t5.json) on the sources branch. |

## Change log

| Line | Cited | Class | Registry entry | Reason |
|---|---|---|---|---|
| 50 | `b805acd9` | binding | changelog-15-regenerated | Change-log entry 15, the regenerated file after the guard fixes. Located by provenance: commit 31a5cde ('freeze the UniProt question set'), the commit after the guard fix ad85b72. |
| 50 | `aaf797f7` | binding | changelog-15-calibrated | Change-log entry 15, the file calibration round 1 ran on. Located by provenance: commit e92a147 ('update the UniProt question set to the audited version'). |
| 66 | `d1278589` | binding | changelog-31-audited | Change-log entry 31, the round-2 file the independent audit read. Located by provenance: commit 99e6fd4 ('UniProt calibration round 2'), the review snapshot snap-qgen-r2, before the audit fixes in 2d53607. |
| 113 | `d1278589` | binding | changelog-54-audited | Change-log entry 54: the audited round-2 file, the same artefact as entry 31 (commit 99e6fd4, by provenance). |
| 114 | `2d53607` | binding | changelog-54-activity-quantifier | Change-log entry 54: the commit that changed the activity question's gold quantifier; it must be in the UniProt generator's history. |
| 115 | `2ecc58e` | binding | changelog-54-activity-annotated | Change-log entry 54: the commit that requires an annotated catalytic activity; in the UniProt generator's history. |
| 116 | `0b247dd` | binding | changelog-54-anchor-redraws | Change-log entry 54: the anchor-redraw commit (entry 42); in the UniProt generator's history. |
| 118 | `17eef284` | binding | changelog-54-round2-set | Change-log entry 54, 'Filled': the round-2 set, the same artefact as the prereg's uniprot-round2-questions. |
| 118 | `5ca011cb` | binding | changelog-54-feedback-format | Change-log entry 54, 'Filled': the feedback format hash, the same artefact as feedback-format-v3. |
| 156 | `b204846e` | binding | changelog-65-rhea-final | Change-log entry 65: the final Rhea round-2 set (generator 0.4.3) on the Rhea generator branch. |
| 169 | `17eef284` | binding | changelog-68-uniprot-frozen | Change-log entry 68: the UniProt set frozen after calibration round 2. |
| 194 | `b204846e` | binding | changelog-rhea-frozen | Change-log entry for Rhea calibration round 2: the Rhea set frozen after round 2 (generator 0.4.3) on the Rhea generator branch. |
| 218 | `bd17577e` | binding | changelog-83-uniprot-bank | Change-log entry 83: the frozen UniProt fault bank's manifest. |
| 218 | `8b05bff4` | binding | changelog-83-rhea-bank | Change-log entry 83: the frozen Rhea fault bank's manifest. |
| 249 | `3b389cf7` | binding | prompt-uniprot-curated-changelog-88 | Change-log entry 88, the UniProt prompts: cited as short hashes of the prompt files (the same artefacts the prereg calls renders), curated.prompt.txt on the prompt-conditions branch. On integration/freeze the prompts … |
| 249 | `7d1b3299` | binding | prompt-uniprot-census-changelog-88 | Change-log entry 88, the UniProt prompts: cited as short hashes of the prompt files (the same artefacts the prereg calls renders), census.prompt.txt on the prompt-conditions branch. |
| 249 | `950f366c` | binding | prompt-uniprot-generated-1-changelog-88 | Change-log entry 88, the UniProt prompts: cited as short hashes of the prompt files (the same artefacts the prereg calls renders), generated.prompt.txt on the prompt-conditions branch. |
| 273 | `ea36665` | binding | changelog-sources-commit | Change-log entry 92 (the files line): the sources branch's commit. |
| 273 | `7f804abeebf7…` | binding | changelog-census-t5 | Change-log entry 92: census-t5, the frozen bank's census input. |
| 273 | `c707e769` | binding | changelog-census-source-digest | Change-log entry 92: the census source's digest (the canonical OntologySource JSON), as the UniProt prompt manifest records it for the census condition. |
| 273 | `dbe64520` | binding | changelog-generated-dc-1 | Change-log entry 92: the masked generated draft 1 (data/sources/uniprot-sp-2026_03/generated-dc-1.json). |
| 273 | `dbaa4c43` | binding | changelog-generated-dc-2 | Change-log entry 92: the masked generated draft 2 (data/sources/uniprot-sp-2026_03/generated-dc-2.json). |
| 273 | `daef4bcd` | binding | changelog-generated-dc-3 | Change-log entry 92: the masked generated draft 3 (data/sources/uniprot-sp-2026_03/generated-dc-3.json). |
| 273 | `4027df50` | binding | changelog-curated | Change-log entry 92: the curated UniProt source (data/sources/uniprot-sp-2026_03/curated.json.gz, the file as committed). |
| 298 | `06af3f8` | binding | changelog-a15-ontology-commit | Change-log entries 95, 96 and 98: the ontology commit the a15 sources are built from (a build input, bound to that field of the committed a15 build record). |
| 298 | `d33c543b` | binding | changelog-a15-tbox-digest | Change-log entries 95 and 96: the ontology digest (a build input, bound to that field of the committed a15 build record). |
| 298 | `839abbd` | historical | changelog-a15-stale-pin | Historical (change-log entry 95): the stale ontology pin that was replaced; nothing is built from it. |
| 299 | `06af3f8` | binding | changelog-a15-ontology-commit | Change-log entries 95, 96 and 98: the ontology commit the a15 sources are built from (a build input, bound to that field of the committed a15 build record). |
| 300 | `06af3f8` | binding | changelog-a15-ontology-commit | Change-log entries 95, 96 and 98: the ontology commit the a15 sources are built from (a build input, bound to that field of the committed a15 build record). |
| 301 | `06af3f8` | binding | changelog-a15-ontology-commit | Change-log entries 95, 96 and 98: the ontology commit the a15 sources are built from (a build input, bound to that field of the committed a15 build record). |
| 304 | `839abbd` | historical | changelog-a15-stale-pin | Historical (change-log entry 95): the stale ontology pin that was replaced; nothing is built from it. |
| 304 | `06af3f8` | binding | changelog-a15-ontology-commit | Change-log entries 95, 96 and 98: the ontology commit the a15 sources are built from (a build input, bound to that field of the committed a15 build record). |
| 305 | `06af3f8` | binding | changelog-a15-ontology-commit | Change-log entries 95, 96 and 98: the ontology commit the a15 sources are built from (a build input, bound to that field of the committed a15 build record). |
| 305 | `d33c543b` | binding | changelog-a15-tbox-digest | Change-log entries 95 and 96: the ontology digest (a build input, bound to that field of the committed a15 build record). |
| 305 | `23c10717` | unregistered | – | No full value in git: the store snapshot reference is private (the committed README gives a prefix only) |
| 310 | `4fcf5f89` | unregistered | – | No full value in git: the vocabulary table is private |
| 312 | `06af3f8` | binding | changelog-a15-ontology-commit | Change-log entries 95, 96 and 98: the ontology commit the a15 sources are built from (a build input, bound to that field of the committed a15 build record). |
| 316 | `fbaac3af` | binding | changelog-a15-reviewed-mapping | Change-log entry 98: the curator's reviewed mapping (a build input, bound to that field of the committed a15 build record). |
| 316 | `06af3f8` | binding | changelog-a15-ontology-commit | Change-log entries 95, 96 and 98: the ontology commit the a15 sources are built from (a build input, bound to that field of the committed a15 build record). |
| 316 | `4fcf5f89` | unregistered | – | No full value in git: the vocabulary table is private |
| 352 | `42cd54d8` | binding | changelog-step2-addendum | Change-log entry 106: step 2's dated addendum (data/analysis/weighted-holm-step2.json) on integration/freeze. The config's own binding of it is the recomputations work's (feat/freeze-recomputations). |
| 390 | `36556c92` | binding | changelog-corporate-set | Change-log entry 109: the corporate question set. |
| 390 | `7a3454f` | binding | changelog-corporate-commit | Change-log entry 109: the commit the corporate set was frozen at, reachable from the corporate generator branch. |
| 444 | `5d3fc1d` | historical | changelog-stray-copies | Historical (change-log entry 115): a directory name of copies that were removed. |
| 467 | `4fca135f` | historical | changelog-scorer-v4 | Historical (change-log entries 118 and 120): scorer-rules-v4's digest, which v5 replaced in section 3 (entry 121). |
| 490 | `4fca135f` | historical | changelog-scorer-v4 | Historical (change-log entries 118 and 120): scorer-rules-v4's digest, which v5 replaced in section 3 (entry 121). |
| 493 | `3d6b1e6a` | binding | changelog-scorer-v5 | Change-log entry 121: scorer-rules-v5's digest, by the scorer's own scorer_digest() at the merge that brought it (the same artefact as scorer-digest). |
| 493 | `9a64f4e` | binding | changelog-scorer-v5-code | Change-log entry 121: the scorer-rules-v5 code commit, reachable from the merge that brought it. |
