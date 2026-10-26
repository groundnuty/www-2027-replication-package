#!/usr/bin/env python3
"""Change-log entry 189, items 5 and 7 (science, the design log Oct 23:05Z): the ONE script for every
exploratory interval estimate, partitioned or not.

- Every interval is §4.1's unstratified cluster bootstrap: ClusterMap.resampling (entry 52's clusters), seed 20261026,
  R = 10,000, read from the frozen analysis config. The frozen per-arm path's stratified intervals are not reported
  (§4.1: stratified resampling understated the variance, entries 27 and 40). Point estimates are the frozen per-question
  means: the frozen per-arm code (pipeline._arm_result, real_output_ledger, refusal_report) runs unchanged, given the
  clusters' resample matrix in place of the stratified one. A cluster's draw weights each of its questions, so
  ``counts @ x`` there is the cluster bootstrap of the per-question values (stats.ratio_draws).
- Partitioned run directories (entries 179/180: Qwen3.8's Rhea and opaque; entry 184: GLM's and Gemma's opaque) are
  pooled by ONE call over the union of a model x graph's partition OUTs: the frozen loaders per partition, attempts and
  ledger rows merged, one question index over the union and one resample matrix over it. Per-partition intervals are
  never computed or combined. The union is refused when:
  - a question or a run id is in two partitions;
  - a question's sha256("20261022|" + id) mod K is not its OUT's recorded partition (the partition record, bound to
    the OUT by its deployment identity's partition block);
  - an OUT has no partition block (GLM's source natural OUT: its parents count once, through the partitions' copies),
    or is the source another partition names (derived_from.path, manifest entry184.source_out);
  - an OUT is voided (voided.jsonl, matched on the OUT's own path and its identity's on-host path);
  - the partitions' deployment identities name different stacks for a (graph, model). This stack check replaces
    check_one_stack's one-route-URL and one-store-endpoint refusals for the exploratory union only (entries 150, 181,
    184): each OUT's store copy, and a GLM partition's parents' copy, must carry its passed identity check instead.
    A plain-run partition union of a privately pinned graph (--plain-partition-record-sha256, --plain-union-outs) takes
    the identity view's store note, compared exactly (VIEW_STORE_CHECK_NOTE), in place of that check, provided the
    reference pin and both private tables are given and list every partition's copy (design ruling (b), 25 Oct);
    its results say so (``partition_union.copy_identity``) and record the OUT manifest pins. Its (OUT, endpoint) pairs
    that the several-endpoints rule (below) would hold to store-proofs entries are held to the same pinned description
    instead (``partition_union.endpoint_proofs``): listed by their OUT, and the private endpoint copies' endpoint of the
    replica the OUT's identity names.
  - not every partition of the record is present exactly once (a missing one would make its questions missing
    attempts), or the record does not follow the rule over the question set.
  With ``--partition-binding BINDING.json --partition-plan PLAN.json ...`` (DeepSeek's dsh partitions) the
  store copy each OUT must have is its partition's plan's, not the record's; the union is refused unless the binding is
  the cited record's (sha256, K, rule, every partition), there is one plan per partition naming the binding, an OUT name
  and one of the binding's copies, and each OUT's partition block names the binding and its plan (binding_sha256,
  plan_sha256), its OUT's name is the plan's, and its copy and store replica are the plan's. An OUT whose partition block
  names a binding is refused when none is cited, and a cited binding never falls back to the record's copies. A binding
  is cited whole, with one plan for each of its K partitions, whether the call is the pooled union or a --partitions
  subset (below): the binding and every plan are read and checked as one object, and only the OUTs of the analysed
  partitions are then matched to their plans.
  With ``--partitions I,J,...`` (a strict subset of the record's partitions, at least two) the same union is refused
  unless exactly the named partitions are present, each once, and an OUT of any other partition is refused. With a cited
  binding only a PROPER subset is accepted (science, 25 Oct ~11:5xZ: the pooled call stays the reported one): naming every
  partition is refused there, because the pooled call cites the binding without the option; each analysed partition's OUT
  is checked against its own plan exactly as in the pooled call, and the partitions not named need no OUT. The record
  is still required and checked in full (rule, question set, every partition), and so is every named OUT. The design,
  the question index and the resample matrix then cover only the named partitions' questions: the design is restricted
  to them before the missing attempts are made (and refused if a base question has questions both inside and outside
  the subset). A cluster of the cluster map with questions in a partition outside the
  subset is resampled as the cluster of its subset questions alone (the map is unchanged; the results count these
  clusters, ``partition_union.subset.clusters_cut``). The subset is recorded in ``partition_union.subset``; naming every
  partition is the call without the option, and its results are the same bytes.
  A call without a partition record is the frozen per-arm path's (its check_one_stack unchanged), and refuses any
  partition OUT and any voided OUT. A partition that moved within its OUT (store_move_remote.py) is its record's copy
  when its identity's store changes join up from that copy.
- Store copies (science, ~09:50Z; entry 150's in-OUT store changes): a (graph, model) whose records, attempts and
  set-aside tries, name more than one store endpoint (store.endpoint, read by STORE_FIELDS alone) is refused unless each
  (OUT, endpoint) is listed in the OUT (its identity's store blocks, or a manifest store_changes from/to) and proven by
  an entry of the reviewed store-proofs.json: its recorded storecheck and known-answer files (--store-proof, checked by
  content, never a query) show the copy identical to the graph's reference stats, with the reference's index, its known
  answers passed. The records' own recorded endpoint is the ground truth (science, 25 Oct 00:10Z): ordered by their own
  started_at, the records of an OUT with manifest store changes must run through its chain of endpoints in order, each
  copy's last attempt ending before the next copy's first starts, and a store change's change point is the start of the
  first record on its new endpoint (``store_timeline``). An ordinary entry's change_at that is another instant is
  disclosed beside it with the gap, and its proof checks are held to the first record (FIRST_RECORD_RULE; science,
  25 Oct, Gemma e190 K=3); a second_period entry's change_at, or that of an entry whose OUT name is also another OUT's
  of the call, must still be that start to the millisecond. Records of
  two copies interleaving are a mix of copies, not a late timestamp, and are refused. The manifest's own placement of
  records (pre_change_run_ids) is only compared, as booleans in the results. Branch-phase rows (science, 25 Oct
  ~04:55Z: started at or after the instant of the manifest's additions entry that added "phase branch") carry the
  recorded endpoint of their parent attempt, not the serving copy's (harness/checkloop/online.py:542 copies a branch record's store from its parent), so
  they are outside the order check and the periods (BranchPhase); each one's live retry is attributed to the copy
  current at its start, the chain's last copy (branch_copy_endpoint); a malformed
  additions list refuses. A branch-only OUT (science, 25 Oct ~09:5xZ: the manifest names the branch phase and has store
  changes, it declares the Tier 1 parents it copied (``copied_parents_declared``) and its directory holds
  order-branch.json and no other order file; ``is_branch_only``), whose records name
  its parents' copies alone, takes the periods of its copies from the instants of its manifest's store changes: each
  branch row's retry is attributed to the copy current at its start, every copy of its chain is listed and proven
  though no record names it, and the rows made before the branch phase must record the copy current at their start
  (``branch_only_timeline``, BRANCH_ONLY_RULE). A copy behind an endpoint the OUT lists (a route change behind an unchanged
  forward) needs its own entry and makes that endpoint's row unsplittable.
  Each graph's reference is pinned before any call (REFERENCE_STATS_SHA256: Rhea, opaque, UniProt and so SIB, and
  corporate); a15's stays in the private tree (PRIVATE_REFERENCE_PINS: its pin file's sha256, --reference-pin).
  Two reviewed allowances (science, ~11:30Z): a copy behind a spoke forward may be proven by check-copies known answers
  at its direct endpoint where the entry names the forward -> direct mapping and its watchdog log shows it (FORWARD_*);
  and a copy checked before entry180-store-check/2 by the store agent's ident.py file, where the entry lists what that
  format does not compare (IDENT_PY_NOT_CHECKED), carried into the results.
  Each copy's share of attempts is reported per arm, over the call and per OUT, and per store change before and after
  it. One endpoint per (graph, model) runs as before.
- Cluster maps: UniProt's is the frozen config's. Rhea 142's and opaque Rhea 142's are built here (``clusters``) by the
  frozen rule (prereg.connected_clusters and write_cluster_map: shared declared anchors within a template, every
  question of the set, dev included; restricted to the analysed split by ClusterMap.resampling). Opaque's anchors are the
  opaque question records' own, never the held-out map. Both are pinned below by sha256, recorded before any call.
  Another graph has no map and is refused.

The frozen modules are imported from ``--code-root`` (a deploy of the frozen analysis: harness/, scripts/, data/),
checked to come from there, and the code digest recorded.

    $PY -I exploratory.py clusters --code-root DEPLOY --out-dir clusters
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 $PY -I exploratory.py analyse --code-root DEPLOY --graph G \\
        --split test --batch OUT [--batch OUT ...] --questions Q.jsonl [...] --voided voided.jsonl [...] \\
        [--partition-record RECORD.json [--partitions I,J,...]] [--ledger LEDGER.jsonl ...] [--n-reps N] \\
        [--tier-if-missing T] [--store-proof CHECK.json|FORWARD.log ...] [--reference-pin PIN] \\
        [--private-copy-classes TABLE --private-copy-classes-sha256 SHA256] \\
        [--private-endpoint-copies TABLE --private-endpoint-copies-sha256 SHA256] --out DIR
    $PY -I exploratory.py parity  (the same inputs; dev only): the frozen per-arm path's results against this
        script's with the frozen stratified matrix, section by section, no number printed
    $PY -I exploratory.py compare A/results.json B/results.json   (section by section, no number printed)
    $PY -I exploratory.py store-timeline --code-root DEPLOY OUT [OUT ...]   (one JSON object per OUT, no counts: exit 0
        clean or no store changes, 2 STOP, 3 unreadable)
    $PY -I exploratory.py check-stores --code-root DEPLOY --graph G --batch OUT [--batch OUT ...] \\
        [--store-proof CHECK.json|FORWARD.log ...]   (the call's store check alone: one JSON object,
        no counts: exit 0 ok, 2 refused with its message, 3 anything else with the exception class)
"""

from __future__ import annotations

import argparse
import bisect
import dataclasses
import gzip
import hashlib
import importlib
import importlib.util
import itertools
import json
import math
import os
import re
import secrets
import socket
import stat
import sys
import urllib.parse
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import datetime
from fractions import Fraction
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
RULING = ("science, the design log, change-log entry 189 (22 Oct 23:05Z): item 7, every exploratory interval is "
          "§4.1's unstratified cluster bootstrap (seed 20261026, R = 10,000) in one script, point estimates the frozen "
          "per-question means; item 5, partitioned estimates from one clustered bootstrap over the union of the "
          "partitions (entries 179, 180, 184), each question and each copied parent counted once")
SEED, N_BOOT = 20261026, 10_000  # entry 189 item 7: the registered §4.1/§4.2 values, checked against the frozen config
LEVEL = 0.95
SALT = "20261022|"  # entry 179's partition rule
RECORD_SCHEMAS = ("entry180-partition/1", "entry184-partition/1", "entry190-partition/1")
PLAIN_PARTITION_SCHEMA = "private-plain-partition/1"
PLAIN_RECORD_FIELDS = {"rule", "salt", "k", "copies", "reps", "questions_sha256", "test_questions", "partitions"}
PLAIN_PART_FIELDS = {"copy", "questions", "attempts", "ids_sha256", "file", "file_sha256"}
PLAIN_RULE = "int(sha256('20261022|' + question id).hexdigest(), 16) mod K; partition p runs on copy p of the record's list"
# A WEIGHTED plain partition record (the producer's partition_of with weights): the plain record's fields plus the weights and
# their reduction, and a rule string of its own. A record with the mod-K rule keeps exactly PLAIN_RECORD_FIELDS.
WEIGHTED_PLAIN_RULE = ("slot = int(sha256('20261022|' + question id).hexdigest(), 16) mod sum(w), w = the record's weights divided "
                       "by their gcd; partition p is the weight interval containing slot; partition p runs on copy p of the "
                       "record's list")
WEIGHTED_PLAIN_RECORD_FIELDS = PLAIN_RECORD_FIELDS | {"weights", "weights_reduced"}
# A record schema accepted only as one exact file (read_record compares the file's sha256 with the pin): the Coder dh2 record
# (K = 2, both partitions on one copy) has the fields of entry180-partition/1 and a schema name of its own (README, "Partition
# records and bindings")
PINNED_RECORDS = {"coder-dh2-partition/1": "639bb65af66828821bb774bf9e65b6194510fe343587c145a1615fb66b856714"}
# The DeepSeek dsh partitions' binding and plans (--partition-binding, --partition-plan): their store copies come from the
# plans, not from the record's list
BINDING_SCHEMA, PLAN_SCHEMA = "deepseek-opaque-binding/1", "deepseek-opaque-plan/2"
CONFIG = "data/analysis/prereg-config.json"
# The frozen analysis this script runs on: the config-fill head's code (harness/ and scripts/, digest.code_digest) and its
# final analysis config (the pinned e46fd504...); any other deploy is refused, not merely recorded
FROZEN_COMMIT = "847cc27563a3c7a505c8ec0fd647c9314acaae8c"
FROZEN_CODE_DIGEST = "a5e7607b5d2027078437879b32de270c715ccba78b891f5c6ed149c683551387"
FROZEN_CONFIG_SHA256 = "ddca19da589e268ed5b88c8275c4d51106ab2cd40bf25920840848e63f5d97dd"
# The voided OUTs recorded when this script was written (entry 173: SiteB Rhea 213, opaque 33; entry 180: gw-node-1 Rhea 99,
# opaque 28, a15 16): every voided-run record given must hold at least these, so a wrong or partial file is refused
KNOWN_VOIDED = (
    "/scratch/user/project/runs/test-20261021/rhea-qwen3.8-27b",
    "/scratch/user/project/runs/test-20261021/opaque-qwen3.8-27b",
    "/home/user/project/runs/test-20261021/rhea-qwen3.8-27b",
    "/home/user/project/runs/test-20261021/opaque-qwen3.8-27b",
    "/home/user/project/private/a15/runs/test-20261021/a15-qwen38",
)
# A store copy's passed identity check as partition_run.py and glm_partition.py record it in the OUT's identity
IDENTITY_CHECK = re.compile(r"\?cmd=stats equal to \S.* in [1-9][0-9]* fields at \S+ \(store check sha256 [0-9a-f]{64}\)")
# Design ruling (b), 25 Oct: a plain partition union's per-copy rule is met by each partition's manifest identity plus its
# copy's pinned store description (require_plain_pinned_copies), not by a store-stats check. Its OUTs are read through an
# identity view whose store block carries exactly this note (a15_identity_view.py STORE_CHECK_NOTE); the plain union accepts
# that string, compared for equality, in place of IDENTITY_CHECK, and says so in its results (PLAIN_UNION_COPY_IDENTITY).
# Every other union keeps IDENTITY_CHECK.
VIEW_STORE_CHECK_NOTE = "not recorded for this OUT (identity view): no store check is claimed"
PLAIN_UNION_COPY_IDENTITY = "a15 copies: identity by manifest and pinned description; no store-stats check (not QLever)"
# The same ruling for every (OUT, endpoint) of a plain union (science, 25 Oct): no store-proofs.json entry is required for one.
PLAIN_UNION_ENDPOINT_PROOFS = ("a15 copies: no store-proofs.json entry is required for any (OUT, endpoint); each endpoint is "
                               "named by its OUT's identity and is the pinned endpoint copies' endpoint of that OUT's copy")
REGISTERED_SPLIT = "test"  # prereg §1: every OUT of it carries its deployment identity (freeze-day e)
PER_ARM_MODULES = ("pipeline", "inputs", "gate", "prereg", "stats", "metrics", "digest")

# The question sets the maps are built from (pinned), and the maps (entry 52's rule), hashed before any call
QUESTION_SETS = {
    "rhea-142": ("data/questions-frozen/rhea-142/questions.jsonl",
                 "b204846e7c20e82971ec65590298d25d54dd480efcea4f178e5044847dfa96b7"),
    "opaque-rhea-142": ("data/questions/opaque-rhea-142/questions.jsonl",
                        "4153ad29e7d4d659a12425a685e8b460e95e0966930221adea8119190b263507"),
}
# The Rhea generator's own record of each variant's declared anchor parameters (test_anchor_exclusion) and of the
# entities shared between questions (entity_reuse): cross-checks of the anchors read from the records, never inputs
RHEA_MANIFEST = ("data/questions-frozen/rhea-142/manifest.json",
                 "111e3c6d6fe290d243b9a86fe395d94e959b9ff8aaa222860f665346288a7827")
CLUSTER_MAPS = {
    "rhea-142": ("clusters/rhea-142.json", "1f7346fadeffb1d81074255a48c7f9f0eed9d48e8ac65dc8fcd73322aafd881d"),
    "opaque-rhea-142": ("clusters/opaque-rhea-142.json",
                        "86fbd61c0765a29abc614c88fa71dccd4f0fafc6966245384019e7a6e2484441"),
    # corporate (science, design log, 23 Oct 00:57Z): from the frozen generator's anchors (cluster-map), 23 Oct
    "text2sparql-ck26": ("clusters/text2sparql-ck26.json",
                         "b3d09ff4e83f4a6009eea80ac0b7caa9113e269151d2fd7069c0768eff633780"),
}
CORPORATE = "text2sparql-ck26"
A15 = "a15"
SIB = "sib-uniprot-sp-2026_03"
# Every graph a call may name besides the registered one (whose map and question set are the frozen config's): the store
# its records name, the question set (sha256) and how its clusters are found (science, design log, 23 Oct 00:57Z):
# "committed", a map built here by the frozen rule (CLUSTER_MAPS); "private", a15's, built by the same rule and kept
# only in the private tree (mode 600), given at run time (--cluster-map) and read by its pinned sha256; "singletons", SIB:
# no generation anchors, and its anchors would come from held-out golds, so each question is its own cluster and nothing
# is read for it
GRAPHS: dict[str, dict[str, str]] = {
    "rhea-142": {"store": "rhea-142", "questions_sha256": QUESTION_SETS["rhea-142"][1], "map": "committed"},
    "opaque-rhea-142": {"store": "opaque-rhea-142", "questions_sha256": QUESTION_SETS["opaque-rhea-142"][1],
                        "map": "committed"},
    CORPORATE: {"store": CORPORATE, "questions_sha256": "36556c92858afa98a17f46622350077cf4913cdd8d14be9393191324f7228d06",
                "map": "committed"},
    A15: {"store": A15, "questions_sha256": "e22bc7c63f1b306afe906dc83cf58e5c9f72889a0e225df25ec3f2b429284fab",
          "map": "private", "map_sha256": "2277abc0a303b441735de6fa68cbbc0552271fe8fc5b9706e85be546c666f9f8"},
    SIB: {"store": "uniprot-sp-2026_03", "questions_sha256": "d4b895c4d2a2c1da56526dcd639845be274ec47b8f2582dc63adf3118b6f5a31",
          "map": "singletons"},
}
# The graphs whose maps are built from the frozen generator's templates (profiles.TEMPLATE_SETS, generate.anchor_entities:
# their records carry no anchors list), cross-checked against the generator's manifest. Rhea's records carry their
# anchors (and harness/qgen/templates.py holds only its first round's variants), so its map is built from them
TEMPLATE_MAP_GRAPHS = (CORPORATE, A15)
ANCHOR_REUSE_GRAPHS = (CORPORATE,)  # their generator profile writes anchor_reuse (unique anchors per template)
# Deployment shares and the conditional sensitivity of every exploratory contrast (science, design log, ~03:35Z, recorded in
# change-log entry 189 item 4): each attempt's serving deployment from infrastructure fields only, by entry 189's
# attribution (attribution.py, byte for byte the attribution check's: feat/e189-balance cc643509), never an outcome
ATTRIBUTION_SHA256 = "9299922566fd8f0a3878663f813386f8eaee4f6a785db5f4bba5732ba527f875"
TRIGGER = Fraction(1, 10)  # two arms' shares of one deployment differing by strictly more than 10 points
RETRY_WITHOUT_RECORD = "retry without record"  # a checker arm's retried output whose retry has no record
JOIN_MUST_BE_ZERO = ("unjoined", "ambiguous", "without_id")  # a fronted own turn must join one backend (balance.py's)
GENERIC_PREFIX = "generic_retry/"  # harness/checkloop/ledger.py: a generic cell's source names
# Served-fingerprint changes (science, design log, ~05:55Z; the rule as the lead relayed it, accepted by science with
# two conditions): within an OUT, every own turn's served fingerprint of a model must be the OUT's baseline (the
# stack identity's fingerprint, else served-models.json's first), or one admitted (b) by the OUT's own
# admitted-fingerprints.json (admit.py's entry-167/170 gate) or (c) by this script's reviewed config, ADMISSIONS; either
# way with the baseline's vLLM version (parsed from the fingerprint) and, where the stack identity records a checkpoint
# revision, that revision. Anything else refuses. This lives in the exploratory script only: the frozen confirmatory
# analysis, its code path and its config pin e46fd504 are untouched (UNTOUCHED_LIST_SHA256)
ADMISSIONS = "admitted-fingerprints.json"  # beside this script; an empty list until a relaunch is admitted
ADMISSIONS_SCHEMA = "e189-admitted-fingerprints/1"
ADMISSION_FIELDS = ("route_model", "stack", "system_fingerprint", "vllm_version", "checkpoint",
                    "entry170_verdict_sha256", "admitted_at")
ADMISSION_CITATIONS = ("ruling", "change_log")  # each entry names the ruling or change-log entry admitting it
CHECKPOINT_KEYS = ("snapshot", "revision")  # a stack identity's checkpoint revision (Gemma's snapshot, Qwen's revision)
# Store copies (science, ~09:50Z, the analysis side of change-log entry 150's in-OUT store changes; design changelog,
# 23 Oct 09:30Z prevention): runs move between identical store copies within their OUT, and each partition of a union
# runs on its own copy. A (graph, model) whose records (attempts and set-aside tries) name more than one store endpoint
# is analysed only when every (OUT, endpoint) is listed in the OUT (its deployment identity's store blocks, or a
# store_changes entry's from/to in its manifest) and proven by a STORE_PROOFS entry whose recorded check files (given
# by --store-proof, checked by content; never a query) show the copy identical to the graph's reference stats, with the
# reference's index and its known answers passed. One endpoint per (graph, model) runs as before. The runner-written
# formats are frozen, so the binding lives in this reviewed config (the lead, ~09:55Z). One kind of entry needs
# no listing in the OUT: a copy behind an endpoint the OUT already lists (``behind_endpoint``: a route change behind an
# unchanged forward), with its switch instant; the reviewed readers expose no per-attempt store-query time, so that
# endpoint's attempts are reported as one row of its copies, unsplittable. Exploratory script only (UNTOUCHED_LIST_SHA256)
STORE_PROOFS = "store-proofs.json"  # beside this script; an empty list until the copies' check files are listed
COPY_CLASSES = "copy-classes.json"  # beside this script: each attributable copy's speed class (copy_classes.py)
# beside this script: the copy behind an endpoint that OUTs with a single store endpoint used and whose deployment
# identities name no replica (copy_classes.parse_endpoint_copies). It names a copy for the label and the class only: it is
# never a proof (the proofs, the order check and the store changes read STORE_PROOFS alone), and a store-proofs entry or an
# identity that names another replica for its endpoint refuses (endpoint_copy_conflicts)
ENDPOINT_COPIES = "endpoint-copies.json"
STORE_PROOFS_SCHEMA = "e189-store-proofs/1"
STORE_PROOF_FIELDS = ("graph", "out", "endpoint_sha256", "replica", "storecheck_sha256", "known_answers_sha256")
STORE_FIELDS = ("run_id", "model", "store.name", "store.endpoint", "started_at", "ended_at")  # all the store reader
# reads of a record; the last two are the runner's wall-clock instants of the attempt (infrastructure timestamps)
STORECHECK_SCHEMA = "entry180-store-check/2"  # partition_run.py storecheck's file
# Each graph's recorded reference stats and the index it names (name-index "<graph>@<sha256 of the index list>"),
# pinned before any exploratory call (science, ~10:35Z): the sha256 of the store agent's recorded reference file, the
# registered store's ?cmd=stats that check-copies.sh compares every copy with in all fields (the store agent's
# ref-<graph>-stats.json; partition_run.REFERENCE_STATS_SHA256, entry 180, for Rhea and opaque). UniProt's is
# its reference-stats file, whose stats are copies.json's "reference" that its copies were checked identical
# to; SIB's records are on the UniProt store, so it is SIB's too. Corporate's is ref-text2sparql-ck26-stats.json. A
# graph not listed here or in PRIVATE_REFERENCE_PINS has no copies to prove
REFERENCE_STATS_SHA256 = {"rhea-142": "63f7562f8b90f430741002431926a843c863447f2eaf34a82ff0d6d107b89638",
                          "opaque-rhea-142": "07ed3cc31849a9e3e0adb6103bde39e5cd431a65b96a82a5d5ae4b027216777c",
                          "uniprot-sp-2026_03": "0513d3f79dfdd75bade876d713e29eb23cfa46962d499e82a36cf3ccf811fee8",
                          "text2sparql-ck26": "61d538bb8e110a01213d0b9c084667a631981f2d2fcc299bc1e0413c7833d5c9"}
REFERENCE_NAME_INDEX = {
    "rhea-142": "rhea-142@c0174e728db659c40c96b988a7c96c1a7f4fa3b396534ca97e16594271e38a33",
    "opaque-rhea-142": "opaque-rhea-142@c454dc75b2d5a31f39975c18fef02190af6c84b8ed711af57040d7063fd046bf",
    "uniprot-sp-2026_03": "uniprot-sp-2026_03@9484989e47e8903a78937402a0b3c69b43a12499721ea71d3c940ec9e62a62b1",
    "text2sparql-ck26": "text2sparql-ck26@c51a9638d28fd0c27b262336aa04df3d9b7faa371d216678c1e8d77e4cbc9691"}
# A graph whose recorded reference stays in the private tree (science, ~10:35Z: a15): its pin is a mode-600 file there
# (the cluster-map pattern) naming the reference's sha256, given at run time by --reference-pin and read by this sha256
# alone; only the pin's hash is public
PRIVATE_REFERENCE_PINS = {A15: "72b99843cb35aa120b2940c0549b1d923f638923efdd3f64c68a85e4e1356cc0"}
REFERENCE_PIN_SCHEMA = "e189-reference-pin/1"
# What the results show in place of the basis and the source of a copy only a privately pinned graph's own class table
# describes (the table keeps both; they stay in the private tree)
# A copy name a private table may give: the call's results print it as the copy's label (stores.copies, the share rows),
# so it is a plain name, never a path or a sentence
PLAIN_COPY_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")
# The private trees of the privately pinned graph (science, 25 Oct): a private table and the output of a call that uses
# one live only under these, on the host that has them, never on the hub (/home/user/project-*,
# /Users/...). Each root counts only on a host whose name its predicate accepts (the spoke's path string is the hub's
# home too, so the path alone cannot tell them apart: the spoke's first dot-separated label is exactly "agents-host",
# the hub's is "agents-host-2") and only where it exists as a directory only its owner can enter and is not itself a
# symlink. SiteB: the first label is login<N> (a login node) or t<4 digits> (a compute node), with or without a domain,
# and the SiteB scratch path itself exists on no other host: the host names are an assumption (read on a login node, not
# on a compute node or inside the container; the first label is what a container shares with its host).
# SiteA's private tree: not listed (its path is not known here); add it when science names it.
def _first_label(host: str) -> str:
    """The host name up to its first dot (a host reports its short name or its fully qualified one by configuration)."""
    return host.split(".", 1)[0]


PRIVATE_A15_ROOTS: tuple[tuple[Path, Callable[[str], bool]], ...] = (
    (Path("/home/user/project/private/a15"), lambda host: _first_label(host) == "agents-host"),  # the spoke
    (Path("/scratch/user/project/private/a15"),  # SiteB
     lambda host: re.fullmatch(r"login\d+|t\d{4}", _first_label(host)) is not None),
)
PRIVATE_BASIS = "from the private copy classes: the basis stays in the private table"
PRIVATE_SOURCE = "private copy classes"
KNOWN_ANSWERS = 10  # the store agent's load10.py: ten dev gold queries at once, rows compared as sets
# A copy whose OUT endpoint is a spoke forward to it, with its known answers run at its DIRECT endpoint (science,
# ~11:30Z: GLM e184 p0/p1/p2's own copies, checked by the store agent's check-copies run there, the storecheck through
# the forward): accepted only where the entry names the forward -> direct mapping (``forward``: the sha256 of the
# direct endpoint, as "http://HOST:PORT/", and of the forward's record) and the record, the forward's watchdog log
# (go_e184.sh's forward step starts spoke_forward_watchdog.sh, which logs every ssh it starts and its exit), shows every
# ssh of it listening on the OUT's endpoint and forwarding to that direct endpoint, one of them running at change_at
# (the first record on the copy, for the ordinary entry of a period a change began: FIRST_RECORD_RULE),
# and the watchdog not ended by then. Only a check-copies entry counts there: it names the replica, which ties the
# direct endpoint to the copy
FORWARD_FIELDS = ("direct_endpoint_sha256", "record_sha256")
FORWARD_START = re.compile(r"(?P<at>\S+) (?P<name>\S+) ssh pid (?P<pid>[0-9]+) (?P<listen>127\.0\.0\.1:[0-9]+) -> "
                           r"(?P<target>[0-9.]+:[0-9]+)")
FORWARD_EXIT = re.compile(r"(?P<at>\S+) (?P<name>\S+) ssh pid (?P<pid>[0-9]+) exited rc=-?[0-9]+ after [0-9]+ s")
FORWARD_END = re.compile(r"(?P<at>\S+) (?P<name>\S+) (?:STOP file: watchdog ends|watchdog stopped)")
# A copy its OUTs name by its node's host name while its recorded identity checks and known answers name it by address
# (by ruling, the same host by the copy table): accepted only for
# the pair pinned here, by sha256 (no address committed), and only where the entry's ``alias`` names that pair's alias
# endpoint and the copy table pinned with it (the store-copy tooling's copy-assignments/1 snapshot), given
# by --store-proof: its "aliases" map both the OUT endpoint's and the alias's host:port to the entry's replica, whose
# "copies" entry lists the alias among its endpoints. The alias counts for ident.py's copy URL and load10.py's endpoint
# alone. Never a general host-name rule. Such a copy's proof has three points (the same ruling): its entry's pair, and
# ``further`` points (FURTHER_FIELDS), each checked as its pair is but with no time bound: the graph's pinned reference
# file read from that copy (its "reference" the OUT's endpoint or the alias) with its known answers, and another
# identity check with its known answers
ALIAS_FIELDS = ("endpoint_sha256", "record_sha256")
FURTHER_FIELDS = ("point", "storecheck_sha256", "known_answers_sha256")
COPY_TABLE_SCHEMA = "copy-assignments/1"
STORE_ALIASES = {  # (graph, sha256 of the OUT's endpoint) -> (sha256 of the alias endpoint, sha256 of the copy table)
    ("rhea-142", "a1477ccde3766bea29cacd8ca3245d570203678b2093eb8c341fa15f353f2ab4"): (
        "54ef810f2bd38c722e09a50776e23d99cb78777421f0c25b67d456b32c5cb03d",
        "2a36d6b2aa1775a5a1c057b66f2066df262b4b92dfec5276814e58dbe1789274"),
    # a second pair: an OUT of the corporate graph names its node by host name too; its copy table maps both names to
    # the entry's replica. The key is that
    # OUT endpoint's sha256 and the value the alias endpoint's, then the copy table's: no address committed
    ("text2sparql-ck26", "75b1b34649e44b4da267e822be64ab5a3d8e6be3b921de6a63c1fb96bb3d66aa"): (
        "7ba4d5531600b913827cd5e5e7df0fc7c0daddf601c85ae9d9bdd16b08df99b3",
        "6f195ca5f876f52b936a7f270ef6d10e02c75dfc8345fafca206885a225b3142")}
# The store agent's ident.py identity file (science, ~11:30Z: entry 150's UniProt copies, checked before
# entry180-store-check/2 existed: an ident.json, and the copies block of the UniProt reference file itself), read in
# place of a storecheck only for an entry naming identity_format "ident.py" and listing as not_checked exactly what
# that format does not compare. It records the copy's ?cmd=stats in the reference's 13 fields (IDENT_PY_FIELDS: the
# stats, the name-index, the triple counts), compared here again, by type and value, with the pinned reference file's;
# the known answers are load10.py's output, as for any copy
IDENT_PY = "ident.py"
IDENT_PY_FIELDS = ("git-hash-index", "git-hash-server", "version-server", "name-index", "num-permutations",
                   "num-predicates-normal", "num-predicates-internal", "num-subjects-normal", "num-subjects-internal",
                   "num-objects-normal", "num-objects-internal", "num-triples-normal", "num-triples-internal")
IDENT_PY_NOT_CHECKED = {
    "replica": "the copy's name: ident.py names a copy by its URL alone",
    "index-files": "the index files' list hash (check-copies' index_list_sha256 and index_identity_ok): ident.py has "
                   "the index's name only",
    "server-image": "the server's container image (check-copies' image: the SIF its job holds open, by sha256, part "
                    "of its ok): ident.py has the server's version fields only",
    "server-command-line": "the server's settings (check-copies' settings and settings_ok)",
    "stats-beyond-reference-fields": "?cmd=stats fields beyond the reference's 13 (entry180-store-check/2 compares "
                                     "every field, a text index's among them; ident.py records the 13 only)"}
NO_ENDPOINT = "no endpoint"
UNSPLITTABLE = ("unsplittable: the reviewed record readers expose no per-attempt store-query time, so the attempts on an "
                "endpoint with a copy behind it are one row of its copies (the lead, ~09:55Z)")
# The analysis-untouched list (www2027-reviews/e190/analysis_untouched.sh's): every harness/analysis file, every scorer
# module (harness/score/digest.py SCORER_MODULES) and digest.py, the frozen config and its pin, as at 847cc275 (config
# e46fd504): the sha256 of its "sha256  path" lines
UNTOUCHED_LIST_SHA256 = "e74b844c42b639cfba874078ee3ced6fc721b7d5c4ae1da121ef596844deba81"  # 33 files
RESAMPLING = ("§4.1's unstratified cluster bootstrap (change-log entry 189 item 7): B = 10,000 draws with replacement of "
              "the clusters of the question index (entry 52's cluster map, ClusterMap.resampling, seed 20261026), each "
              "question weighted by its cluster's draw count; one matrix for every arm; over a partition union, one "
              "index and one matrix over the union (entry 189 item 5)")
STATUS = ("exploratory (prereg §6; change-log entry 189 item 7): point estimates are the frozen per-question means, every "
          "bootstrap interval the unstratified cluster bootstrap (the frozen per-arm path's stratified intervals are not "
          "reported); the Clopper-Pearson bounds on question-level events stay as §3 registered them (science, design log, "
          "23 Oct 00:57Z)")
# analyse()'s own descriptions, verbatim: the weighting, intervals and exact bounds are the frozen per-arm code's
WEIGHTING = "proportions over N_att averaged per question then over questions; ratios as pooled counts (prereg §3)"
INTERVAL = "percentile over defined resamples, inverted-CDF quantiles (prereg §4)"
EXACT_BOUNDS = "Clopper-Pearson for binary question-level events only (prereg §3)"


class Refused(Exception):
    """The call may not run; the message says why."""


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_file(path: Path) -> str:
    return sha256_bytes(Path(path).read_bytes())


def reduce_weights(weights: Any) -> list[int]:
    """The weights divided by their gcd (the producer's reduce_weights). Refused: fewer than two, a bool, a non-integer, zero
    or negative."""
    if not isinstance(weights, (list, tuple)) or len(weights) < 2:
        raise Refused("weights: at least two integers")
    for w in weights:
        if isinstance(w, bool) or not isinstance(w, int) or w < 1:
            raise Refused("weights: each a positive integer")
    g = math.gcd(*weights)
    return [w // g for w in weights]


def partition_of(question_id: str, k: int, weights: Sequence[int] | None = None) -> int:
    """Entry 179's rule (partition_run.partition_of). Without ``weights`` (every mod-K record, every other union kind) the
    partition is the hash mod K, exactly as before. With the weights of a weighted plain record (the producer's
    partition_of): slot = the hash mod sum(w) over the weights divided by their gcd, and the partition is the weight interval
    containing the slot (intervals in partition order, each partition's interval [sum of the earlier w, + its w))."""
    h = int(sha256_text(SALT + question_id), 16)
    if weights is None:
        return h % k
    w = reduce_weights(weights)
    if len(w) != k:
        raise Refused("weights: one per partition")
    slot = h % sum(w)
    total = 0
    for i, x in enumerate(w):
        total += x
        if slot < total:
            return i
    raise AssertionError("unreachable")


def ids_digest(ids: Iterable[str]) -> str:
    """The sha256 of ids, sorted, one per line (partition_run.ids_digest)."""
    return sha256_text("".join(f"{q}\n" for q in sorted(ids)))


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


# --- the frozen code ------------------------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Frozen:
    root: Path
    pipeline: ModuleType
    inputs: ModuleType
    gate: ModuleType
    prereg: ModuleType
    stats: ModuleType
    metrics: ModuleType
    checked_call: Callable
    identity: dict

    @property
    def refusals(self) -> tuple[type[BaseException], ...]:
        return (Refused, self.gate.BatchRefused, self.prereg.PreregError, self.inputs.InputError)


def load_frozen(code_root: Path) -> Frozen:
    """The frozen analysis modules, imported from ``code_root`` and checked to come from there; contract v2 rule 22's
    checked_call from its scripts/pilot_summary.py, as scripts/analyse.py takes it."""
    root = Path(code_root).resolve()
    if not (root / "harness" / "analysis" / "pipeline.py").is_file():
        raise Refused(f"--code-root {root} has no harness/analysis/pipeline.py")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    mods = {name: importlib.import_module(f"harness.analysis.{name}") for name in PER_ARM_MODULES}
    for module in mods.values():
        if not Path(module.__file__).resolve().is_relative_to(root):
            raise Refused(f"{module.__name__} was imported from {module.__file__}, not from --code-root {root}")
    summary = root / "scripts" / "pilot_summary.py"
    spec = importlib.util.spec_from_file_location("pilot_summary", summary)
    module = importlib.util.module_from_spec(spec)
    sys.modules["pilot_summary"] = module
    spec.loader.exec_module(module)
    commit = root / "COMMIT"
    identity = {"code_root": str(root), "commit": commit.read_text(encoding="utf-8").strip() if commit.is_file() else None,
                "code_digest": mods["digest"].code_digest(root), "pilot_summary_sha256": sha256_file(summary)}
    if identity["code_digest"] != FROZEN_CODE_DIGEST or identity["commit"] not in (None, FROZEN_COMMIT):
        raise Refused(f"--code-root {root}: code digest {identity['code_digest'][:12]} (COMMIT {identity['commit']}), not "
                      f"the frozen analysis's {FROZEN_CODE_DIGEST[:12]} ({FROZEN_COMMIT[:12]})")
    identity["untouched_list_sha256"], _ = untouched_list(root)
    if identity["untouched_list_sha256"] != UNTOUCHED_LIST_SHA256:  # the confirmatory files, as at 847cc275 (e46fd504)
        raise Refused(f"--code-root {root}: the analysis-untouched list is {identity['untouched_list_sha256'][:12]}, "
                      f"not the frozen {UNTOUCHED_LIST_SHA256[:12]}")
    return Frozen(root, mods["pipeline"], mods["inputs"], mods["gate"], mods["prereg"], mods["stats"], mods["metrics"],
                  module.checked_call, identity)


def frozen_config(F: Frozen) -> tuple[Any, dict]:
    """The frozen analysis config (by its pinned sha256: pipeline.frozen_config as a confirmatory run takes it), holding
    the registered seed and R."""
    cfg = F.prereg.PreregConfig.load(F.root / CONFIG)
    record = F.pipeline.frozen_config(cfg, testing=False)
    if record["config_sha256"] != FROZEN_CONFIG_SHA256:
        raise Refused(f"the analysis config is {record['config_sha256']}, not the final frozen {FROZEN_CONFIG_SHA256}")
    if (cfg.seed, cfg.n_boot) != (SEED, N_BOOT):
        raise Refused(f"the frozen config's seed and R are {cfg.seed}, {cfg.n_boot}, not entry 189's {SEED}, {N_BOOT}")
    return cfg, record


def untouched_list(root: Path) -> tuple[str, int]:
    """The analysis-untouched list (www2027-reviews/e190/analysis_untouched.sh's files) under ``root``: every
    harness/analysis file, every scorer module (harness/score/digest.py SCORER_MODULES) and digest.py, the frozen config
    and its pin, as "sha256  path" lines sorted by path; returns the sha256 of those lines and their number."""
    import ast

    root = Path(root)
    tree = ast.parse((root / "harness" / "score" / "digest.py").read_text(encoding="utf-8"))
    scorer = [e.value for n in ast.walk(tree) if isinstance(n, ast.Assign)
              and any(getattr(x, "id", "") == "SCORER_MODULES" for x in n.targets) for e in n.value.elts]
    analysis = [p.relative_to(root).as_posix() for p in (root / "harness" / "analysis").rglob("*")
                if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"]
    files = sorted({*analysis, *scorer, "harness/score/digest.py", "data/freeze/frozen-config.json", CONFIG})
    lines = "".join(f"{sha256_file(root / f)}  {f}\n" for f in files)
    return sha256_text(lines), len(files)


# --- cluster maps ---------------------------------------------------------------------------------------------------


def graph_rule(graph: str, cfg: Any) -> tuple[dict[str, str] | None, str | None]:
    """A call's graph: its GRAPHS entry (None for the frozen config's own graph) and how its clusters are found, or None
    for a graph no call may name (cluster_map_for refuses it, and so does the check-stores command)."""
    spec = GRAPHS.get(graph)
    return spec, spec["map"] if spec else "config" if graph == cfg.graph else None


def unknown_graph(graph: str, cfg: Any) -> Refused:
    return Refused(f"{graph}: no cluster map (graphs {sorted(GRAPHS)} and the frozen config's {cfg.graph}); another "
                   f"graph's map is science's decision, never singletons by default")


def cluster_map_for(F: Frozen, cfg: Any, graph: str, private_map: Path | None = None) -> tuple[Any, dict]:
    """The graph's cluster map (by its pinned sha256: ClusterMap.load checks it against itself) and what it is; None for
    SIB's rule, each question its own cluster (prepare builds it over the call's index). A private map (a15) is given
    at run time, outside any git work tree and readable by its owner alone."""
    spec, rule = graph_rule(graph, cfg)
    if (private_map is not None) != (rule == "private"):
        raise Refused(f"{graph}: --cluster-map is " + ("required (its map is kept in the private tree only)"
                                                       if rule == "private" else "given only for a private map"))
    if rule == "config":
        return (F.prereg.load_clusters(cfg, F.root),
                {"graph": graph, "store": graph, "rule": "the frozen config's map", "path": cfg.cluster_map,
                 "sha256": cfg.cluster_map_sha256, "questions_sha256": cfg.frozen_questions()})
    if rule is None:
        raise unknown_graph(graph, cfg)
    record = {"graph": graph, "store": spec["store"], "questions_sha256": spec["questions_sha256"]}
    if rule == "committed":
        rel, sha = CLUSTER_MAPS[graph]
        return (F.prereg.ClusterMap.load(HERE / rel, sha),
                {**record, "rule": "built here by entry 52's rule", "path": f"results/{HERE.name}/{rel}", "sha256": sha})
    if rule == "singletons":
        return None, {**record, "rule": "each question its own cluster (no generation anchors; science, design log, 23 Oct "
                                        "00:57Z): nothing is read for it", "sha256": None}
    path = outside_git(Path(private_map))
    if path.stat().st_mode & 0o077:
        raise Refused(f"{graph}: its private map must be readable by its owner alone (mode 600)")
    return (F.prereg.ClusterMap.load(path, spec["map_sha256"]),
            {**record, "rule": "built by entry 52's rule, kept in the private tree", "path": "private",
             "sha256": spec["map_sha256"]})


def outside_git(path: Path) -> Path:
    """``path``, refused when it, or what it resolves to through any symlink, is inside a git work tree, or when it is
    itself a symlink (partition_run.outside_git, made stricter: private files never go near git)."""
    given = Path(path).absolute()
    if given.is_symlink():
        raise Refused(f"{given}: a symlink; a private file is named by its own path")
    for candidate in (given, given.resolve()):
        for d in candidate.parents:
            dot = d / ".git"
            if (dot.is_dir() and (dot / "HEAD").exists()) or (dot.is_file() and dot.read_text(
                    encoding="utf-8", errors="replace").startswith("gitdir:")):
                raise Refused(f"{given}: inside the git work tree {d}; a private file stays outside git")
    return given.resolve()


def cluster_counts(F: Frozen, cluster_map: Any, index: Any, cfg: Any,
                   K_clusters: np.ndarray | None = None) -> tuple[np.ndarray, Any]:
    """The per-question resample weights of §4.1's cluster bootstrap over ``index``: ClusterMap.resampling's B
    unstratified draws of the clusters (seed and R from the frozen config, or ``K_clusters`` over the clusters), each
    question taking its cluster's column; and the Clusters."""
    clusters = cluster_map.resampling(index, cfg, K_clusters)
    return clusters.matrix[:, clusters.group], clusters


def record_anchors(rows: Sequence[dict], params_of: Mapping[str, Sequence[str]]) -> tuple[dict, dict]:
    """Base question id -> its declared anchor entities (the record's anchors list) and its template, every question of
    the set; refused unless each record's anchors are exactly the entities its variant's declared anchor parameters
    name (the generator's manifest), so the record field is the frozen rule's anchor set."""
    anchors: dict[str, list[str]] = {}
    templates: dict[str, str] = {}
    wrong = []
    for r in rows:
        base = r.get("base_question_id") or r["id"]  # the analysis index's key (prereg.index_from_questions)
        if base in anchors:
            raise Refused(f"base question {base} appears twice in the question set")
        ents = [a.get("entity") for a in r.get("anchors") or []]
        if not all(isinstance(e, str) and e for e in ents):
            raise Refused(f"{r['id']}: an anchor without its entity")
        key = f"{r['template']}/{r['params']['variant']}"
        if key not in params_of or {str(r["params"][p]) for p in params_of[key]} != set(ents):
            wrong.append(r["id"])
        anchors[base], templates[base] = sorted(set(ents)), r["template"]
    if wrong:
        raise Refused(f"{len(wrong)} records' anchors are not their variant's declared anchor parameters, e.g. "
                      f"{wrong[:3]}")
    return anchors, templates


def components(pairs: Iterable[tuple[str, str]], nodes: Iterable[str]) -> dict[str, str]:
    """Connected components of ``pairs`` over ``nodes``, each named by its smallest node (an independent check of the
    map's clusters, from the generator's own shared-entity record)."""
    parent = {n: n for n in nodes}

    def root(n: str) -> str:
        while parent[n] != n:
            n = parent[n]
        return n

    for a, b in pairs:
        ra, rb = root(a), root(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    return {n: root(n) for n in parent}


def build_clusters(F: Frozen, out_dir: Path) -> dict:
    """Both maps, written by the frozen write_cluster_map into a new directory, reloaded by ClusterMap.load by their
    sha256, cross-checked, and summarised (counts only)."""
    out_dir = Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise Refused(f"{out_dir}: not empty; a built map is never overwritten")
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg, cfg_record = frozen_config(F)
    man_path, man_sha = RHEA_MANIFEST
    manifest_bytes = (F.root / man_path).read_bytes()
    if sha256_bytes(manifest_bytes) != man_sha:
        raise Refused(f"{man_path}: sha256 {sha256_bytes(manifest_bytes)}, not the pinned {man_sha}")
    manifest = json.loads(manifest_bytes)
    exclusion = manifest["test_anchor_exclusion"]
    params_of = {**exclusion["lookup_anchor_params"], **exclusion["anchor_params"]}
    if set(exclusion["lookup_anchor_params"]) & set(exclusion["anchor_params"]):
        raise Refused("the manifest's lookup and compositional anchor parameters overlap")
    built: dict[str, dict] = {}
    clusters_of: dict[str, dict[str, str]] = {}
    for graph, (rel, sha) in QUESTION_SETS.items():
        raw = (F.root / rel).read_bytes()
        if sha256_bytes(raw) != sha:
            raise Refused(f"{rel}: sha256 {sha256_bytes(raw)}, not the pinned {sha}")
        rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
        anchors, templates = record_anchors(rows, params_of)
        path = out_dir / f"{graph}.json"
        F.prereg.write_cluster_map(path, anchors, templates)
        map_sha = sha256_file(path)
        cm = F.prereg.ClusterMap.load(path, map_sha)
        clusters_of[graph] = dict(cm.clusters)
        test = F.prereg.index_from_questions({r["id"]: r for r in rows}, "test")
        _, test_clusters = cluster_counts(F, cm, test, cfg)
        shared = Counter((templates[q], a) for q in anchors for a in anchors[q])
        test_ids = set(test.questions)
        shared_test = Counter((templates[q], a) for q in test_ids for a in anchors[q])
        built[graph] = {
            "map": f"results/{HERE.name}/{out_dir.name}/{path.name}", "sha256": map_sha,
            "questions": {"path": rel, "sha256": sha}, "keyed_by": "base_question_id (the analysis index's key)",
            "anchors_from": "each question record's own anchors list (anchors[].entity), checked equal to its variant's "
                            "declared anchor parameters in the Rhea generator's manifest",
            "all": {"questions": len(anchors), "clusters": len(set(cm.clusters.values())),
                    "shared_anchors_within_template": sum(1 for n in shared.values() if n > 1)},
            "test": {**test_clusters.summary(),
                     "shared_anchors_within_template": sum(1 for n in shared_test.values() if n > 1)},
        }
    # Rhea: the generator's own shared-entity record (entity_reuse), within each template, gives the same components
    rhea_rows = {json.loads(line)["id"]: json.loads(line)
                 for line in (F.root / QUESTION_SETS["rhea-142"][0]).read_text(encoding="utf-8").splitlines()
                 if line.strip()}
    pairs = []
    for users in manifest["entity_reuse"]["shared"].values():
        by_template = defaultdict(list)
        for u in users:
            by_template[rhea_rows[u["id"]]["template"]].append(u["id"])
        pairs += [(ids[0], other) for ids in by_template.values() for other in ids[1:]]
    reuse = components(pairs, rhea_rows)
    differ = sorted(q for q in rhea_rows if reuse[q] != clusters_of["rhea-142"][q])
    if differ:
        raise Refused(f"Rhea's map differs from the generator's entity_reuse record for {len(differ)} questions, e.g. "
                      f"{differ[:3]}")
    built["rhea-142"]["cross_check_entity_reuse"] = {"pairs_within_template": len(pairs), "questions_differing": 0}
    same = sum(clusters_of["opaque-rhea-142"][q] == clusters_of["rhea-142"].get(q) for q in clusters_of["opaque-rhea-142"])
    built["opaque-rhea-142"]["same_clusters_as_rhea"] = {
        "questions": len(clusters_of["opaque-rhea-142"]), "same_cluster_id": same,
        "note": "by base question id; opaque's anchors are its own records' (no held-out map), so equality is the "
                "renaming's, observed"}
    record = {"schema": "e189-cluster-maps/1", "ruling": RULING,
              "rule": "prereg.connected_clusters (change-log entry 52): base questions of one template sharing a declared "
                      "anchor entity, closed under chains, over every question of the set (dev included); the analysis "
                      "restricts them to the analysed split (ClusterMap.resampling)",
              "code": F.identity, "config": cfg_record, "rhea_manifest": {"path": man_path, "sha256": man_sha},
              "maps": built}
    (out_dir / "build-record.json").write_text(json.dumps(record, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return record


def frozen_module(F: Frozen, name: str) -> ModuleType:
    """A frozen module imported from the code root, checked to come from there."""
    module = importlib.import_module(name)
    if not Path(module.__file__).resolve().is_relative_to(F.root):
        raise Refused(f"{name} was imported from {module.__file__}, not from --code-root {F.root}")
    return module


OTHER_KIND = "other_kind_anchor"  # harness/qgen/a15.py: the reading that pairs a label parameter with its entity's IRI


def label_entity_params(variant: Any) -> dict[str, list[str]]:
    """Label parameter -> the entity parameter(s) the variant's other_kind_anchor readings pair it with: that frozen
    helper's transform resolves the anchor's entity from params[entity_param], the IRI the gold names, and its label from
    params[label_param] (read from the transform's own free variables, by name)."""
    pairs: dict[str, list[str]] = defaultdict(list)
    for reading in getattr(variant, "readings", ()) or ():
        transform = getattr(reading, "transform", None)
        if reading.name != OTHER_KIND or transform is None or transform.__closure__ is None:
            continue
        cells = dict(zip(transform.__code__.co_freevars, (c.cell_contents for c in transform.__closure__), strict=True))
        if isinstance(cells.get("label_param"), str) and isinstance(cells.get("entity_param"), str):
            pairs[cells["label_param"]].append(cells["entity_param"])
    return pairs


def anchor_ids(generate: ModuleType, variant: Any, params: Mapping[str, Any], text: str) -> list[str]:
    """The entities a question names by its declared anchors (§4.1: any entity named by label, or a declared anchor),
    each as its IRI. An anchor with an entity parameter is read by generate.anchor_entities (as is when every anchor has
    one). An anchor declaring no entity parameter (a15's label_anchor, named by a label unique among its class's
    entities) is the IRI of the one entity parameter its label parameter is paired with by the variant's
    other_kind_anchor reading (label_entity_params); its label must be named in the question text. Anything else is
    refused, never keyed by its label."""
    if all(a.param for a in variant.anchors):
        return sorted(generate.anchor_entities(variant, params))
    paired = label_entity_params(variant)
    out = set()
    for a in variant.anchors:
        if a.param:
            out.add(str(params[a.param]))
            continue
        if len(a.label_params) != 1:
            raise Refused("a label anchor with other than one label parameter")
        label = a.label_params[0]
        entity = paired.get(label, [])
        iri = params.get(entity[0]) if len(entity) == 1 else None
        if not (isinstance(iri, str) and iri.startswith(("http://", "https://"))):
            raise Refused("a label anchor whose entity is not one IRI parameter paired by its variant's other_kind_anchor "
                          "reading")
        if not str(params.get(label) or "") or str(params[label]) not in text:
            raise Refused("a label anchor's label is not named in its question's text")
        out.add(iri)
    return sorted(out)


def template_anchors(F: Frozen, graph: str, rows: Sequence[dict]) -> tuple[dict, dict]:
    """Base question id -> the entities its declared anchors name (anchor_ids, the variant from the frozen generator's
    profiles.TEMPLATE_SETS[graph]), and its template; every question of the set."""
    generate = frozen_module(F, "harness.qgen.generate")
    by_id = {t.id: t for t in frozen_module(F, "harness.qgen.profiles").TEMPLATE_SETS[graph]}
    anchors: dict[str, list[str]] = {}
    templates: dict[str, str] = {}
    for r in rows:
        base = r.get("base_question_id") or r["id"]  # the analysis index's key (prereg.index_from_questions)
        if base in anchors:
            raise Refused(f"a base question appears twice in the {graph} question set")
        try:
            variant = by_id[r["template"]].variant(r["params"]["variant"])
            anchors[base] = anchor_ids(generate, variant, r["params"], r["question"])
        except (KeyError, generate.TemplateError) as exc:
            raise Refused(f"a {graph} question's variant or anchor is not the frozen generator's ({type(exc).__name__})"
                          ) from None
        templates[base] = r["template"]
    return anchors, templates


def check_manifest(manifest: Mapping[str, Any], anchors: Mapping[str, Sequence[str]], templates: Mapping[str, str],
                   question_of: Mapping[str, str], require_anchor_reuse: bool = False) -> list[str]:
    """The anchors against the generator's own records, naming no entity: its anchor_reuse (where the profile wrote one:
    per template the questions, the distinct anchors and the anchors two questions share) must be equal; its
    entity_reuse (every IRI-valued parameter: a superset of the IRI anchors; an anchor that is a key, not an IRI, is not
    in it) must hold every IRI anchor two questions share, with those questions. Returns the checks passed."""
    owners: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    users: dict[str, set[str]] = defaultdict(set)
    for base in sorted(anchors, key=lambda b: question_of[b]):
        for e in anchors[base]:
            owners[templates[base]][e].append(question_of[base])
            users[e].add(question_of[base])
    passed = []
    by_template = (manifest.get("anchor_reuse") or {}).get("by_template")
    if by_template is None and require_anchor_reuse:
        raise Refused("the manifest has no anchor_reuse record, which this graph's generator writes")
    if by_template is not None:
        mine = {t: {"questions": sum(1 for b in anchors if templates[b] == t), "anchor_entities": len(owners[t]),
                    "reused": {e: ids for e, ids in owners[t].items() if len(ids) > 1}} for t in sorted(set(templates.values()))}
        theirs = {t: {k: d.get(k) for k in ("questions", "anchor_entities", "reused")} for t, d in by_template.items()}
        if mine != theirs:
            raise Refused("the anchors read from the generator's templates are not its manifest's anchor_reuse record "
                          "(per template: questions, distinct anchors or shared anchors differ)")
        passed.append("anchor_reuse: equal")
    reuse = manifest.get("entity_reuse")
    shared = {e: {u["id"] for u in us} for e, us in ((reuse or {}).get("shared") or {}).items()}
    iris = {e: q for e, q in users.items() if e.startswith(("http://", "https://"))}
    if not isinstance(reuse, dict) or any(len(q) > 1 and not q <= shared.get(e, set()) for e, q in iris.items()) \
            or len(iris) > reuse.get("entities", -1):
        raise Refused("the anchors read from the generator's templates are not within its manifest's entity_reuse "
                      "record (an anchor two questions share is missing there, with them)")
    passed.append("entity_reuse: every shared IRI anchor recorded with its questions")
    return passed


def build_template_map(F: Frozen, graph: str, questions: Path, manifest: Path, out: Path, record_out: Path,
                       private: bool) -> dict:
    """One graph's map by entry 52's rule from the frozen generator's anchors (template_anchors), cross-checked with its
    manifest, written by write_cluster_map (mode 600 outside git when ``private``: a15's stays in the private tree),
    reloaded by its sha256; the record holds hashes and cluster counts only."""
    if graph not in TEMPLATE_MAP_GRAPHS or private != (GRAPHS[graph]["map"] == "private"):
        raise Refused(f"{graph}: built here from templates only for {TEMPLATE_MAP_GRAPHS}, privately exactly when its map "
                      f"is private")
    cfg, cfg_record = frozen_config(F)
    raw = Path(questions).read_bytes()
    if sha256_bytes(raw) != GRAPHS[graph]["questions_sha256"]:
        raise Refused(f"{questions}: sha256 {sha256_bytes(raw)[:12]}, not {graph}'s question set "
                      f"{GRAPHS[graph]['questions_sha256'][:12]}")
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    anchors, templates = template_anchors(F, graph, rows)
    passed = check_manifest(read_json(manifest), anchors, templates,
                            {r.get("base_question_id") or r["id"]: r["id"] for r in rows},
                            require_anchor_reuse=graph in ANCHOR_REUSE_GRAPHS)
    out, record_out = Path(out), Path(record_out)
    if private:  # the map and its full build record both stay in the private tree
        out, record_out = outside_git(out), outside_git(record_out)
    tmp = out.with_name(out.name + ".tmp")  # write_cluster_map's temporary file, renamed onto the map
    if any(os.path.lexists(p) for p in (out, record_out, tmp)):  # lexists: a dangling symlink counts too
        raise Refused(f"{out}, its temporary {tmp.name} or {record_out} exists; a built map is never overwritten, and "
                      f"nothing already there is written through")
    mask = os.umask(0o077) if private else None  # the map and the record are created mode 600
    try:
        F.prereg.write_cluster_map(out, anchors, templates)
        record_out.touch()
    finally:
        if mask is not None:
            os.umask(mask)
    map_sha = sha256_file(out)
    cm = F.prereg.ClusterMap.load(out, map_sha)
    _, test = cluster_counts(F, cm, F.prereg.index_from_questions({r["id"]: r for r in rows}, "test"), cfg)
    record = {"schema": "e189-cluster-map/1", "ruling": "science, the design log Oct 00:57Z: by the same frozen "
                                                        "rule" + (", kept in the private tree only" if private else ""),
              "graph": graph, "rule": "prereg.connected_clusters (change-log entry 52) over every question of the set, "
                                      "dev included",
              "anchors_from": "the frozen generator's templates (profiles.TEMPLATE_SETS, generate.anchor_entities)",
              "manifest_checks_passed": passed,
              "questions_sha256": sha256_bytes(raw), "manifest_sha256": sha256_file(manifest), "map_sha256": map_sha,
              "kept": "the private tree only (mode 600)" if private else f"results/{HERE.name}/clusters/{out.name}",
              "all": {"questions": len(anchors), "clusters": len(set(cm.clusters.values()))}, "test": test.summary(),
              "code": {"commit": F.identity["commit"], "code_digest": F.identity["code_digest"]},
              "config_sha256": cfg_record["config_sha256"]}
    record_out.write_text(json.dumps(record, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return record


# --- inputs: voided OUTs, partitions ---------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Out:
    path: Path
    identity: dict | None
    manifest: dict

    @classmethod
    def read(cls, path: Path, manifest_sha256: str | None = None) -> Out:
        path = Path(path)
        ident = path / "deployment-identity.json"
        identity = read_json(ident) if ident.is_file() else None
        if manifest_sha256 is None:
            manifest = read_json(path / "manifest.json")
        else:
            try:
                raw = (path / "manifest.json").read_bytes()
            except OSError as exc:
                raise Refused(f"{path}: cannot read pinned manifest.json") from exc
            if sha256_bytes(raw) != manifest_sha256:
                raise Refused(f"{path}: manifest.json sha256 differs from the OUT pin")
            try:
                manifest = json.loads(raw)
            except (ValueError, UnicodeDecodeError) as exc:
                raise Refused(f"{path}: pinned manifest.json is not JSON") from exc
        return cls(path, identity, manifest)

    def names(self) -> set[str]:
        """Every path this OUT is known by: here, and on its host as its identity records it."""
        found = {str(self.path.absolute()), str(self.path.resolve())}
        ident = self.identity or {}
        found |= {p for p in (ident.get("out"), (ident.get("store") or {}).get("out")) if isinstance(p, str) and p}
        return found

    @property
    def partition(self) -> dict | None:
        return (self.identity or {}).get("partition")


def read_voided(paths: Sequence[Path]) -> tuple[set[str], list[dict]]:
    """The voided OUTs' on-host paths, from each voided-run record (voided.jsonl: one object per line with ``out``)."""
    if not paths:
        raise Refused("no voided-run record given (--voided voided.jsonl): a voided OUT is never analysed")
    outs: set[str] = set()
    files = []
    for path in paths:
        lines = [line for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
        if not lines:
            raise Refused(f"{path}: no entries; pass the voided-run record itself")
        for i, line in enumerate(lines, 1):
            entry = json.loads(line)
            if not isinstance(entry, dict) or not isinstance(entry.get("out"), str) or not entry["out"]:
                raise Refused(f"{path}:{i}: a voided-run entry without its OUT (out)")
            outs.add(entry["out"])
        files.append({"path": str(path), "sha256": sha256_file(path), "entries": len(lines)})
    lacking = [o for o in KNOWN_VOIDED if o not in outs]
    if lacking:
        raise Refused(f"the voided-run record(s) lack {len(lacking)} of the {len(KNOWN_VOIDED)} voided OUTs known when this "
                      f"script was written (entries 173 and 180): not the voided-run record")
    return outs, files


def check_not_voided(outs: Sequence[Out], voided: set[str]) -> None:
    hit = [(str(o.path), sorted(o.names() & voided)) for o in outs if o.names() & voided]
    if hit:
        raise Refused(f"voided OUTs are never analysed (voided.jsonl): {hit}")


def store_chain(identity: Mapping[str, Any]) -> tuple[list[dict], str | None]:
    """The store copies an OUT's deployment identity names, in order: its first copy, then each identity store change's
    new copy (store_move_remote.py identity: ``from`` the store block it replaced, ``to`` the new one, now the identity's
    ``store``), and why the chain does not join up (None when it does)."""
    store = identity.get("store") if isinstance(identity.get("store"), dict) else {}
    changes = identity.get("store_changes") or []
    if not isinstance(changes, list):
        return [store], "its identity's store_changes is not a list"
    if not changes:
        return [store], None
    chain = [changes[0].get("from") if isinstance(changes[0], dict) else None]
    for i, ch in enumerate(changes, 1):
        if not isinstance(ch, dict) or not isinstance(ch.get("from"), dict) or not isinstance(ch.get("to"), dict):
            return [store], "its identity's store change lacks its from/to store blocks"
        if ch["from"] != chain[-1]:
            return [store], "its identity's store change does not start at the copy the one before it moved to"
        chain.append(ch["to"])
    if chain[-1] != store:
        return [store], "its identity's store block is not its last identity store change's copy"
    return chain, None


@dataclasses.dataclass(frozen=True)
class Union:
    """A partition union, checked: the record, its sha256, K, and each OUT's partition index."""

    record: dict
    record_sha256: str
    k: int
    index_of: dict[str, int]  # OUT path -> its partition
    run_partition: dict[str, int]  # every record's run id (prompt arms and branch cells) -> its OUT's partition
    subset: tuple[int, ...] | None = None  # --partitions, when a strict subset of the record's partitions (else None)
    binding: dict | None = None  # --partition-binding, when cited: its sha256 and each partition's plan (else None)
    weights = None  # (not a field, so asdict is the base's for every union) the mod-K rule; WeightedUnion carries a record's weights


@dataclasses.dataclass(frozen=True)
class WeightedUnion(Union):
    """A plain partition union of a WEIGHTED record: a Union that also carries the record's weights, which every membership
    decision of the union takes (partition_of's ``weights``)."""

    weights: tuple[int, ...] = ()


def union_weights(record: Mapping[str, Any]) -> tuple[int, ...] | None:
    """The weights of a union's record: those of a weighted plain record (its normalized form carries them), else None (the
    mod-K rule; a record of any other schema is never read for weights)."""
    if record.get("schema") != PLAIN_PARTITION_SCHEMA or "weights" not in record:
        return None
    return tuple(record["weights"])


def read_record(path: Path) -> tuple[dict, str]:
    raw = Path(path).read_bytes()
    record = json.loads(raw)
    schema = record.get("schema") if isinstance(record, dict) else None
    sha = sha256_bytes(raw)
    if isinstance(record, dict) and schema in PINNED_RECORDS:
        if sha != PINNED_RECORDS[schema]:
            raise Refused(f"{path}: schema {schema!r} is accepted only as the one pinned file "
                          f"({PINNED_RECORDS[schema][:12]}), this file is {sha[:12]}")
    elif not isinstance(record, dict) or schema not in RECORD_SCHEMAS:
        raise Refused(f"{path}: schema {schema!r}, not one of {RECORD_SCHEMAS} (nor a pinned record's: "
                      f"{sorted(PINNED_RECORDS)})")
    return record, sha


def read_plain_partition_record(path: Path, pin: str) -> tuple[dict, str]:
    """Read the producer's exact schema, with the file pinned by the caller's sha256."""
    if re.fullmatch(r"[0-9a-f]{64}", pin) is None:
        raise Refused("--plain-partition-record-sha256 needs 64 lower-case hex digits")
    raw = Path(path).read_bytes()
    sha = sha256_bytes(raw)
    if sha != pin:
        raise Refused("the plain partition record is not the command-line pinned file")
    record = json.loads(raw)
    weighted = (isinstance(record, dict) and set(record) == WEIGHTED_PLAIN_RECORD_FIELDS
                and record["rule"] == WEIGHTED_PLAIN_RULE)
    if not isinstance(record, dict) or set(record) != (WEIGHTED_PLAIN_RECORD_FIELDS if weighted else PLAIN_RECORD_FIELDS):
        raise Refused("the plain partition record has an extra or missing field")
    k, copies, parts = record["k"], record["copies"], record["partitions"]
    if (type(k) is not int or k < 2 or not isinstance(copies, list) or len(copies) != k
            or len(set(map(str, copies))) != k or not all(_plain_name(c) for c in copies)
            or not isinstance(parts, dict) or set(parts) != {str(i) for i in range(k)}
            or any(not isinstance(p, dict) or set(p) != PLAIN_PART_FIELDS for p in parts.values())):
        raise Refused("the plain partition record does not have K exact partitions and copies")
    if (record["rule"] != (WEIGHTED_PLAIN_RULE if weighted else PLAIN_RULE) or record["salt"] != SALT or record["reps"] != 3
            or type(record["test_questions"]) is not int or record["test_questions"] < 1
            or not isinstance(record["questions_sha256"], str)
            or re.fullmatch(r"[0-9a-f]{64}", record["questions_sha256"]) is None):
        raise Refused("the plain partition record's rule or registered question pin is invalid")
    if weighted:
        weights, reduced = record["weights"], record["weights_reduced"]
        if (not isinstance(weights, list) or len(weights) != k
                or any(type(w) is not int or w < 1 for w in weights)):
            raise Refused("the weighted plain partition record's weights are not one positive integer per partition")
        gcd = math.gcd(*weights)
        if (not isinstance(reduced, list) or any(type(w) is not int for w in reduced)
                or reduced != [w // gcd for w in weights]):
            raise Refused("the weighted plain partition record's weights_reduced are not its weights divided by their gcd")
    for i in range(k):
        p = parts[str(i)]
        if (p["copy"] != copies[i] or type(p["questions"]) is not int or p["questions"] < 1
                or type(p["attempts"]) is not int or p["attempts"] != 3 * p["questions"]
                or p["file"] != f"questions-p{i}of{k}.jsonl"
                or any(not isinstance(p[f], str) or re.fullmatch(r"[0-9a-f]{64}", p[f]) is None
                       for f in ("ids_sha256", "file_sha256"))):
            raise Refused(f"the plain partition record's partition {i} is malformed")
    return record, sha


def plain_record_weights(record: Mapping[str, Any]) -> list[int] | None:
    """The weights a plain record's questions are placed by: its own when its rule is the weighted rule (refused when it has
    none: never the mod-K rule by default), else None (the mod-K rule; a stray ``weights`` key of a mod-K record is not read)."""
    if record.get("rule") != WEIGHTED_PLAIN_RULE:
        return None
    weights = record.get("weights")
    if weights is None:
        raise Refused("the weighted plain partition record has no weights")
    return weights


def plain_partition_files(record: dict, record_path: Path, questions_path: Path,
                          split: str) -> tuple[list[Path], dict]:
    """Rebuild the producer's byte-for-byte test files from the registered questions file."""
    raw = Path(questions_path).read_bytes()
    if sha256_bytes(raw) != record["questions_sha256"] or split != REGISTERED_SPLIT:
        raise Refused("the plain partition record is not for the registered test questions file")
    k = record["k"]
    weights = plain_record_weights(record)
    rows: dict[int, list[tuple[str, bytes]]] = {i: [] for i in range(k)}
    seen: set[str] = set()
    for line in raw.split(b"\n"):
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("split") != split:
            continue
        qid = row["id"]
        if not isinstance(qid, str) or qid in seen or row.get("base_question_id") not in (None, qid):
            raise Refused("the registered test questions do not have unique producer-compatible ids")
        seen.add(qid)
        rows[partition_of(qid, k, weights)].append((qid, line))
    if not seen or len(seen) != record["test_questions"]:
        raise Refused("the plain partition record does not cover the registered test questions")
    paths = []
    for i in range(k):
        p = record["partitions"][str(i)]
        mine = rows[i]
        body = b"".join(line + b"\n" for _, line in mine)
        path = Path(record_path).parent / p["file"]
        if (p["questions"] != len(mine) or p["ids_sha256"] != ids_digest(q for q, _ in mine)
                or p["file_sha256"] != sha256_bytes(body)):
            raise Refused(f"the plain partition record's partition {i} is not the producer's rule")
        if sha256_file(path) != p["file_sha256"] or path.read_bytes() != body:
            raise Refused(f"partition {i}'s questions file differs from the registered rows")
        paths.append(path)
    normalized = {"schema": PLAIN_PARTITION_SCHEMA, "salt": SALT, "graph": None, "split": split,
                  "questions": {"sha256": record["questions_sha256"]}, "all_ids_sha256": ids_digest(seen),
                  "n_questions": len(seen), "k": k,
                  "partitions": [{"index": i, "copy": record["copies"][i], "n_questions": len(rows[i]),
                                  "ids_sha256": record["partitions"][str(i)]["ids_sha256"],
                                  "question_id_sha256": sorted(sha256_text(q) for q, _ in rows[i])}
                                 for i in range(k)]}
    if weights is not None:  # only a weighted record's normalized form has the key: a mod-K record's is the base's, byte for byte
        normalized["weights"] = list(weights)
    return paths, normalized


def parse_plain_union_outs(spec: str, batches: Sequence[Path]) -> tuple[str, tuple[str, ...]]:
    """Bind the ordered batch paths to an independently supplied prefix and manifest pins."""
    prefix, sep, hashes = spec.partition(":")
    if (not sep or not prefix.startswith("/") or prefix.endswith("/") or
            any(part in ("", ".", "..") for part in prefix.split("/")[1:]) or
            re.search(r"[\s|*?\[\]:,]", prefix)):
        raise Refused("--plain-union-outs needs an absolute clean OUT prefix and sha256 pins")
    pins = tuple(hashes.split(","))
    if not pins or any(re.fullmatch(r"[0-9a-f]{64}", pin) is None for pin in pins):
        raise Refused("--plain-union-outs needs one lower-case manifest sha256 per partition")
    if len(batches) != len(pins):
        raise Refused("--plain-union-outs pin count differs from the OUT count")
    if any(str(path) != f"{prefix}-p{i}of{len(pins)}" for i, path in enumerate(batches)):
        raise Refused("--plain-union-outs requires OUTs in exact partition index order")
    return prefix, pins


def plain_union_outs(outs: Sequence[Out], record: dict, source_parts: dict, record_sha: str,
                     prefix: str) -> list[Out]:
    """Identify each plain OUT by its name and fingerprint, then supply the checked partition view."""
    file_sha = {p["file_sha256"]: int(i) for i, p in source_parts.items()}
    if len(file_sha) != record["k"] or len(outs) != record["k"]:
        raise Refused("a plain partition union needs one OUT per distinct partition")
    adapted = []
    for out in outs:
        if out.partition is not None:
            raise Refused(f"{out.path}: a plain partition OUT must have no partition block")
        if out.manifest.get("schema") != "pilot-manifest/1":
            raise Refused(f"{out.path}: a plain partition union needs a plain-run manifest")
        fingerprint = out.manifest.get("fingerprint")
        sha = fingerprint.get("questions_sha256") if isinstance(fingerprint, dict) else None
        if sha not in file_sha:
            raise Refused(f"{out.path}: its plain fingerprint does not pin a partition questions file")
        i = file_sha[sha]
        if str(out.path) != f"{prefix}-p{i}of{record['k']}":
            raise Refused(f"{out.path}: its OUT name does not match partition {i} from its questions fingerprint")
        named_out = (out.identity or {}).get("out")
        if not isinstance(named_out, str) or Path(named_out).name != out.path.name:
            raise Refused(f"{out.path}: its deployment identity does not name this partition OUT")
        identity = dict(out.identity or {})
        identity["partition"] = {"index": i, "k": record["k"], "record_sha256": record_sha,
                                 "ids_sha256": record["partitions"][i]["ids_sha256"],
                                 "copy": record["partitions"][i]["copy"]}
        adapted.append(Out(out.path, identity, out.manifest))
    return adapted


def require_plain_private_graph(graph: str) -> None:
    if graph not in PRIVATE_REFERENCE_PINS:
        raise Refused("a plain partition union is allowed only for a privately pinned graph")


def require_plain_pinned_copies(record: dict, graph: str, pin: Mapping[str, str] | None,
                                private_classes: PrivateTable | None, private_endpoints: PrivateTable | None) -> None:
    """Design ruling (b), 25 Oct: a plain partition union's per-copy rule is met by each partition's manifest identity plus
    its copy's pinned store description. The description is the privately pinned graph's reference pin and its two pinned
    private tables (copy classes, endpoint copies), all given: without any of them nothing pins the copies, so the call is
    refused. Every partition's copy in the record must be listed in the copy classes and be a replica of the endpoint copies
    (of this graph). The refusals name no copy, endpoint, path or table content, only the partitions whose copy is unlisted."""
    if pin is None or private_classes is None or private_endpoints is None:
        raise Refused("a plain partition union needs the privately pinned graph's reference pin, private copy classes and "
                      "private endpoint copies (--reference-pin, --private-copy-classes, --private-endpoint-copies)")
    classed = set(private_classes.doc["copies"])
    replicas = {e["replica"] for e in private_endpoints.doc if e["graph"] == graph}
    unlisted = [i for i, part in enumerate(record["partitions"])
                if part["copy"] not in classed or part["copy"] not in replicas]
    if unlisted:
        raise Refused(f"a plain partition union's copies must be listed in the pinned copy classes and endpoint copies: "
                      f"not listed for partitions {unlisted}")


@dataclasses.dataclass(frozen=True)
class BoundPlan:
    """One partition's plan as a cited binding gives it: the OUT it names, its store copy, and the plan file's sha256."""

    partition: int
    out: str
    copy: str
    sha256: str


@dataclasses.dataclass(frozen=True)
class Binding:
    sha256: str
    plans: dict[int, BoundPlan]  # partition index -> its plan


def _plain_name(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", value) is not None


def read_binding(binding_path: Path, plan_paths: Sequence[Path], record: dict, record_sha256: str) -> Binding:
    """A binding cited for a partition record: the store copies the OUTs ran on come from its plans instead of the record's list
    (DeepSeek's dsh partitions). Accepted only when ALL hold; any other case is Refused, never a fallback to the record's copies:
    the binding and each plan are in their schemas; the binding's record is the cited record (sha256, K, rule, question count and
    every partition's index, count and ids digest); there is exactly one plan per partition 0..K-1, each of the binding's K and
    naming the binding's sha256, its partition's question count, a plain OUT name (distinct among the plans) and a store copy that
    is one of the binding's copies. The OUT-level match (its identity names this binding and plan, its OUT name, its copy) is
    check_out_partition's."""
    raw = Path(binding_path).read_bytes()
    binding = json.loads(raw)
    bsha = sha256_bytes(raw)
    if not isinstance(binding, dict) or binding.get("schema") != BINDING_SCHEMA:
        raise Refused(f"{binding_path}: not a binding in schema {BINDING_SCHEMA!r}")
    k, parts = record.get("k"), record.get("partitions")
    if (type(k) is not int or k < 1 or not isinstance(parts, list) or len(parts) != k
            or not all(isinstance(p, dict) for p in parts) or [p.get("index") for p in parts] != list(range(k))):
        raise Refused("the cited record has no K partitions to bind")
    if type(binding.get("k")) is not int or binding["k"] != k:
        raise Refused(f"the binding's K is {binding.get('k')!r}, not the record's {k!r}")
    brec = binding.get("record")
    if not isinstance(brec, dict) or brec.get("sha256") != record_sha256:
        raise Refused("the binding is not for the cited partition record (its record.sha256)")
    want = [[p.get("index"), p.get("n_questions"), p.get("ids_sha256")] for p in parts]
    if brec.get("partitions") != want or brec.get("rule") != record.get("rule") or brec.get("n_questions") != record.get("n_questions"):
        raise Refused("the binding's record block (rule, question count, partitions' index/count/ids digest) is not the cited "
                      "record's")
    copies = binding.get("copies")
    if not isinstance(copies, dict) or not copies or not all(_plain_name(c) for c in copies):
        raise Refused("the binding names no store copies")
    if len(plan_paths) != k:
        raise Refused(f"the binding needs exactly one plan per partition ({k}), not {len(plan_paths)} (--partition-plan)")
    plans: dict[int, BoundPlan] = {}
    for pp in plan_paths:
        praw = Path(pp).read_bytes()
        plan = json.loads(praw)
        if not isinstance(plan, dict) or plan.get("schema") != PLAN_SCHEMA:
            raise Refused(f"{pp}: not a plan in schema {PLAN_SCHEMA!r}")
        i = plan.get("partition")
        if type(i) is not int or not 0 <= i < k or i in plans:
            raise Refused(f"{pp}: its partition {i!r} is not one of 0..{k - 1} named once")
        if plan.get("binding_sha256") != bsha:
            raise Refused(f"{pp}: its binding_sha256 is not the cited binding's")
        if type(plan.get("k")) is not int or plan["k"] != k or plan.get("n_questions") != parts[i].get("n_questions"):
            raise Refused(f"{pp}: its K or question count is not partition {i}'s of the record")
        if not _plain_name(plan.get("out")):
            raise Refused(f"{pp}: its OUT is not a plain name")
        if not _plain_name(plan.get("copy")) or plan["copy"] not in copies:
            raise Refused(f"{pp}: its store copy is not one of the binding's copies")
        plans[i] = BoundPlan(i, plan["out"], plan["copy"], sha256_bytes(praw))
    if sorted(plans) != list(range(k)) or len({pl.out for pl in plans.values()}) != k:
        raise Refused("the binding's plans are not one per partition with distinct OUTs")
    return Binding(bsha, plans)


def check_record(record: dict, design_ids: Sequence[str], graph: str, split: str, questions_sha256: str) -> int:
    """The record follows entry 179's rule over the question set: its salt, graph, split and question file; its full id
    set is the design's; each partition's count, ids digest and sorted id hashes are the rule's."""
    k = record.get("k")
    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        raise Refused(f"the record's K is {k!r}")
    want = {"salt": SALT, "graph": graph, "split": split}
    got = {f: record.get(f) for f in want}
    if got != want or (record.get("questions") or {}).get("sha256") != questions_sha256:
        raise Refused(f"the record is of {got} on questions {(record.get('questions') or {}).get('sha256')}, not "
                      f"{want} on {questions_sha256}")
    if record.get("all_ids_sha256") != ids_digest(design_ids) or record.get("n_questions") != len(design_ids):
        raise Refused("the record's question set is not the analysed split's (all_ids_sha256 / n_questions)")
    parts = record.get("partitions")
    if not isinstance(parts, list) or [p.get("index") for p in parts] != list(range(k)):
        raise Refused(f"the record's partitions are not 0..{k - 1} in order")
    for p in parts:
        mine = [q for q in design_ids if partition_of(q, k) == p["index"]]
        if (p.get("n_questions") != len(mine) or p.get("ids_sha256") != ids_digest(mine)
                or p.get("question_id_sha256") != sorted(sha256_text(q) for q in mine)):
            raise Refused(f"the record's partition {p['index']} is not the rule's (sha256('{SALT}' + id) mod {k})")
    return k


def check_record_weighted(record: dict, design_ids: Sequence[str], graph: str, split: str, questions_sha256: str,
                          weights: Sequence[int]) -> int:
    """check_record for a weighted plain record's normalized form: every check of check_record, with each partition's questions
    taken by the weighted rule (partition_of's ``weights``). check_record itself is the base's, byte for byte, and serves
    every mod-K and every other record."""
    k = record.get("k")
    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        raise Refused(f"the record's K is {k!r}")
    want = {"salt": SALT, "graph": graph, "split": split}
    got = {f: record.get(f) for f in want}
    if got != want or (record.get("questions") or {}).get("sha256") != questions_sha256:
        raise Refused(f"the record is of {got} on questions {(record.get('questions') or {}).get('sha256')}, not "
                      f"{want} on {questions_sha256}")
    if record.get("all_ids_sha256") != ids_digest(design_ids) or record.get("n_questions") != len(design_ids):
        raise Refused("the record's question set is not the analysed split's (all_ids_sha256 / n_questions)")
    parts = record.get("partitions")
    if not isinstance(parts, list) or [p.get("index") for p in parts] != list(range(k)):
        raise Refused(f"the record's partitions are not 0..{k - 1} in order")
    for p in parts:
        mine = [q for q in design_ids if partition_of(q, k, weights) == p["index"]]
        if (p.get("n_questions") != len(mine) or p.get("ids_sha256") != ids_digest(mine)
                or p.get("question_id_sha256") != sorted(sha256_text(q) for q in mine)):
            raise Refused(f"the record's partition {p['index']} is not the rule's (sha256('{SALT}' + id) mod the sum of "
                          f"the weights {list(weights)} divided by their gcd, by weight interval)")
    return k


def parse_partitions(text: str) -> tuple[int, ...]:
    """--partitions: comma-separated partition indices, each once, sorted (an empty, repeated or non-numeric entry is
    refused, never dropped)."""
    items = text.split(",")
    if not all(re.fullmatch(r"0|[1-9][0-9]*", i) for i in items):
        raise argparse.ArgumentTypeError(f"--partitions {text!r}: not comma-separated partition indices (e.g. 1,3,5)")
    found = tuple(int(i) for i in items)
    if len(set(found)) != len(found):
        raise argparse.ArgumentTypeError(f"--partitions {text!r}: a partition named twice")
    return tuple(sorted(found))


def named_partitions(subset: Sequence[int] | None, k: int) -> tuple[int, ...]:
    """The partitions a call analyses: all of the record's 0..K-1 (no option, or every partition named), else the strict
    subset named, which must be indices of the record, each once, and at least two (a partition is never analysed
    alone)."""
    whole = tuple(range(k))
    if subset is None:
        return whole
    found = tuple(subset)
    if any(not isinstance(p, int) or isinstance(p, bool) for p in found) or len(set(found)) != len(found):
        raise Refused(f"--partitions {list(found)}: not partition indices, each once")
    if any(not 0 <= p < k for p in found):
        raise Refused(f"--partitions {sorted(found)}: not all of 0..{k - 1}, the record's partitions")
    if tuple(sorted(found)) == whole:
        return whole
    if len(found) < 2:
        raise Refused(f"--partitions {sorted(found)}: a partition is analysed only in a union of at least two")
    return tuple(sorted(found))


def binding_problems(o: Out, part: dict, binding: Binding | None, p: int) -> list[str]:
    """An OUT's partition block against the cited binding. A cited binding binds only OUTs whose block names it and the plan of
    their partition, and whose name is that plan's OUT. With no binding cited, a block that names a binding or a plan is
    refused (the record's copies would otherwise be read for an OUT that ran on a plan's copy)."""
    if binding is None:
        named = [f for f in ("binding_sha256", "plan_sha256") if f in part]
        return ([f"{o.path}: its partition block names {' and '.join(named)}, but no binding is cited "
                 f"(--partition-binding, --partition-plan)"] if named else [])
    plan = binding.plans[p]
    found: list[str] = []
    if part.get("binding_sha256") != binding.sha256:
        found.append(f"{o.path}: its partition block does not name the cited binding "
                     f"(binding_sha256 {str(part.get('binding_sha256'))[:12]}, cited {binding.sha256[:12]})")
    if part.get("plan_sha256") != plan.sha256:
        found.append(f"{o.path}: its partition block does not name partition {p}'s plan "
                     f"(plan_sha256 {str(part.get('plan_sha256'))[:12]}, cited {plan.sha256[:12]})")
    out = (o.identity or {}).get("out")
    if o.path.name != plan.out or not isinstance(out, str) or Path(out).name != plan.out:
        found.append(f"{o.path}: its name or its identity's out is not partition {p}'s plan OUT ({plan.out})")
    return found


def check_union(F: Frozen, outs: Sequence[Out], record: dict, record_sha256: str, design_ids: Sequence[str],
                graph: str, split: str, questions_sha256: str, subset: Sequence[int] | None = None,
                binding: Binding | None = None, *, identity_note: str | None = None) -> Union:
    """Entry 189 item 5's refusals before any record is analysed (the module docstring lists them). ``subset``
    (--partitions): the union of exactly these partitions of the record; the record itself is checked in full.
    ``binding`` (--partition-binding): the store copy each OUT must have run on is its partition's plan's, not the record's;
    every other rule stands. A binding is cited whole (one plan per partition of the record) for the pooled union or for a
    proper ``subset``; naming every partition as a subset is refused with a binding (the pooled call is the one without
    --partitions), and the OUTs of the named partitions alone are matched to their plans.
    ``identity_note`` (a plain partition union alone, design ruling (b), 25 Oct): each store block's identity_check must
    equal this string exactly, in place of IDENTITY_CHECK; the pinned-description half of the ruling is
    require_plain_pinned_copies'. Without it every store block needs IDENTITY_CHECK, as for every other union."""
    weights = union_weights(record)  # None for every mod-K and every non-plain record: check_record, the base's rule
    k = (check_record(record, design_ids, graph, split, questions_sha256) if weights is None
         else check_record_weighted(record, design_ids, graph, split, questions_sha256, weights))
    want = named_partitions(subset, k)
    if binding is not None and subset is not None and want == tuple(range(k)):
        raise Refused("--partitions names every partition of the record: with a cited binding that is the pooled call, which is "
                      "made without --partitions (a binding takes --partitions only for a proper subset)")
    if binding is not None and sorted(binding.plans) != list(range(k)):
        raise Refused("the cited binding does not hold one plan per partition of the record")
    parts = record["partitions"]
    index_of: dict[str, int] = {}
    problems: list[str] = []
    for o in outs:
        part = o.partition
        if not isinstance(part, dict):
            problems.append(f"{o.path}: no partition block in its deployment identity, so it is no partition OUT (GLM's "
                            f"source natural OUT supplies the parents through the partitions' copies and is never "
                            f"analysed beside them)")
            continue
        p = part.get("index")
        if (part.get("k") != k or part.get("record_sha256") != record_sha256 or not isinstance(p, int)
                or isinstance(p, bool) or not 0 <= p < k or part.get("ids_sha256") != parts[p]["ids_sha256"]):
            problems.append(f"{o.path}: its partition block (k {part.get('k')}, index {p}, record "
                            f"{str(part.get('record_sha256'))[:12]}) is not a partition of the record "
                            f"{record_sha256[:12]} (K = {k})")
            continue
        if p not in want:
            problems.append(f"{o.path}: partition {p}, which --partitions {list(want)} does not name (a subset's union "
                            f"holds only the named partitions)")
            continue
        index_of[str(o.path)] = p
        problems += binding_problems(o, part, binding, p)
        copy_ = parts[p].get("copy") if binding is None else binding.plans[p].copy  # a cited binding's plan, never a fallback
        chain, why = store_chain(o.identity or {})
        if why is not None:
            problems.append(f"{o.path}: {why}")
        if part.get("copy") != copy_ or chain[0].get("replica") != copy_:  # moved since: its first copy is the record's or the cited plan's
            problems.append(f"{o.path}: its store copy ({part.get('copy')}, replica {chain[0].get('replica')}) is not "
                            f"partition {p}'s copy in {'the record' if binding is None else 'the cited binding'} ({copy_})")
        checks = [("store copy" if i == 0 else f"store copy after its identity's store change {i}", block)
                  for i, block in enumerate(chain)]
        if "parents_store" in (o.identity or {}):
            checks.append(("parents' store copy", o.identity.get("parents_store") or {}))
        for what, block in checks:
            if identity_note is not None:  # a plain partition union: the note, exactly, never a pattern or a QLever check
                if block.get("identity_check") != identity_note:
                    problems.append(f"{o.path}: its {what} does not carry the identity view's store note exactly (a plain "
                                    f"partition union's per-copy identity is its manifest identity and the pinned store "
                                    f"description)")
                continue
            if not isinstance(block.get("identity_check"), str) or not IDENTITY_CHECK.fullmatch(block["identity_check"]):
                problems.append(f"{o.path}: its {what} carries no passed identity check as the partition builders record "
                                f"it (?cmd=stats equal to ... (store check sha256 ...))")
    if problems:
        raise Refused(" | ".join(problems))
    named = Counter(index_of.values())
    if sorted(named) != list(want) or any(n > 1 for n in named.values()):
        raise Refused(f"the union holds partitions {sorted(index_of.values())}, not each of "
                      + (f"0..{k - 1}" if want == tuple(range(k)) else f"the named {list(want)}") + " once")
    sources = set()
    for o in outs:
        src = [((o.identity or {}).get("derived_from") or {}).get("path"), (o.manifest.get("entry184") or {}).get("source_out")]
        sources |= {s for s in src if isinstance(s, str) and s}
    named_sources = [(str(o.path), sorted(o.names() & sources)) for o in outs if o.names() & sources]
    if named_sources:
        raise Refused(f"OUTs another partition names as its source (GLM's natural OUT: its parents count once, through "
                      f"the partitions' copies): {named_sources}")
    run_partition: dict[str, int] = {}
    run_out: dict[str, str] = {}
    question_out: dict[str, str] = {}
    hashes = [set(p["question_id_sha256"]) for p in parts]
    for o in outs:
        p = index_of[str(o.path)]
        batch = F.inputs.load_batch(o.path, F.gate.grid_cells(o.path))
        for r in [*batch.records, *batch.branch]:
            q, run = r["question_id"], r["run_id"]
            if partition_of(q, k, weights) != p or sha256_text(q) not in hashes[p]:
                problems.append(f"{o.path} (partition {p}): record {run}'s question is partition "
                                f"{partition_of(q, k, weights)}'s")
            if question_out.setdefault(q, str(o.path)) != str(o.path):
                problems.append(f"question {q} in two partitions ({question_out[q]}, {o.path})")
            if run_out.setdefault(run, str(o.path)) != str(o.path):
                problems.append(f"run id {run} in two partitions ({run_out[run]}, {o.path})")
            run_partition[run] = p
    if problems:
        raise Refused(f"{len(problems)} partition problems: " + " | ".join(problems[:20]))
    fields = (record, record_sha256, k, index_of, run_partition, None if want == tuple(range(k)) else want,
              None if binding is None else {"sha256": binding.sha256, "plans": [
                  {"partition": i, "out": pl.out, "copy": pl.copy, "plan_sha256": pl.sha256}
                  for i, pl in sorted(binding.plans.items())]})
    return Union(*fields) if weights is None else WeightedUnion(*fields, weights)


def union_stack_check(F: Frozen) -> Callable[[Iterable[tuple[Path, list[dict], dict | None]]], None]:
    """One stack per (graph, model) over the union, from each partition's deployment identity (stacks_by_route_model),
    in place of check_one_stack's one-route-URL and one-store-endpoint refusals (entries 150, 181, 184: partitions run
    on identical store copies, through more than one router; check_union requires each copy's passed identity check)."""

    def check(batches: Iterable[tuple[Path, list[dict], dict | None]]) -> None:
        stacks: dict[tuple[str, str], dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
        problems: list[str] = []
        for path, records, identity in batches:
            if identity is None:
                problems.append(f"{path}: no deployment-identity.json")
                continue
            named = identity.get("stacks_by_route_model") or {}
            unnamed = set()
            for r in records:
                if r["model"] in named:
                    stacks[(F.inputs.store_name(r), r["model"])][str(named[r["model"]])].add(str(path))
                else:
                    unnamed.add(r["model"])
            problems += [f"{path}: its deployment identity names no stack for {m}" for m in sorted(unnamed)]
        problems += [f"{model} in {graph}: {len(seen)} stacks ("
                     + "; ".join(f"{s} in {', '.join(sorted(p))}" for s, p in sorted(seen.items())) + ")"
                     for (graph, model), seen in sorted(stacks.items()) if len(seen) > 1]
        if problems:
            raise Refused("one stack per (model, graph) over the partition union (entries 144, 184): "
                          + " | ".join(problems))

    return check


# --- the frozen per-arm path's loading, over one OUT set or one union ---------------------------------------------------


@dataclasses.dataclass
class Loaded:
    attempts: list
    ledger: list
    questions: dict[str, dict]
    design: dict[str, dict]
    index: Any
    side: dict
    settings: dict
    messages: dict
    scorer_changes: Any
    model_settings: Any
    output_cap_hits: dict
    questions_sha256: set[str]
    graphs: set[str]


def load_inputs(F: Frozen, batches: Sequence[Path], questions: Sequence[Path], *, split: str, n_reps: int | None,
                ledgers: Sequence[Path], tier_if_missing: str | None,
                stack_check: Callable[[list[tuple]], None],
                design_keep: Callable[[str], bool] | None = None) -> Loaded:
    """pipeline.analyse_batches' loading without a registered config (the frozen exploratory path), line for line, with
    ``stack_check`` in place of check_one_stack: each batch's replay check and questions file, the frozen loaders,
    the split's attempts, the side counts, refusals, output-cap hits, scorer changes and per-model settings, the
    design completed over every batch's questions, and the ledgers merged. ``design_keep`` (a subset of a record's
    partitions): the design is only the split's questions it accepts, before the missing attempts are made."""
    P, I, G = F.pipeline, F.inputs, F.gate
    n_reps = P.design_repetitions(n_reps, None)
    attempts: list = []
    side: dict[tuple[str, str, str], Counter] = {}
    inputs, all_questions = [], {}
    settings_seen: Counter = Counter()
    messages: dict[str, dict[str, str | None]] = {}
    cap_records: list[tuple[dict, str]] = []
    checker_records: list[dict] = []
    stack_inputs: list[tuple] = []
    changes = P.ScorerChanges()
    fired = P.ModelSettings()
    shas: set[str] = set()
    graphs: set[str] = set()
    for path in batches:
        check = G.require_replay_check(path)
        qpath = G.select_questions(path, questions)
        manifest_sha = G.require_questions_match(path, qpath)
        shas.add(manifest_sha)
        batch_questions = I.load_questions(qpath)
        if tier_if_missing:
            for q in batch_questions.values():
                q.setdefault("tier", tier_if_missing)
        P.merge_questions(all_questions, batch_questions, str(qpath))
        batch = I.load_batch(path, G.grid_cells(path))
        identity_path = Path(path) / "deployment-identity.json"
        stack_inputs.append((path, [*batch.records, *batch.branch],
                             json.loads(identity_path.read_text(encoding="utf-8")) if identity_path.exists() else None))
        graphs |= {I.store_name(r) for r in [*batch.records, *batch.branch]}
        rows = [dataclasses.replace(a, questions_sha256=manifest_sha)
                for a in I.attempts_from_batch(batch, batch_questions, checked_call=F.checked_call,
                                               tier_if_missing=tier_if_missing)]
        attempts.extend(a for a in rows if a.split == split)
        in_split = {a.run_id: a for a in rows if a.split == split}
        for r in batch.records:
            if r["run_id"] in in_split:
                a = in_split[r["run_id"]]
                changes.add(r, batch_questions.get(r["question_id"]), a)
                fired.add(r, f"{a.graph}|{a.model}|{a.cohort}")
        for r in batch.branch:
            if (batch_questions.get(r.get("question_id")) or {}).get("split") == split:
                fired.add(r, P.branch_arm(r))
                cap_records.append((r, P.branch_arm(r)))
        settings_seen.update(I.refusal_setting(r) for r in batch.records if r["run_id"] in in_split)
        cap_records.extend((r, f"{in_split[r['run_id']].graph}|{r['model']}|{in_split[r['run_id']].cohort}")
                           for r in batch.records if r["run_id"] in in_split)
        messages.update({r["run_id"]: I.final_message(r) for r in batch.records
                         if r["run_id"] in in_split and in_split[r["run_id"]].status == "refused"})
        for key, counts in batch.side_counts.items():
            side.setdefault(key, Counter()).update(counts)
        checker_records.extend(r for r in (*batch.records, *batch.branch) if r.get("checker"))
        inputs.append({"batch": str(path), "records_sha256": check["records_sha256"], "n_records": check["n_records"],
                       "replay_check_code": check.get("code"), "questions_sha256": manifest_sha, "scorer_digest": None,
                       "records_outside_split": sum(1 for a in rows if a.split != split),
                       "branch_records": dict(sorted(Counter(r["load"]["arm"] for r in batch.branch).items())),
                       "branch_side": {cell: dict(c) for cell, c in sorted(batch.branch_side.items())}})
    stack_check(stack_inputs)
    design = {qid: q for qid, q in all_questions.items()
              if q.get("split") == split and (design_keep is None or design_keep(qid))}
    attempts = I.complete_design(attempts, design, n_reps)
    ledger: list = []
    for path in ledgers:
        ledger.extend(I.load_checker_ledger(path, split, single_split=False, registered=False))
    ledger = P.ledger_source_names(ledger, None, P.curated_generic_rows(ledger, checker_records))
    settings = {"gated": True, "inputs": inputs, "ledgers": [str(p) for p in ledgers], "split": split,
                "n_reps": n_reps, "tier_if_missing": tier_if_missing, "bank": None, "frozen_config": None,
                "refusal_settings": dict(sorted(settings_seen.items()))}
    cap_arms = {r["run_id"]: arm for r, arm in cap_records}
    return Loaded(attempts, ledger, all_questions, design, F.prereg.index_from_questions(design, split),
                  {k: dict(v) for k, v in side.items()}, settings, messages, changes.table(), fired.table(),
                  I.output_cap_report([r for r, _ in cap_records], lambda r: cap_arms[r["run_id"]]), shas, graphs)


def check_union_ledger(ledger: Sequence[Any], union: Union) -> None:
    """Every ledger row is an output of the union (its run id a record of one partition) on a question of that
    partition, and no (run id, source, use) twice across the ledger files."""
    problems = []
    for row in ledger:
        p = union.run_partition.get(row.run_id)
        if p is None:
            problems.append(f"ledger row {row.run_id} ({row.source}, {row.use}): no record of the union")
        elif partition_of(row.question_id, union.k, union.weights) != p:
            problems.append(f"ledger row {row.run_id}: its question is partition "
                            f"{partition_of(row.question_id, union.k, union.weights)}"
                            f"'s, its record partition {p}'s")
    dup = [k for k, n in Counter((r.run_id, r.source, r.use) for r in ledger).items() if n > 1]
    problems += [f"ledger row {d} in two ledger files" for d in dup[:5]]
    if problems:
        raise Refused(f"{len(problems)} ledger problems: " + " | ".join(problems[:20]))


def check_bases_whole(questions: Mapping[str, dict], split: str, union: Union) -> None:
    """The index counts a base question once, so a subset must not split one: refused when a base question of the split
    has questions both inside and outside the subset (entry 179's rule partitions by question id, not by base)."""
    inside: set[str] = set()
    outside: set[str] = set()
    for qid, q in questions.items():
        if q.get("split") == split:
            member = partition_of(qid, union.k, union.weights) in union.subset
            (inside if member else outside).add(q.get("base_question_id") or qid)
    if inside & outside:
        raise Refused(f"{len(inside & outside)} base questions have questions both inside and outside the subset "
                      f"{list(union.subset)}: a subset must not split a base question")


SUBSET_NOTE = ("A subset of the record's partitions (--partitions): the record, the questions file and every OUT are "
               "checked in full, but the design, the question index and the resample matrix cover only the named "
               "partitions' questions. A cluster with questions in a partition outside the subset is resampled as the "
               "cluster of its subset questions alone (the cluster map is not changed): clusters_cut counts them. "
               "These intervals are not the full union's.")


def subset_block(union: Union, loaded: Loaded, cluster_map: Any, split: str) -> dict[str, Any]:
    """What a --partitions call records in ``partition_union.subset``: the partitions named, the questions of the record
    and of the subset, and how many clusters of the map have questions both inside and outside the subset (cut by it;
    a question outside the subset that the map does not hold is counted apart, as the map restricts to the index)."""
    inside: set[str] = set()
    outside: set[str] = set()
    unmapped = 0
    for qid, q in loaded.questions.items():
        if q.get("split") != split:
            continue
        cid = cluster_map.clusters.get(q.get("base_question_id") or qid)
        if partition_of(qid, union.k, union.weights) in union.subset:
            if cid is not None:
                inside.add(cid)
        elif cid is None:
            unmapped += 1
        else:
            outside.add(cid)
    return {"partitions": list(union.subset), "of_k": union.k, "record_questions": union.record["n_questions"],
            "subset_questions": len(loaded.design), "clusters_cut": len(inside & outside),
            "outside_questions_not_in_map": unmapped, "note": SUBSET_NOTE}


# --- the per-arm numbers on a given resample matrix ---------------------------------------------------------------------


def per_arm(F: Frozen, loaded: Loaded, K: np.ndarray, *, resampling: str, status: str,
            strata: bool = True) -> dict[str, Any]:
    """pipeline.analyse's per-arm result without a registered config (prereg None, no bank index), on ``K``: the
    frozen arms, ledger on real outputs and refusals, each interval from ``K``."""
    P = F.pipeline
    named = P.ledger_source_names(loaded.ledger, None)
    arms = F.metrics.build_arms(loaded.attempts, named, loaded.questions)
    index = loaded.index
    result: dict[str, Any] = {
        "schema": P.RESULTS_SCHEMA,
        "settings": {"seed": SEED, "n_boot": N_BOOT, "level": LEVEL, "n_questions": len(index),
                     "strata": dict(sorted(Counter(index.strata).items())), "resampling": resampling,
                     "weighting": WEIGHTING, "interval": INTERVAL, "exact_bounds": EXACT_BOUNDS, "status": status,
                     **loaded.settings},
        "arms": {key.label: P._arm_result(key, units, index, K, LEVEL,
                                          loaded.side.get((key.graph, key.model, key.cohort), {}), strata)
                 for key, units in sorted(arms.items())},
        "real_output_ledger": P.real_output_ledger(arms, index, K, LEVEL, None, K),
        "refusals": P.refusal_report(loaded.attempts, None, loaded.messages),
        "scorer_changes": loaded.scorer_changes,
        "model_settings": loaded.model_settings,
        "confirmatory": None,
        "output_cap_hits": loaded.output_cap_hits,
    }
    result["settings"]["code_digest"] = P.code_digest()
    return result


# --- online verdicts against their offline replay twins (design ruling (B), 25 Oct 2026) --------------------------------

ONLINE_USES = ("checker", "both")
REPLAY_USE = "replay"
ONLINE_VS_REPLAY_RULE = (
    "For each online ledger row (use checker or both: a source with an online branch cell) of a run, the row with use "
    "replay of the same graph, model, cohort, run and source is its twin; the pair agrees when the two rows' status letters "
    "(F, P, U) are equal and disagrees when they differ, and an online row with no twin is not comparable; counts only, per "
    "online arm (graph|model|cohort|use|source), and the section is empty when the ledger holds no replay row.")


def online_vs_replay(ledger: Sequence[Any]) -> dict[str, Any]:
    """The online-against-replay disagreement counts per online arm: how many online rows agree with their replay twin,
    disagree with it, or have none. Counts only (no status, no run id, no value); ``ledger`` is the source-named ledger."""
    twins = {(r.graph, r.model, r.cohort, r.run_id, r.source): r.status for r in ledger if r.use == REPLAY_USE}
    arms: dict[str, Counter] = {}
    if twins:
        for r in ledger:
            if r.use not in ONLINE_USES:
                continue
            counts = arms.setdefault(f"{r.graph}|{r.model}|{r.cohort}|{r.use}|{r.source}", Counter())
            twin = twins.get((r.graph, r.model, r.cohort, r.run_id, r.source))
            counts["not_comparable" if twin is None else "agree" if twin == r.status else "disagree"] += 1
    return {"rule": ONLINE_VS_REPLAY_RULE,
            "arms": {k: {n: c.get(n, 0) for n in ("agree", "disagree", "not_comparable")} for k, c in sorted(arms.items())}}


def check_replay_coverage(loaded: "Loaded") -> None:
    """Design ruling (B): when the ledger holds replay rows at all, every cohort (graph, model, cohort) that has a scored output
    has a replay arm for every replay source in the ledger. ``build_arms`` refuses a replay arm that covers only some of its
    cohort's scored outputs, but it never creates (and so never checks) an arm for a cohort with no replay row: a cohort whose
    matrices were left out, or whose runs all went missing from the replay, would be analysed without them. Silent without a
    replay row (a call without matrices), and the fault bank's rows (cohort bank) are no attempt cohort."""
    replay = [r for r in loaded.ledger if r.use == REPLAY_USE and r.cohort != "bank"]
    if not replay:
        return
    sources = {r.source for r in replay}
    have = {(r.graph, r.model, r.cohort, r.source) for r in replay}
    cohorts = {(a.graph, a.model, a.cohort) for a in loaded.attempts if a.output}
    missing = sorted(f"{g}|{m}|{c}|{REPLAY_USE}|{src}" for (g, m, c) in cohorts for src in sources if (g, m, c, src) not in have)
    if missing:
        raise Refused(f"{len(missing)} replay arms are missing for cohorts that have scored outputs while the ledger holds replay "
                      f"rows for other cohorts or sources (a cohort left out of the replay matrices?): {missing[:10]}")


# --- deployment shares and the conditional sensitivity ---------------------------------------------------------------


def load_attribution() -> ModuleType:
    """Entry 189's attribution (attribution.py beside this script, checked by its pinned sha256)."""
    path = HERE / "attribution.py"
    if sha256_file(path) != ATTRIBUTION_SHA256:
        raise Refused(f"{path}: sha256 {sha256_file(path)[:12]}, not the attribution check's {ATTRIBUTION_SHA256[:12]}")
    spec = importlib.util.spec_from_file_location("e189_attribution", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["e189_attribution"] = module
    spec.loader.exec_module(module)
    return module


@dataclasses.dataclass
class Deployments:
    """Each analysed record's serving deployment (attribution.attempt's served_by, by run id), the branch records by
    parent and cell, each checker source's branch cells, and how the own turns were attributed."""

    served: dict[str, str]
    retries: dict[str, dict[str, list[str]]]
    cells_of_source: dict[str, set[str]]
    how: Counter
    fingerprint_mismatch: int
    plain: set[str] = dataclasses.field(default_factory=set)  # run ids of plain runs' records (no branch cells)
    started: dict[str, Any] = dataclasses.field(default_factory=dict)  # run id -> started_at
    out_of: dict[str, str] = dataclasses.field(default_factory=dict)  # run id -> its OUT
    changes: list = dataclasses.field(default_factory=list)  # check_fingerprints: every admitted non-baseline fingerprint
    admissions: dict = dataclasses.field(default_factory=dict)  # the config read (path, sha256, entries)


def branch_sources(F: Frozen, out: Path) -> dict[str, str]:
    """A grid OUT's branch cells -> the checker arm's source name in the analysis: the cell's pinned arm name (a feedback
    cell) or generic_retry/<name> (a generic cell), as harness/checkloop/ledger.py names its rows, then
    pipeline.LEDGER_SOURCES; one source per cell over every role (empty for a plain run's OUT). A cell of another mode
    (a baseline's, design log) has no checker arm in the ledger (ledger.from_online_record exports feedback and generic
    cells only), so it is left out."""
    manifest = read_json(Path(out) / "manifest.json")
    arms = ((((manifest.get("fingerprint") or {}).get("phases") or {}).get("branch") or {}).get("arms")) or {}
    out_map = {}
    for cell, pins in arms.items():
        names, modes = set(), set()
        for role in ((pins or {}).get("by_role") or {}).values():
            name, mode = (role.get("arm") or {}).get("name"), role.get("mode")
            modes.add(mode)
            if mode not in ("feedback", "generic"):
                continue
            if not isinstance(name, str) or not name:
                raise Refused(f"{out}: branch cell {cell!r} pins a {mode} arm without its name")
            raw = name if mode == "feedback" else f"{GENERIC_PREFIX}{name}"
            names.add(F.pipeline.LEDGER_SOURCES.get(raw, raw))  # ledger_source_names maps these and keeps the rest
        if not modes & {"feedback", "generic"}:
            continue  # no checker arm
        if len(names) != 1 or len(modes) != 1:
            raise Refused(f"{out}: branch cell {cell!r} names {len(names)} sources ({len(modes)} modes) over its roles")
        out_map[cell] = names.pop()
    return out_map


def read_admissions(path: Path) -> tuple[list[dict], str]:
    """The reviewed config of admitted fingerprints (c): each entry names its model and stack, the fingerprint, the
    evidence (vLLM version, checkpoint revision, the entry-170 verdict's sha256, the admission time) and the ruling or
    change-log entry admitting it; refused otherwise."""
    raw = Path(path).read_bytes()
    doc = json.loads(raw)
    entries = doc.get("admitted") if isinstance(doc, dict) and doc.get("schema") == ADMISSIONS_SCHEMA else None
    if not isinstance(entries, list):
        raise Refused(f"{path}: not a {ADMISSIONS_SCHEMA} file with an admitted list")
    seen = set()
    for i, e in enumerate(entries):
        if not isinstance(e, dict) or not all(isinstance(e.get(k), str) and e[k] for k in ADMISSION_FIELDS):
            raise Refused(f"{path}: entry {i} lacks one of {ADMISSION_FIELDS}")
        if not any(isinstance(e.get(k), str) and e[k].strip() for k in ADMISSION_CITATIONS):
            raise Refused(f"{path}: entry {i} cites no ruling or change-log entry ({' or '.join(ADMISSION_CITATIONS)})")
        if not re.fullmatch(r"[0-9a-f]{64}", e["entry170_verdict_sha256"]):
            raise Refused(f"{path}: entry {i}: entry170_verdict_sha256 is not a sha256")
        try:
            datetime.fromisoformat(e["admitted_at"])
        except ValueError:
            raise Refused(f"{path}: entry {i}: admitted_at is not a time") from None
        key = (e["route_model"], e["stack"], e["system_fingerprint"])
        if key in seen:
            raise Refused(f"{path}: entry {i} admits {key} twice")
        seen.add(key)
    return entries, sha256_bytes(raw)


def check_fingerprints(at: ModuleType, code: Any, out: Path, records: Sequence[Any], admissions: Sequence[dict]
                       ) -> list[dict]:
    """One OUT's served fingerprints, per route model (the own turns of every record: attempts and set-aside tries): a
    mixed fingerprint (one reply from several servers) is refused whatever is admitted, as the runner counts it a
    change (ServedModels.check); each other fingerprint than the baseline must be admitted by (b) the OUT's admitted-fingerprints.json or (c) the config, with the baseline's vLLM
    version and, where the stack identity records one, its checkpoint revision; refused otherwise. Returns each admitted
    change with its sources and its first turn (the earliest record started serving it)."""
    out = Path(out)
    info = at.OutInfo.read(out, code.admit.MIXED)
    runner_of = {route: name for name, route in info.route_models.items()}
    ident_path = out / "deployment-identity.json"
    identity = read_json(ident_path) if ident_path.is_file() else {}
    admitted_path = out / code.admit.ADMITTED_FILE
    runner_doc = None
    if admitted_path.is_file():
        doc = read_json(admitted_path)
        runner_doc = None if code.admit.malformed(doc) else doc  # the runner ignores a malformed file whole
    served: dict[str, dict[str, tuple[Any, Any]]] = defaultdict(dict)  # model -> fingerprint -> first (started, run)
    for rec in records:
        for t in at.own_turns(rec):
            for fp in ((t.fingerprint,) if t.fingerprint else ()):  # whole: a mixed one is never split
                first = served[rec.model].get(fp)
                if first is None or (str(rec.started_at), str(rec.run_id)) < (str(first[0]), str(first[1])):
                    served[rec.model][fp] = (rec.started_at, rec.run_id, t.turn)
    changes, problems = [], []
    for model, fps in sorted(served.items()):
        name = runner_of.get(model)
        sitea = identity.get("sitea") if isinstance(identity.get("sitea"), dict) else {}
        own = sitea if name is not None and sitea.get("model") == name else {}
        baseline = own.get("system_fingerprint") or (info.first_turns.get(model) or {}).get("system_fingerprint")
        stack = (identity.get("stacks_by_route_model") or {}).get(model)
        stack_id = own.get("identity") if isinstance(own.get("identity"), dict) else {}
        checkpoint = next((stack_id[k] for k in CHECKPOINT_KEYS if isinstance(stack_id.get(k), str) and stack_id[k]),
                          None)
        parsed = code.admit.FINGERPRINT_RE.match(baseline) if isinstance(baseline, str) else None
        version = parsed.group("version") if parsed else None
        runner = ((runner_doc or {}).get("models") or {}).get(name) or {}
        for fp, (started, run, turn) in sorted(fps.items()):
            where = f"{out} {model} fingerprint {fp}"
            if code.admit.MIXED in fp:  # before the baseline: even a mixed baseline is no baseline
                problems.append(f"{where}: a mixed fingerprint (one reply from several servers), a change whatever is "
                                f"admitted")
                continue
            if fp == baseline:
                continue
            if baseline is None:
                problems.append(f"{where}: no baseline fingerprint (neither the stack identity nor served-models.json "
                                f"records one)")
                continue
            m = code.admit.FINGERPRINT_RE.match(fp)
            if version is None or m is None or m.group("version") != version:
                problems.append(f"{where}: its vLLM version is not the baseline's ({baseline}), admitted or not")
                continue
            sources = []
            for e in runner.get("admitted") or []:  # (b) admit.py's entry-167/170 gate
                if e.get("system_fingerprint") != fp:
                    continue
                if checkpoint is not None and runner.get("revision") != checkpoint:
                    problems.append(f"{where}: admitted by the OUT for checkpoint {runner.get('revision')}, the "
                                    f"stack's is {checkpoint}")
                    continue
                ev = e.get("evidence") or {}
                sources.append({"source": "the OUT's admitted-fingerprints.json (admit.py)",
                                "admission_id": ev.get("admission_id"), "backend": e.get("backend"),
                                "admitted_at": e.get("admitted_at"),
                                "behaviour_verdict_sha256": (ev.get("behaviour_verdict") or {}).get("file_sha256")})
            for e in admissions:  # (c) this script's reviewed config
                if (e["route_model"], e["stack"], e["system_fingerprint"]) != (model, stack, fp):
                    continue
                if e["vllm_version"] != version or (checkpoint is not None and e["checkpoint"] != checkpoint):
                    problems.append(f"{where}: listed in {ADMISSIONS} with vLLM {e['vllm_version']} and checkpoint "
                                    f"{e['checkpoint']}, the stack's are {version} and {checkpoint}")
                    continue
                sources.append({"source": f"results/{HERE.name}/{ADMISSIONS}",
                                **{k: e.get(k) for k in (*ADMISSION_FIELDS, *ADMISSION_CITATIONS) if k in e}})
            if not sources:
                problems.append(f"{where}: admitted neither by the OUT's admitted-fingerprints.json nor by {ADMISSIONS}")
                continue
            changes.append({"out": str(out), "route_model": model, "stack": stack, "baseline": baseline,
                            "system_fingerprint": fp, "sources": sources,
                            "first_turn": {"run_id": run, "started_at": started, "turn": turn}})
    if problems:
        raise Refused(f"{len(problems)} served fingerprints not admitted: " + " | ".join(problems[:10]))
    return changes


def attribute(F: Frozen, batches: Sequence[Path], logs: Sequence[Path],
              admissions_path: Path | None = None) -> Deployments:
    """Every record of the batches by deployment (attribution.py: a fronted turn by the front's provenance log, a turn
    of a model with admitted endpoints by its route URL, any other by its route URL and served fingerprint; an attempt
    is a deployment only when all its own turns were served by it), as balance.attribute_batches does it, over the
    union of the batches."""
    at = load_attribution()
    try:
        code = at.CodeRoot.load(F.root)
        log = at.read_log(code, logs)
        read = {}
        for out in batches:
            every, unreadable = at.read_out(Path(out), sealed=True)
            if unreadable:
                raise Refused(f"{out}: unreadable record files {unreadable[:5]}")
            read[out] = every
        pre = at.pre_front_ids(code, [r for every in read.values() for r in every])
        served: dict[str, str] = {}
        retries: dict[str, dict[str, list[str]]] = {}
        how: Counter = Counter()
        mismatch = 0
        cells_of_source: dict[str, set[str]] = defaultdict(set)
        source_of_cell: dict[str, str] = {}
        plain: set[str] = set()
        started: dict[str, Any] = {}
        out_of: dict[str, str] = {}
        admissions_path = Path(admissions_path) if admissions_path is not None else HERE / ADMISSIONS
        admissions, admissions_sha = read_admissions(admissions_path)
        changes: list[dict] = []
        for out in batches:
            is_plain = F.gate.grid_cells(Path(out)) is None
            changes += check_fingerprints(at, code, Path(out), read[out], admissions)  # set-aside tries too
            for cell, source in branch_sources(F, out).items():
                if source_of_cell.setdefault(cell, source) != source:
                    raise Refused(f"branch cell {cell!r} names two sources across the batches")
                cells_of_source[source].add(cell)
            attributor = at.Attributor(code, at.OutInfo.read(Path(out), code.admit.MIXED), log, pre)
            for rec in read[out]:
                a = at.attempt(rec, attributor)
                how.update(x.how for x in a.own)  # set-aside tries' fronted turns must join too (join.py's rule)
                if rec.directory != at.ATTEMPT_DIR:
                    continue
                if rec.run_id in served:
                    raise Refused(f"run id {rec.run_id} is in two records")
                served[rec.run_id] = a.served_by
                started[rec.run_id], out_of[rec.run_id] = rec.started_at, str(out)
                if is_plain:
                    plain.add(rec.run_id)
                mismatch += sum(x.fingerprint_mismatch for x in a.own)
                if rec.branched_from:
                    retries.setdefault(rec.branched_from, {}).setdefault(rec.cell, []).append(rec.run_id)
    except at.AttributionError as exc:
        raise Refused(f"deployment attribution: {exc}") from None
    bad = {k: how[k] for k in JOIN_MUST_BE_ZERO if how[k]}
    if bad:
        raise Refused(f"fronted turns that do not join to one backend {bad}: give the front's provenance log as it is now "
                      f"(--front-log)")
    return Deployments(served, retries, dict(cells_of_source), how, mismatch, plain, started, out_of, changes,
                       {"path": str(admissions_path), "sha256": admissions_sha, "entries": len(admissions)})


def unit_served(unit: Any, key: Any, dep: Deployments, prompt: str) -> tuple[str | None, list[str] | None]:
    """A unit's attempt's row (None: no record, an attempt without one) and, in a grid's checker arm, its retries' rows,
    one per retry attempt with a record, in the arm's cells. The second is None when no retry is considered (a prompt
    arm, a unit not retried, a plain run's checker record, which holds its own retry as its later turns, or an attempt
    without a record) and an empty list when the output was retried and no retry has a record."""
    parent = dep.served.get(unit.attempt.run_id)
    if parent is None or key.use == prompt or unit.checked is None or not unit.checked.retried \
            or unit.attempt.run_id in dep.plain:
        return parent, None
    cells = dep.cells_of_source.get(key.source, set())
    return parent, [dep.served[r] for c in sorted(cells) for r in dep.retries.get(unit.attempt.run_id, {}).get(c, [])]


def unit_row(at: ModuleType, unit: Any, key: Any, dep: Deployments, prompt: str) -> str | None:
    """A unit's deployment row: its attempt's (None: no record, an attempt without one); in a grid's checker arm, a
    retried output's row is the deployment that served the output and its retries in the arm's cells, mixed when they
    differ, RETRY_WITHOUT_RECORD when no retry has a record. A plain run's checker record holds its own retry (its later
    turns), so its row already covers it."""
    parent, retried = unit_served(unit, key, dep, prompt)
    if retried is None:
        return parent
    if not retried:
        return RETRY_WITHOUT_RECORD
    rows = {parent, *retried}
    return rows.pop() if len(rows) == 1 else at.MIXED


# The store section's variant of a unit's row (science, 25 Oct ~06:2xZ, addendum 1 to the branch-phase ruling). A branch
# row's live retry is attributed to the chain's last copy, so a retried output whose parent attempt ran on an earlier
# copy has two copies: 'mixed' is the truth, and the row names both.
ARROW = " → "  # U+2192 with one space each side: "<parent copy> → <retry copy>"
SAME_CLASS = "mixed copies, one class"  # every copy of a unit has one and the same non-null class
MIXED_CLASS = "MIXED CLASS"  # the copies' classes differ, or one of them has none: unknown is not the same class
# A copy no entry of the copy classes describes (an endpoint label that neither the endpoint copies nor an identity
# names a replica for, or a replica the table does not cover): a single-copy unit on it carries this explicit label,
# which is not a class mix. A
# unit on SEVERAL copies, one of which has no entry, is a refusal (science, 25 Oct ~06:5xZ): a real two-copy unit comes
# from a store change, which needs store-proofs entries, and copy-classes.json covers those, so it is a build defect.
NO_COPY_CLASS = "no copy class (no entry in the copy classes)"
STORE_UNIT_RULE = ("a checker arm's retried output is a unit row naming its copies: its copy alone when the output and "
                   f"every retry ran on it, else '<parent copy>{ARROW}<retry copies>' (the retries' copies sorted, the "
                   "parent's own left out); its class label is its copy's class label, or, for several copies, "
                   f"'{SAME_CLASS}' when each has one and the same class, '{MIXED_CLASS}' otherwise (a copy whose table "
                   "class is null counts as a different class: only that is a class mix; a copy with no table entry at "
                   "all refuses a unit on several copies, and a unit on that one copy carries the label "
                   f"'{NO_COPY_CLASS}', no class mix); attempt_shares count each output's attempt "
                   "for its copy and each retry attempt for its own (science, 25 Oct ~06:2xZ)")


@dataclasses.dataclass(frozen=True)
class StoreUnit:
    """A unit as the store section counts it: its row (None: no record), its class label and the copy of each of its
    attempts: the output's attempt and one per retry attempt with a record (a unit without a record is a single None;
    an output whose retry has no record counts its own attempt only, while its row and class label are
    RETRY_WITHOUT_RECORD, as in the deployment section)."""

    row: str | None
    class_label: str | None
    attempts: tuple[str | None, ...]


def store_unit_row(unit: Any, key: Any, dep: Deployments, prompt: str,
                   copy_class: Callable[[str], tuple[str | None, str]]) -> StoreUnit:
    """A unit's store row, class label and attempts (StoreUnit). ``dep.served`` is the copy of each attempt (the store
    rows' labels), ``copy_class`` a copy label's (its one class or None, its class label). The same cells and retries
    as unit_row's (unit_served), but a retried output whose attempts ran on several copies is not 'mixed': its row names
    them, parent first, then the retries' copies (those that are not the parent's) sorted and joined by ", "."""
    parent, retried = unit_served(unit, key, dep, prompt)
    if parent is None:
        return StoreUnit(None, None, (None,))
    if retried is None:
        return StoreUnit(parent, copy_class(parent)[1], (parent,))
    if not retried:
        return StoreUnit(RETRY_WITHOUT_RECORD, RETRY_WITHOUT_RECORD, (parent,))
    attempts = (parent, *retried)
    others = sorted(set(retried) - {parent})
    if not others:
        return StoreUnit(parent, copy_class(parent)[1], attempts)
    for copy in (parent, *others):  # a real two-copy unit comes from a store change, whose copies the table covers
        if copy_class(copy)[1] == NO_COPY_CLASS:
            raise Refused(f"the copy {copy} of a retried output on several copies has no entry in the copy classes")
    one = {copy_class(c)[0] for c in (parent, *others)}
    return StoreUnit(f"{parent}{ARROW}{', '.join(others)}", MIXED_CLASS if None in one or len(one) > 1 else SAME_CLASS,
                     attempts)


def share_of(rows: Sequence[str | None]) -> dict[str, Any]:
    """An arm's attempts with a record, those without, and each deployment row's exact count and share."""
    recorded = [r for r in rows if r is not None]
    n = len(recorded)
    return {"attempts": n, "without record": len(rows) - n,
            "rows": {d: {"n": k, "of": n, "share": k / n} for d, k in sorted(Counter(recorded).items())}}


def compare_shares(s1: Mapping[str, Any], s2: Mapping[str, Any], not_deployments: Iterable[str]) -> dict[str, Any]:
    """Two arms' shares, row by row as exact fractions: triggered when any row differs by strictly more than TRIGGER; the
    shared deployments are the real deployments (not mixed, unattributed, no own turns or a retry without record) that
    served attempts of both arms. An arm without recorded attempts cannot be compared."""
    if not s1["attempts"] or not s2["attempts"]:
        return {"comparable": False, "triggered": False}
    diffs = {}
    for row in sorted(set(s1["rows"]) | set(s2["rows"])):
        a = Fraction((s1["rows"].get(row) or {"n": 0})["n"], s1["attempts"])
        b = Fraction((s2["rows"].get(row) or {"n": 0})["n"], s2["attempts"])
        if a != b:
            diffs[row] = {"first": str(a), "second": str(b), "difference": str(a - b)}
    triggered = any(abs(Fraction(d["difference"])) > TRIGGER for d in diffs.values())
    skip = set(not_deployments)
    shared = sorted(d for d in set(s1["rows"]) & set(s2["rows"]) if d not in skip)
    return {"comparable": True, "triggered": triggered, "differences": diffs, "shared_deployments": shared}


def attempt_arms(F: Frozen, loaded: Loaded) -> dict[Any, list]:
    """The frozen arms and their units, without the fault bank's (a bank item is no attempt: nothing served it)."""
    P, M = F.pipeline, F.metrics
    return {key: units for key, units in M.build_arms(loaded.attempts, P.ledger_source_names(loaded.ledger, None),
                                                      loaded.questions).items() if key.cohort != F.inputs.BANK}


def deployment_section(F: Frozen, loaded: Loaded, K: np.ndarray, dep: Deployments) -> dict[str, Any]:
    """Per arm, its deployment shares; per model, every pair of its arms compared (compare_shares); for each triggered
    pair, both arms' per-arm results restricted to the attempts served wholly by each shared deployment (units removed,
    never made missing), on the call's question index and resample matrix, beside the main result."""
    at = sys.modules["e189_attribution"]
    P, M = F.pipeline, F.metrics
    arms = attempt_arms(F, loaded)
    rows ={key: [unit_row(at, u, key, dep, M.PROMPT) for u in units] for key, units in arms.items()}
    shares = {key.label: share_of(r) for key, r in sorted(rows.items())}
    not_deployments = (*at.NOT_DEPLOYMENTS, RETRY_WITHOUT_RECORD)
    by_model: dict[tuple[str, str], list[Any]] = defaultdict(list)
    for key in sorted(arms):
        by_model[(key.graph, key.model)].append(key)
    pairs, wanted = [], defaultdict(set)
    for keys in by_model.values():
        for i, k1 in enumerate(keys):
            for k2 in keys[i + 1:]:
                c = compare_shares(shares[k1.label], shares[k2.label], not_deployments)
                if c["triggered"]:
                    pairs.append({"arms": [k1.label, k2.label], **c})
                    for d in c["shared_deployments"]:
                        wanted[d] |= {k1, k2}
    restricted: dict[str, dict[str, Any]] = {}
    for d in sorted(wanted):
        restricted[d] = {}
        for key in sorted(wanted[d]):
            units = [u for u, r in zip(arms[key], rows[key], strict=True) if r == d]
            arm = P._arm_result(key, units, loaded.index, K, LEVEL, {}, True)
            for k in P.SIDE_COUNTS:  # a whole arm's infrastructure counts, never attributable to one deployment
                arm["counts"].pop(k, None)
            restricted[d][key.label] = arm
    compared = sum(len(k) * (len(k) - 1) // 2 for k in by_model.values())
    changes = []
    (graph,) = loaded.graphs  # one graph per call (prepare)
    for c in dep.changes:  # per OUT: the change point, and each of the model's arms before and after it, in that OUT
        cut = str(c["first_turn"]["started_at"])
        split = {}
        for key in by_model.get((graph, c["route_model"]), []):
            mine = [(u, r) for u, r in zip(arms[key], rows[key], strict=True)
                    if dep.out_of.get(u.attempt.run_id) == c["out"]]
            split[key.label] = {
                "before": share_of([r for u, r in mine if str(dep.started.get(u.attempt.run_id)) < cut]),
                "after": share_of([r for u, r in mine if str(dep.started.get(u.attempt.run_id)) >= cut])}
        changes.append({**c, "shares_by_arm": split})
    return {
        "rule": "science, design log, ~03:35Z (change-log entry 189 item 4 extended to every exploratory contrast): each "
                "arm's share of attempts per serving deployment; two arms of a model whose shares of any row differ by "
                "strictly more than 10 points are also reported restricted to each deployment that served both "
                "(attempts served wholly by it; a checker arm's retried output with its retries)",
        "attribution": {"module": f"results/{HERE.name}/attribution.py (feat/e189-balance cc643509)",
                        "sha256": ATTRIBUTION_SHA256, "own_turns_by_how": dict(sorted(dep.how.items())),
                        "fingerprint_mismatches": dep.fingerprint_mismatch},
        "fingerprint_changes": changes, "admissions": dep.admissions,
        "shares": shares, "pairs_compared": compared, "triggered_pairs": pairs,
        "restricted": restricted, "restricted_counts_note": "infra_tries and failed_harness are whole-arm counts, left "
                                                            "out of restricted results"}


# --- store copies: listed, identity-proven endpoints and each copy's share of attempts --------------------------------


@dataclasses.dataclass(frozen=True)
class StoreRow:
    """A record's store fields (STORE_FIELDS) and its file's name (<dir>/<file>)."""

    name: str
    run_id: Any
    model: Any
    store: Any  # store.name: the record's graph
    endpoint: Any  # store.endpoint: where the runner sent the attempt's store queries
    started: Any = None  # started_at: the runner's wall-clock instant the attempt began (a branch record's: its own)
    ended: Any = None  # ended_at: the instant it ended; both read as written, never a file's mtime

    @property
    def directory(self) -> str:
        return self.name.split("/", 1)[0]


def store_row(data: Any, name: str) -> StoreRow:
    """A record's STORE_FIELDS, by ``get`` only: never an outcome, score, answer or transcript field."""

    def get(d: Any, key: str) -> Any:
        return d.get(key) if isinstance(d, dict) else None

    store = get(data, "store")
    return StoreRow(name, get(data, "run_id"), get(data, "model"), get(store, "name"), get(store, "endpoint"),
                    get(data, "started_at"), get(data, "ended_at"))


def read_store_rows(at: ModuleType, out: Path) -> list[StoreRow]:
    """Every record of the OUT, its attempts and its set-aside tries (attribution.record_paths: records/ and the set-aside
    directories, no sidecar), by its store fields alone."""
    rows, unreadable = [], []
    for path in at.record_paths(Path(out)):
        name = f"{path.parent.name}/{path.name}"
        try:
            raw = path.read_bytes()
            data = json.loads(gzip.decompress(raw) if path.name.endswith(".gz") else raw)
        except at.DAMAGED:
            unreadable.append(name)
            continue
        rows.append(store_row(data, name))
    if unreadable:
        raise Refused(f"{out}: unreadable record files {sorted(unreadable)[:5]}")
    return rows


def endpoint_label(endpoint: Any) -> str:
    """An endpoint as the results name it: by sha256 only (the replication package carries no hosts)."""
    return f"endpoint sha256:{sha256_text(str(endpoint))[:12]}" if endpoint else NO_ENDPOINT


def parse_utc(value: Any) -> datetime | None:
    """An ISO 8601 time with its zone (a recorded check's or switch's instant), else None."""
    if not isinstance(value, str):
        return None
    try:
        t = datetime.fromisoformat(value)
    except ValueError:
        return None
    return t if t.tzinfo is not None else None


def _exact(value: Any, n: int) -> bool:
    return type(value) is int and value == n


@dataclasses.dataclass(frozen=True)
class StoreChange:
    """One manifest store_changes entry (run_pilot.check_manifest's form): the endpoints it moved from and to, its time,
    and the run ids of the records made before it."""

    index: int  # 1-based, in the manifest's order
    at: Any
    from_endpoint: str
    to_endpoint: str
    before: frozenset[str]


def _side_endpoint(side: Any, *, plain_union: bool = False) -> tuple[str | None, str | None]:
    """A store change side's endpoint (its ``endpoint`` and its store description's, which must name one), or why not.
    ``plain_union`` (a plain partition union's call only; science's rule for a gated side): a side with a non-empty
    ``store_client`` object is taken when its ``endpoint``, its ``store.endpoint`` and its ``store_client.endpoint`` are all
    present and name one endpoint; manifest_store_changes then requires the two sides' store_client to be equal apart from
    their endpoint. Without it a gated side is refused as it always was."""
    if not isinstance(side, dict):
        return None, "no store"
    if side.get("store_client"):
        if not plain_union:
            return None, "a gated store (store_client), which this allowance does not cover"
        client = side["store_client"]
        if not isinstance(client, dict):
            return None, "a gated store whose store_client is not an object"
        store = side.get("store") if isinstance(side.get("store"), dict) else {}
        named = [side.get("endpoint"), store.get("endpoint"), client.get("endpoint")]
        if not all(isinstance(e, str) and e for e in named) or len(set(named)) != 1:
            return None, ("a gated store without one endpoint (endpoint, store.endpoint and store_client.endpoint must all "
                          "be present and name one)")
        return named[0], None
    store = side.get("store") if isinstance(side.get("store"), dict) else {}
    found = {e for e in (side.get("endpoint"), store.get("endpoint")) if isinstance(e, str) and e}
    if len(found) != 1:
        return None, "no endpoint, or more than one (endpoint and store.endpoint must name one)"
    return found.pop(), None


def _gate_regime(side: Any) -> str:
    """A gated side's store_client without its endpoint, in a canonical form that tells 1, 1.0 and true apart."""
    return json.dumps({k: v for k, v in side["store_client"].items() if k != "endpoint"}, sort_keys=True)


def manifest_store_changes(o: Out, *, plain_union: bool = False) -> tuple[list[StoreChange], list[str]]:
    """The OUT's manifest store changes and the problems found: each side names one endpoint; each change moves; its
    pre_change_run_ids are its pre_change_records records' run ids; each starts at the endpoint the one before it moved
    to, with that one's records among its own made before. ``plain_union`` (the call of a plain partition union alone;
    default False, the behaviour without it): a gated side is taken as _side_endpoint says, and a change's two sides must
    both be gated with the same store_client apart from their endpoint (one gate regime before and after)."""
    raw = o.manifest.get("store_changes")
    if raw is None:  # only a missing key, null or [] means no changes; {}, "", 0 or false is malformed
        return [], []
    if not isinstance(raw, list):
        return [], [f"{o.path}: its manifest's store_changes is not a list"]
    changes, problems = [], []
    for i, ch in enumerate(raw, 1):
        where = f"{o.path}: its manifest store change {i}"  # by its position: no manifest string goes into a refusal
        if not isinstance(ch, dict):
            problems.append(f"{where} is not an object")
            continue
        sides = {}
        for side in ("from", "to"):
            sides[side], why = _side_endpoint(ch.get(side), plain_union=plain_union)
            if why is not None:
                problems.append(f"{where}: its {side} side has {why}")
        if plain_union and None not in sides.values():
            gated = [bool(ch[side].get("store_client")) for side in ("from", "to")]
            if gated[0] != gated[1]:
                problems.append(f"{where}: its store_client is on one side only")
                continue
            if gated[0] and _gate_regime(ch["from"]) != _gate_regime(ch["to"]):
                problems.append(f"{where}: its sides' store_client differ beyond their endpoint")
                continue
        before = ch.get("pre_change_run_ids")
        if (not isinstance(before, list) or not all(isinstance(r, str) and r for r in before)
                or not _exact(ch.get("pre_change_records"), len(before))):
            problems.append(f"{where}: its pre_change_run_ids are not its pre_change_records records' run ids")
            continue
        if None in sides.values():
            continue
        if sides["from"] == sides["to"]:
            problems.append(f"{where} moves nothing (its from and to name one endpoint)")
            continue
        changes.append(StoreChange(i, ch.get("at"), sides["from"], sides["to"], frozenset(before)))
    for a, b in itertools.pairwise(changes):
        if a.to_endpoint != b.from_endpoint or not a.before <= b.before:
            problems.append(f"{o.path}: its manifest store change {b.index} does not start where change {a.index} "
                            f"moved to, with change {a.index}'s records among those made before it")
    return changes, problems


BRANCH_PHASE_ADDED = "phase branch"  # the string a manifest additions entry's ``added`` list holds for the branch phase


@dataclasses.dataclass(frozen=True)
class BranchPhase:
    """The instant an OUT's runner entered the branch phase: the ``at`` of the manifest additions entry whose ``added``
    list holds "phase branch", parsed (``at``) and as the manifest wrote it (``text``)."""

    at: datetime
    text: str


def manifest_branch_phase(o: Out) -> tuple[BranchPhase | None, list[str]]:
    """The OUT's branch-phase instant from its manifest's ``additions``, and the problems found. None and no problem
    only when the ``additions`` key is missing, or it is a list in which no entry adds "phase branch". Malformed (a
    problem, never a TypeError): ``additions`` present but not a list (null included), an entry that is not an object,
    an entry's ``added`` present but not a list of strings, more than one entry adding "phase branch", or its ``at`` not
    an ISO 8601 time with a zone. An object without an ``added`` key adds nothing and is ignored. A problem names the
    OUT and the part of the manifest by words; it never carries a manifest string."""
    if "additions" not in o.manifest:
        return None, []
    raw = o.manifest["additions"]
    where = f"{o.path}: its manifest's additions"
    if not isinstance(raw, list):
        return None, [f"{where} is not a list"]
    problems, found = [], []
    for entry in raw:
        if not isinstance(entry, dict):
            problems.append(f"{where} hold an entry that is not an object")
        elif "added" not in entry:
            continue
        elif not isinstance(entry["added"], list) or not all(isinstance(a, str) for a in entry["added"]):
            problems.append(f"{where} hold an entry whose added is not a list of strings")
        elif BRANCH_PHASE_ADDED in entry["added"]:
            found.append(entry)
    problems = list(dict.fromkeys(problems))
    if len(found) > 1:
        problems.append(f"{where} name the branch phase more than once")
    if problems or not found:
        return None, problems
    text, instant = found[0].get("at"), parse_utc(found[0].get("at"))
    if instant is None:
        return None, [f"{where}: its branch-phase entry's at is not an ISO time with a zone"]
    return BranchPhase(instant, text), []


STORE_CHANGE_RULE = ("science, 25 Oct 00:10Z: the records' own recorded endpoint is the ground truth; a store change's "
                     "change_at is the start of the first record on the new endpoint; records of the two copies "
                     "interleaving is a mix and refuses")
# science, 25 Oct ~04:55Z: the rule for the rows made once the OUT's runner entered the branch phase
BRANCH_PHASE_RULE = ("branch-phase rows (started at or after the manifest's phase-branch instant) carry the recorded "
                     "endpoint of their parent attempt, because harness/checkloop/online.py:542 (branch_attempt) copies "
                     "a branch record's store from its parent record; they are outside the order check, and each "
                     "one's live retry is attributed to the copy current at its start, the chain's last copy (every "
                     "change_at precedes the branch-phase instant); per OUT the report lists the parent copies' "
                     "classes and the branch copy's class (science, 25 Oct ~04:55Z)")
# science, 25 Oct (Gemma e190 K=3): where the instant a store-proofs entry records for a move differs from the first
# record on the new copy, the change point is that first record. Carried, with the manifest's recorded instant, the
# entry's and their gaps, only by a move for which the entry's differs (so every other call's results are unchanged).
FIRST_RECORD_RULE = ("science, 25 Oct (Gemma e190 K=3): a store change's change point is the start of the first record "
                     "on the new copy, whatever instant its store-proofs entry records for it; the OUT's manifest "
                     "store change's recorded instant (manifest_at) is disclosed beside it with the gap in seconds "
                     "(manifest_gap_seconds: the first record's start minus that instant, positive when the first "
                     "record came later), and so is the entry's own instant with its gap (entry_change_at, "
                     "entry_gap_seconds, the same way); the entry's proof checks are held to the first record, the "
                     "instant the copy took the OUT's attempts; the order check is unchanged and "
                     "strict on either side: a non-branch record on the old copy that starts after the first record on "
                     "the new copy, or is still running when it starts, refuses as a mix of copies, and no record on "
                     "the new copy can start before the change point, which is the earliest start on that copy; where "
                     "the branch-phase rule says every change_at precedes the branch-phase instant it means these "
                     "change points, the first records, not an instant an entry records")


# check-stores prints one fixed line on stderr (the verdict on stdout is unchanged) when a disclosed instant, the entry's
# or the manifest's, is further from the first record than this: the rule accepts it, with no tolerance, and a reader
# should look (Opus review)
WIDE_GAP_SECONDS = 3600
WIDE_GAP_NOTE = ("check-stores: note: an entry's recorded change_at or its manifest store change's at is more than an "
                 "hour from the first record on its copy (see first_record_change_points)")


@dataclasses.dataclass(frozen=True)
class FirstRecordPoint:
    """One store change whose store-proofs entry records another instant than the first record on the new copy
    (FIRST_RECORD_RULE): the OUT, the change's 1-based manifest number, the endpoint it moved to, the change point (the
    first record's own ``started_at`` text), the manifest store change's recorded ``at`` text (``manifest_at``) and
    ``manifest_gap_seconds`` (the change point minus it, in seconds, a float), and the entry's ``change_at`` text
    (``entry_change_at``) and ``entry_gap_seconds`` (the same, a float). No count, no other value."""

    out: str
    index: int
    to_endpoint: str
    change_point: str
    manifest_at: str
    manifest_gap_seconds: float
    entry_change_at: str
    entry_gap_seconds: float

    def values(self) -> dict[str, Any]:
        """The three instants and their two gaps, the one form the store section and the check-stores verdict carry."""
        return {"change_point": self.change_point, "manifest_at": self.manifest_at,
                "manifest_gap_seconds": self.manifest_gap_seconds, "entry_change_at": self.entry_change_at,
                "entry_gap_seconds": self.entry_gap_seconds}

    def disclosure(self) -> dict[str, Any]:
        """The values and the rule, as the store section's change carries them."""
        return {**self.values(), "rule": FIRST_RECORD_RULE}


# The one form a disclosed instant (the manifest's, the entry's, the first record's) may take, so that its gap is exact:
# datetime reads more than this (a basic form, a comma, any number of fraction digits, truncating beyond a microsecond),
# and a gap computed from a value other than the one the text gives would be wrong (external reviewer r5, r6)
EXACT_INSTANT = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(?:\.[0-9]{1,6})?(?:Z|[+-](?:[01][0-9]|2[0-3]):[0-5][0-9])")


def manifest_instant_of(mine: Sequence[StoreChange], move: ChangeTimeline) -> tuple[tuple[str, datetime] | None,
                                                                                    str | None]:
    """The OUT's manifest store change that ``move`` is, matched by its number and the two endpoints it moves between
    (so by the endpoint change it records), as its ``at`` text and instant, or why not: exactly one of ``mine`` must
    match, and its ``at`` must be an ISO 8601 time with a zone (FIRST_RECORD_RULE discloses it beside the first record,
    with the gap). The reason names the change by its number, never a manifest string (the caller names the OUT)."""
    found = [c for c in mine if (c.index, c.from_endpoint, c.to_endpoint) == (move.index, move.from_endpoint,
                                                                              move.to_endpoint)]
    if len(found) != 1:
        return None, (f"its manifest store change {move.index} is not matched by exactly one of the OUT's manifest "
                      f"store changes, so the manifest's recorded instant cannot be disclosed beside the first record")
    instant = parse_utc(found[0].at)
    if instant is None:
        return None, (f"its manifest store change {move.index} records no instant (an ISO time with a zone) to "
                      f"disclose beside the first record")
    if not EXACT_INSTANT.fullmatch(found[0].at):
        return None, (f"its manifest store change {move.index} records an instant not written as "
                      f"YYYY-MM-DDTHH:MM:SS[.ffffff] with Z or an offset, so its gap to the first record cannot be "
                      f"disclosed exactly")
    return (found[0].at, instant), None


# The whole-OUT booleans that make a timeline unclean, in the order a verdict names them
TIMELINE_FLAGS = ("interleaved", "overlap", "chain_incomplete", "endpoint_outside_chain", "rows_without_endpoint",
                  "unusable_times", "run_id_in_two_runs")


@dataclasses.dataclass(frozen=True)
class ChangeTimeline:
    """One manifest store change against the records' own timeline. ``first_to_started`` is the smallest started_at of
    the run of records on the new endpoint (every row, set-aside tries included; ``first_to_started_text`` is that
    row's own text) and is the store change's change_at. The four booleans compare the manifest with the records
    (disclosed, never a gate) over the records/ rows alone; ``manifest_at`` is the manifest's ``at`` as written. All are
    None for a change the records do not reach in the chain's order, and a boolean that needs a time the manifest or a
    row lacks is None (unknown), never False. No counts."""

    index: int  # the manifest's 1-based number of the change
    from_endpoint: str
    to_endpoint: str
    manifest_at: Any
    first_to_started: datetime | None
    first_to_started_text: str | None
    first_precedes_manifest_at: bool | None
    manifest_lists_to_endpoint_record_as_before: bool | None
    manifest_omits_from_endpoint_record: bool | None
    setaside_changes_first_instant: bool | None


@dataclasses.dataclass(frozen=True)
class RunSummary:
    """One run of consecutive records on one endpoint, in start order: the endpoint, the verbatim started_at text of its
    earliest row (first in (started_at, record name) order), of its latest-starting row (last in that order), the
    verbatim ended_at text of its latest-ending row (the last in that order among equal ends), and whether it holds a
    records/ row and a set-aside row. No counts, run ids or record names."""

    endpoint: str
    first_started: str
    last_started: str
    last_ended: str
    in_records_dir: bool
    in_setaside: bool


@dataclasses.dataclass(frozen=True)
class StoreTimeline:
    """An OUT's records ordered by their own start, against its manifest store changes' chain of endpoints.

    ``clean``: the runs of consecutive records on one endpoint are exactly the chain, in order, no record names an
    endpoint outside it or none, every record has usable times, and each run ends no later than the next one starts.
    ``interleaved``: all endpoints are in the chain and the run sequence is not the chain nor a part of it in order (the
    copies alternate); ``overlap``: a record of a run still ran when the next run's first record started;
    ``chain_incomplete``: the run sequence is the chain with a copy missing or the last copies not reached.
    ``run_id_in_two_runs``: one run id names records/ records in different runs. ``interleaved``, ``overlap``,
    ``chain_incomplete`` and ``run_id_in_two_runs`` are None (not evaluated) while ``unusable_times``.
    ``unusable_rows`` are record names (at most 5, sorted) of rows without an endpoint or a usable time; ``period_of``
    maps a records/ row's run id to its run's index (a branch-only OUT's: to the chain position of the copy current at
    the row's start, ``branch_only_timeline``). ``runs``: the runs in start order (RunSummary), None while
    ``unusable_times``; its length is the number of runs, and there is no count of rows or records.

    Branch phase (science, 25 Oct ~04:55Z; ``store_timeline``'s ``branch_start``): a row that started at or
    after the OUT's manifest phase-branch instant is a BRANCH ROW. It carries the recorded endpoint of its parent
    attempt (harness/checkloop/online.py:542 copies a branch record's store from its parent record), not the serving
    copy's, so the runs, ``clean``/``interleaved``/``overlap``/``chain_incomplete``, every
    ``ChangeTimeline`` field and ``runs`` are computed over the other rows alone; a branch row must still name an
    endpoint of the chain and have usable times. ``period_of`` maps a branch row's run id to None (no period).
    ``branch_phase_start`` is the manifest's text of the instant (None without a branch phase);
    ``branch_row_endpoints`` the sorted endpoints the branch rows name (no count; () for none; None without a branch
    phase, and while ``unusable_times``, when the rows cannot all be classed)."""

    chain: tuple[str, ...]
    changes: tuple[ChangeTimeline, ...]
    clean: bool
    interleaved: bool | None
    overlap: bool | None
    chain_incomplete: bool | None
    endpoint_outside_chain: bool
    rows_without_endpoint: bool
    unusable_times: bool
    unusable_rows: tuple[str, ...]
    period_of: Mapping[str, int | None]
    run_id_in_two_runs: bool | None = None
    runs: tuple[RunSummary, ...] | None = None
    branch_phase_start: str | None = None
    branch_row_endpoints: tuple[str, ...] | None = None
    # A branch-only OUT (``branch_only_timeline``; science, 25 Oct ~09:5xZ): True, ``branch_only_problems`` why its
    # manifest's periods do not hold with its rows (words and record names, no count), and ``branch_row_periods`` the
    # sorted chain positions (0 = the chain's start) of the copies its branch rows are attributed to. The defaults are
    # every other OUT's, which no code path of this change touches.
    branch_only: bool = False
    branch_only_problems: tuple[str, ...] = ()
    branch_row_periods: tuple[int, ...] = ()


def _is_subsequence(part: Sequence[str], whole: Sequence[str]) -> bool:
    it = iter(whole)
    return all(x in it for x in part)


def store_timeline(rows: Sequence[StoreRow], changes: Sequence[StoreChange], attempt_dir: str,
                   branch_start: BranchPhase | None = None) -> StoreTimeline:
    """The timeline of one OUT: ``rows`` are all its records' store rows (records/ and every set-aside directory,
    ``attempt_dir`` being records/), ``changes`` its manifest's store changes (validated sequential, so the chain is the
    first change's from and every change's to; an endpoint may come twice). The population is every row with an
    endpoint string, ordered by (started_at, record name), consecutive rows on one endpoint making a run. Science,
    25 Oct 00:10Z: the records' own endpoint is the ground truth, so a store change's change_at is the start of the
    first record of the run on its new endpoint, and records of two copies interleaving is a mix, not a late
    timestamp. The manifest's placement of records (pre_change_run_ids) stays as booleans for the report.

    ``branch_start`` (science, 25 Oct ~04:55Z; manifest_branch_phase): the OUT's branch-phase instant. A row
    that started at or after it (equal counts) is a branch row: its recorded endpoint is its parent attempt's, so the
    order check, the runs, ``first_to_started`` and every manifest cross-check boolean are computed over the other rows
    alone, and the row has no period. A branch row still needs an endpoint of the chain and usable times, and never
    completes a chain."""
    if not changes:
        raise Refused("a store timeline needs at least one manifest store change")
    chain = (changes[0].from_endpoint, *(c.to_endpoint for c in changes))
    with_endpoint = [r for r in rows if isinstance(r.endpoint, str) and r.endpoint]
    no_endpoint = [r.name for r in rows if not (isinstance(r.endpoint, str) and r.endpoint)]
    outside = any(r.endpoint not in chain for r in with_endpoint)
    timed, bad_time = [], []
    for r in with_endpoint:
        start, end = parse_utc(r.started), parse_utc(r.ended)
        if start is None or end is None or end < start:
            bad_time.append(r.name)
        else:
            timed.append((start, r.name, r, end))
    unusable = tuple(sorted([*no_endpoint, *bad_time])[:5])
    unresolved = tuple(ChangeTimeline(c.index, c.from_endpoint, c.to_endpoint, c.at, None, None, None, None, None, None)
                       for c in changes)
    branch_text = branch_start.text if branch_start is not None else None
    if bad_time:  # the order is unknown: nothing that depends on it is evaluated
        return StoreTimeline(chain, unresolved, False, None, None, None, outside, bool(no_endpoint), True, unusable, {},
                             None, branch_phase_start=branch_text)
    branch_rows = [t for t in timed if branch_start is not None and t[0] >= branch_start.at]
    kept = [t for t in timed if branch_start is None or t[0] < branch_start.at]
    runs: list[tuple[str, list[tuple[datetime, str, StoreRow, datetime]]]] = []
    for item in sorted(kept, key=lambda t: (t[0], t[1])):
        if runs and runs[-1][0] == item[2].endpoint:
            runs[-1][1].append(item)
        else:
            runs.append((item[2].endpoint, [item]))
    sequence = [endpoint for endpoint, _ in runs]
    overlap = any(max(item[3] for item in a) > b[0][0] for (_, a), (_, b) in itertools.pairwise(runs))
    incomplete = not outside and sequence != list(chain) and _is_subsequence(sequence, chain)
    interleaved = not outside and sequence != list(chain) and not incomplete
    attempts = [(index, item[2].run_id) for index, (_, items) in enumerate(runs) for item in items
                if item[2].directory == attempt_dir]
    period_of: dict[str, int | None] = {}
    two_runs = any(isinstance(run_id, str) and period_of.setdefault(run_id, index) != index
                   for index, run_id in attempts)  # a flag, not a raise: the command prints it as a STOP
    for item in branch_rows:  # a branch row has no period; one run id in a run and in the branch phase is the same flag
        if item[2].directory == attempt_dir and isinstance(item[2].run_id, str):
            two_runs = period_of.setdefault(item[2].run_id, None) is not None or two_runs
    clean = sequence == list(chain) and not outside and not overlap and not no_endpoint and not two_runs
    summaries = tuple(RunSummary(
        endpoint, items[0][2].started, items[-1][2].started,
        items[max(range(len(items)), key=lambda i: (items[i][3], i))][2].ended,
        any(item[2].directory == attempt_dir for item in items),
        any(item[2].directory != attempt_dir for item in items)) for endpoint, items in runs)
    resolved = []
    for k, c in enumerate(changes):
        if sequence[:k + 2] != list(chain[:k + 2]):  # the records do not reach this change in the chain's order
            resolved.append(unresolved[k])
            continue
        to_run = runs[k + 1][1]
        first = to_run[0]  # the smallest (started_at, name) of the run on the new endpoint
        mine = [item for item in to_run if item[2].directory == attempt_dir]
        first_attempt = mine[0][0] if mine else None
        manifest_at = parse_utc(c.at)

        def is_before(run_id: Any, c: StoreChange = c) -> bool:
            return isinstance(run_id, str) and run_id in c.before

        resolved.append(ChangeTimeline(
            c.index, c.from_endpoint, c.to_endpoint, c.at, first[0], first[2].started,
            None if first_attempt is None or manifest_at is None else first_attempt < manifest_at,
            any(index >= k + 1 and is_before(run_id) for index, run_id in attempts),
            any(index == k and not is_before(run_id) for index, run_id in attempts),
            first_attempt != first[0]))
    return StoreTimeline(chain, tuple(resolved), clean, interleaved, overlap, incomplete, outside, bool(no_endpoint),
                         False, unusable, period_of, two_runs, summaries, branch_text,
                         None if branch_start is None else tuple(sorted({t[2].endpoint for t in branch_rows})))


def timeline_problems(out: str, line: StoreTimeline) -> list[str]:
    """Why an OUT's timeline is not clean (names the OUT and which booleans are true, no counts), or nothing."""
    if line.clean:
        return []
    flags = [f for f in TIMELINE_FLAGS if getattr(line, f)]
    found = []
    if flags or not line.branch_only:  # (a branch-only OUT may be unclean for its own problems alone)
        why = f"{out}: its records' own endpoints and times do not follow its manifest store changes ({', '.join(flags)})"
        if line.interleaved or line.overlap:
            why += ": records of the copies interleave or overlap in time, a mix of copies, not a late timestamp"
        if line.chain_incomplete:
            why += ": no record names a copy of its chain, or the last copies of it"
        if line.run_id_in_two_runs:
            why += ": one run id names records/ records in different runs of the endpoints"
        if line.unusable_rows:
            why += f"; records without an endpoint or usable times: {list(line.unusable_rows)}"
        found.append(why)
    return found + [f"{out}: {p}" for p in line.branch_only_problems]


def branch_copy_endpoint(out: str, line: StoreTimeline, branch: BranchPhase) -> str:
    """The endpoint of the copy a branch-phase row's live retry ran on (science, 25 Oct ~04:55Z): the copy current when
    the runner entered the branch phase, the chain's last. That holds only when every store change precedes the branch
    phase, which the records must show: each change's first record on its new copy (``first_to_started``, taken from
    the rows outside the branch phase) is a time before ``branch.at``. A change the records do not reach, or one that
    is not before it, is a refusal naming the OUT and the change by its number, never a manifest string."""
    for ct in line.changes:
        if not isinstance(ct.first_to_started, datetime) or not ct.first_to_started < branch.at:
            raise Refused(f"{out}: its store change {ct.index} is not shown by the records to precede the branch phase, "
                          f"so the copy current at the branch phase is not the chain's last")
    return line.chain[-1]


# --- the branch-only OUT (science, 25 Oct ~09:5xZ, option (a): the e150 OUTs) -----------------------------------------
#
# Change-log entry 150's UniProt OUTs hold the model's Tier 1 parents, copied by file (manifest ``entry150.copied``, the
# store change's ``pre_change_run_ids``), and only the branch phase's own rows. Every one of those rows records the copy
# of its parent (online.py:542), so no record names the copy the OUT moved to and the order check cannot show the chain
# (today: chain_incomplete). For such an OUT the manifest's own store changes are the only evidence of the copy a live
# retry ran on, so the periods come from their instants.
BRANCH_ONLY_ORDER = "order-branch.json"
LEFT_OUT_ORDERS = ("order-natural.json", "order-prompted.json")  # what the copied-parents builder says it left out
BRANCH_ONLY_RULE = ("a branch-only OUT (a manifest branch phase and store changes, its builder's declaration of the "
                    "Tier 1 parents it copied with the order files it left out, and order-branch.json the only order "
                    "file of its directory) takes the periods of its copies from the instants of its manifest's store "
                    "changes, because its records name its parents' copies alone: in place of the chain's last copy "
                    "of the branch-phase rule, each branch row's live retry is attributed to the copy current at its "
                    "start (a change's at begins its new copy's period, and a start at that instant belongs to the new "
                    "copy); a branch row before the first change, a row made before the branch phase that records "
                    "another copy than the one current at its start, a row made before the branch phase that is not "
                    "a copied parent the first change lists, that is a second row of one, or that started at or after "
                    "that change (or a listed parent without such a row), instants that are not strictly increasing, "
                    "an attempt that runs across a change, one run id in two periods, or an identity whose store or "
                    "store changes do not follow the chain's copies to its last copy refuses; every copy of the chain "
                    "needs its proof and its class (science, 25 Oct ~09:5xZ)")
BRANCH_ONLY_ATTRIBUTION = ("a branch row records its parent attempt's copy (branch_attempt in harness/checkloop/"
                           "online.py), but its live retry ran on the copy current at its start by the manifest's "
                           "store changes: the row counts for that copy, and a checker arm's retried output whose "
                           "parent attempt ran on another copy is a unit row naming both copies, its retry attempt "
                           "counted for the copy of its start in the attempt shares")


def copied_parents_declared(manifest: Mapping[str, Any], changes: Sequence[StoreChange]) -> bool:
    """Whether the manifest carries its builder's declaration of copied Tier 1 parents (the ``entry150`` block that
    ``entry150_run.py build`` writes): ``copied.records`` is exactly the number of run ids the first store change lists
    as made before it (its ``pre_change_run_ids``, each once), and ``left_out`` records that no branch phase of the
    source came with the copy (``branch_phase``: null where the source had none pinned, ``{"starts": n}`` where the build
    removed it, n the count of its starts entries it took out, here only checked to be an integer >= 0;
    entry150_run.py's prune_manifest writes one or the other) and that
    its natural and prompted order files were left out. This is what tells an OUT that never
    ran those phases from an ordinary OUT whose order files are gone or renamed; the manifest's ``starts``,
    ``additions`` and ``grid_order`` say nothing here, they still carry the phases of the manifest the OUT was derived
    from."""
    block = manifest.get("entry150")
    copied = block.get("copied") if isinstance(block, dict) else None
    left = block.get("left_out") if isinstance(block, dict) else None
    if not (isinstance(copied, dict) and isinstance(left, dict) and changes and changes[0].index == 1):
        return False
    listed = manifest.get("store_changes")  # the first change as the manifest lists it (changes[0] is its parse)
    ids = listed[0].get("pre_change_run_ids") if isinstance(listed, list) and listed and isinstance(listed[0], dict) else None
    if not (isinstance(ids, list) and all(isinstance(i, str) for i in ids) and len(set(ids)) == len(ids)):
        return False  # (a run id listed twice would make the count of records and of ids differ)
    files = left.get("files")
    removed = left.get("branch_phase", "absent")
    none_came = removed is None or (isinstance(removed, dict) and list(removed) == ["starts"]
                                    and type(removed["starts"]) is int and removed["starts"] >= 0)
    return (_exact(copied.get("records"), len(ids)) and none_came and isinstance(files, list)
            and all(f in files for f in LEFT_OUT_ORDERS))


def is_branch_only(o: Out, branch: BranchPhase | None, changes: Sequence[StoreChange]) -> bool:
    """Whether the OUT is a branch-only OUT: its manifest names the branch phase (``branch``, from
    manifest_branch_phase, not None) and has store changes (``changes``, from manifest_store_changes, not empty), it
    declares the Tier 1 parents it copied and the order files it left out (``copied_parents_declared``), and its
    directory holds ``order-branch.json`` as a regular file (not a link) and no other entry whose name starts with
    "order" (any case; order-natural.json, order-prompted.json, order.json, a backup or a directory included). Both
    the declaration and the names are needed: an ordinary OUT whose order files were removed or renamed has no
    declaration, and a declaration with an order file present is an ordinary OUT's. An unreadable directory, or none, is
    not a branch-only OUT: the OUT is then judged as every other is. (The directory is listed when the call reads the
    OUT, with the records; a change to it later is not seen, as for every file of an OUT.)"""
    if branch is None or not changes or not copied_parents_declared(o.manifest, changes):
        return False
    try:
        with os.scandir(o.path) as it:
            orders = [e for e in it if e.name.lower().startswith("order")]
    except OSError:
        return False
    return len(orders) == 1 and orders[0].name == BRANCH_ONLY_ORDER and orders[0].is_file(follow_symlinks=False)


def branch_only_timeline(base: StoreTimeline, rows: Sequence[StoreRow], changes: Sequence[StoreChange],
                         attempt_dir: str, branch: BranchPhase) -> StoreTimeline:
    """The timeline of a branch-only OUT: ``base`` is ``store_timeline`` of the same ``rows``, ``changes`` and
    ``branch`` (unchanged, so its runs, overlap, endpoints-outside-chain, rows-without-endpoint and unusable-times
    verdicts stand), and the result differs from it as follows, nothing else.

    The copy current at an instant is the chain's copy numbered by the change instants (the manifest's ``at`` of each
    change, ISO 8601 with a zone, strictly increasing) at or before it: a start equal to a change's instant belongs to
    its new copy. A row that started at or after the branch instant (a branch row, any directory) must be at or after
    the first change's instant, else ``branch_only_problems`` names it; it has no period in ``period_of``, as in every
    OUT, and ``branch_row_periods`` lists the chain positions its starts fall in. A row made before the branch phase
    (a copied parent) keeps its recorded copy, which must be the copy current at its start; its ``period_of`` is that
    position, and a run id of such a row in two positions is named. No attempt, branch row or not, may run across a
    change (end after the next change's instant): a mix of copies, as the base's ``overlap``. This replaces, for a branch-only OUT, ``branch_copy_endpoint``'s check that the records show every
    change before the branch phase, which no record can show here. The runs of the rows made before the branch phase
    must follow the chain in order, but need not reach all of it (``chain_incomplete`` is never true here: the copies
    the OUT moved to are named by branch rows alone, which record their parents'); two copies alternating is still
    ``interleaved``. Words and record names only: no count, run id or manifest string."""
    if base.unusable_times:  # nothing that needs the order or the times is evaluated, as in the base
        return dataclasses.replace(base, branch_only=True)
    instants = [parse_utc(c.at) for c in changes]
    problems: list[str] = []
    if any(i is None for i in instants):
        problems.append("a manifest store change's at is not an ISO time with a zone, so the periods of its copies "
                        "are not defined")
    elif any(a >= b for a, b in itertools.pairwise(instants)):
        problems.append("its manifest store changes' instants are not strictly increasing, so the periods of its "
                        "copies are not defined")
    period_of: dict[str, int | None] = dict(base.period_of)
    used: set[int] = set()
    if not problems:
        chain, early, wrong, across, two_periods = base.chain, [], [], [], []
        placed: dict[str, int] = {}  # a run id of a records/ row -> the period of its start (the labels are keyed by it)
        for r in rows:
            start, end = parse_utc(r.started), parse_utc(r.ended)
            if not (isinstance(r.endpoint, str) and r.endpoint) or start is None or end is None:
                continue
            index = bisect.bisect_right(instants, start)  # the changes at or before the start
            if index < len(instants) and end > instants[index]:
                across.append(r.name)  # started in one copy's period, ended after the next copy's began
            if r.directory == attempt_dir and isinstance(r.run_id, str) and placed.setdefault(r.run_id, index) != index:
                two_periods.append(r.name)
            if start >= branch.at:
                used.add(index)
                if index == 0:
                    early.append(r.name)
            else:
                if r.endpoint in chain and r.endpoint != chain[index]:
                    wrong.append(r.name)
                if r.directory == attempt_dir and isinstance(r.run_id, str):
                    period_of[r.run_id] = index
        if early:
            problems.append(f"branch rows started before the OUT's first manifest store change, so the copy they ran "
                            f"on is not defined: {sorted(early)[:5]}")
        if wrong:
            problems.append(f"rows made before the branch phase record another copy than the one current at their "
                            f"start by the manifest's store changes: {sorted(wrong)[:5]}")
        if across:
            problems.append(f"attempts ran across a change of copy (started in one copy's period and ended after the "
                            f"next began), a mix of copies, not a late timestamp: {sorted(across)[:5]}")
        if two_periods:
            problems.append(f"one run id names records/ records made in different periods of the manifest's store "
                            f"changes: {sorted(two_periods)[:5]}")
    sequence = [run.endpoint for run in base.runs or ()]
    in_order = _is_subsequence(sequence, base.chain)
    clean = (in_order and not base.endpoint_outside_chain and not base.overlap and not base.rows_without_endpoint
             and not base.run_id_in_two_runs and not problems)
    return dataclasses.replace(
        base, clean=clean, interleaved=not base.endpoint_outside_chain and not in_order, chain_incomplete=False,
        period_of=period_of, branch_only=True, branch_only_problems=tuple(problems),
        branch_row_periods=tuple(sorted(used)))


def copied_parents_problems(rows: Sequence[StoreRow], changes: Sequence[StoreChange], branch: BranchPhase,
                            attempt_dir: str) -> list[str]:
    """Why the rows made before the branch phase are not exactly the copied parents the manifest lists (the first store
    change's ``pre_change_run_ids``, which ``copied_parents_declared`` ties to the builder's declaration): a row of any
    directory that started before the branch instant and whose run id is not listed (or is no text); a listed run id on
    more than one such row, in any directories (an attempt of the OUT's own beside the copied one); a listed parent that
    started at or after the first change's instant (the change lists it as made before it); and a listed run id that no
    records/ row made before the branch phase has. Words and record names (at most five, sorted): no count and no run
    id. A row without a usable start is left to the timeline, which refuses it. This is what separates a branch-only
    OUT from an ordinary one that kept the declaration after running attempts of its own."""
    listed, first = changes[0].before, parse_utc(changes[0].at)
    unlisted, late, by_id = [], [], defaultdict(list)
    for r in rows:
        start = parse_utc(r.started)
        if start is None or start >= branch.at:
            continue
        if not isinstance(r.run_id, str) or r.run_id not in listed:
            unlisted.append(r.name)
            continue
        by_id[r.run_id].append(r)
        if first is not None and start >= first:
            late.append(r.name)
    twice = [r.name for found in by_id.values() if len(found) > 1 for r in found]
    seen = {i for i, found in by_id.items() if any(r.directory == attempt_dir for r in found)}
    problems = []
    if unlisted:
        problems.append(f"records made before the branch phase that the manifest does not list as copied parents (the "
                        f"first store change's pre_change_run_ids): {sorted(unlisted)[:5]}")
    if twice:
        problems.append(f"records made before the branch phase that share a copied parent's run id, so one of them is "
                        f"an attempt of the OUT's own: {sorted(twice)[:5]}")
    if late:
        problems.append(f"copied parents that started at or after the first store change, which lists them as made "
                        f"before it: {sorted(late)[:5]}")
    if listed - seen:
        problems.append("copied parents the manifest lists (the first store change's pre_change_run_ids) have no "
                        "records/ record made before the branch phase")
    return problems


def branch_only_copy(line: StoreTimeline, instants: Sequence[datetime], started: Any) -> str:
    """The endpoint of the copy current at ``started`` in a branch-only OUT's clean timeline: the chain's copy numbered
    by the change ``instants`` at or before it. A start that is not a time, or precedes the first change, is Refused
    (a clean timeline has none among its branch rows)."""
    start = parse_utc(started)
    if start is None or not instants or start < instants[0]:
        raise Refused("a branch row of a branch-only OUT has no copy current at its start")
    return line.chain[bisect.bisect_right(instants, start)]


# The one sentence every OUT's branch_phase_copies entry carries (no count, host or endpoint in it)
BRANCH_ATTRIBUTION = ("a branch row records its parent attempt's copy (branch_attempt in harness/checkloop/online.py), "
                      "but its live retry ran on the branch copy, the chain's last copy: the row counts for that copy, "
                      "and a checker arm's retried output whose parent attempt ran on another copy is a unit row naming "
                      "both copies, its retry attempt counted for the branch copy in the attempt shares")


def copy_object(out: str, label: str, replicas: Sequence[str],
                classes: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """A copy as the report names it: ``{copy, class, label, basis, source}`` from the copy classes. ``label`` is the
    copy's row label and ``replicas`` the replicas behind it (check_stores' own record, not parsed back from the
    label): one for a replica or "<replica> (two periods)", several for "<r1> then <r2> (one endpoint, unsplittable)",
    whose class, label, basis and source are those of each replica joined with " then " in that order (a null class
    reads "null"). No replica behind the label (an endpoint label) or a replica the table lacks: Refused."""
    if not replicas:
        raise Refused(f"{out}: its copy {label} has no replica behind it, so it has no speed class")
    found = []
    for replica in replicas:
        if replica not in classes:
            raise Refused(f"{out}: the replica {replica} of its copy {label} has no entry in the copy classes")
        found.append(classes[replica])
    joined = {k: " then ".join(str(e[k]) for e in found) for k in ("label", "basis", "source")}
    speed = found[0]["class"] if len(found) == 1 else " then ".join("null" if e["class"] is None else e["class"]
                                                                  for e in found)
    return {"copy": label, "class": speed, **joined}


def label_class(replicas: Sequence[str], classes: Mapping[str, Mapping[str, Any]]) -> tuple[str | None, str]:
    """A copy label's class for the store rows of retried outputs (addendum 1): ``(its one class or None, its class
    label)`` from the replicas behind it, as ``copy_object`` joins them (label: each replica's, joined with " then ").
    The class is the replicas' one non-null class; None when one is null, the replicas differ in class, or there is no
    entry to read (no replica behind the label, or one the table lacks: NO_COPY_CLASS, the label then too; the caller
    refuses it for a unit on several copies, store_unit_row, and shows it for a unit on one: a single-endpoint model's
    rows have no proven replica). A null class is unknown, not a class of its own."""
    if not replicas or any(r not in classes for r in replicas):
        return None, NO_COPY_CLASS
    found = [classes[r] for r in replicas]
    speeds = {e["class"] for e in found}
    return (speeds.pop() if len(speeds) == 1 else None), " then ".join(str(e["label"]) for e in found)


def copy_classes_module() -> ModuleType:
    """copy_classes.py beside this script (the copy classes' reader and builder), loaded once."""
    module = sys.modules.get("e189_copy_classes")
    if module is None:
        spec = importlib.util.spec_from_file_location("e189_copy_classes", HERE / "copy_classes.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules["e189_copy_classes"] = module
        spec.loader.exec_module(module)
    return module


def read_copy_classes(path: Path) -> tuple[dict[str, dict], str]:
    """The committed copy classes (COPY_CLASSES), validated strictly by copy_classes.read: {replica: {class, label,
    basis, source}} and the sha256 of the file's bytes. A malformed file refuses."""
    module = copy_classes_module()
    try:
        doc, sha = module.read(path)
    except module.Refused as exc:
        raise Refused(f"{path}: {exc}") from None
    return doc["copies"], sha


def read_endpoint_copies(path: Path) -> tuple[list[dict], str]:
    """The committed endpoint copies (ENDPOINT_COPIES), validated strictly by copy_classes.read_endpoint_copies: their
    entries and the sha256 of the file's bytes. A malformed file refuses."""
    module = copy_classes_module()
    try:
        return module.read_endpoint_copies(path)
    except module.Refused as exc:
        raise Refused(f"{path}: {exc}") from None


def endpoint_copy_conflicts(table: Sequence[Mapping[str, Any]], proofs: Sequence[Mapping[str, Any]]) -> list[str]:
    """Where the endpoint copies and the store proofs disagree (empty when they do not): a proofs entry (any OUT, any
    kind: first copy, later period or behind an endpoint) of the table entry's graph and endpoint that names another
    replica than the table's (an endpoint with a copy behind it is not a single copy either), or that names the table's
    replica at another endpoint or graph (a replica is one endpoint of one graph). The table names a copy; it proves
    nothing, so a copy the proofs know can only be confirmed by it, never renamed."""
    problems = []
    for t in table:
        for e in proofs:
            same_endpoint = (e["graph"], e["endpoint_sha256"]) == (t["graph"], t["endpoint_sha256"])
            if same_endpoint and e["replica"] != t["replica"]:
                problems.append(f"the endpoint copies name {t['replica']} for endpoint sha256:"
                                f"{t['endpoint_sha256'][:12]} of {t['graph']}, the store proofs {e['replica']}")
            elif e["replica"] == t["replica"] and not same_endpoint:
                problems.append(f"the endpoint copies name {t['replica']} for endpoint sha256:"
                                f"{t['endpoint_sha256'][:12]} of {t['graph']}, the store proofs for another endpoint "
                                f"or graph")
    return sorted(set(problems))


def read_store_proofs(path: Path) -> tuple[list[dict], str]:
    """The reviewed config of store copies' proofs (STORE_PROOFS). Each entry: the graph (store name), the OUT (its
    on-host path), the endpoint (sha256 of the exact string the records name), the copy's replica, the sha256 of its
    storecheck file and of its known-answer file, the ruling or change-log entry listing it, and ``change_at`` (UTC, the
    start of the first record on the copy, the store timeline's instant; STATE lines only corroborate) for a copy an
    in-OUT change moved to; ``behind_endpoint`` true for a copy behind an
    endpoint the OUT already lists (a route change behind an unchanged forward), which needs its ``change_at``. One
    entry per (OUT, endpoint) is its first copy; refused otherwise. ``second_period`` true (needing ``change_at``, never
    with ``behind_endpoint``) is a later period of that first copy: an OUT that used the copy, left it and came back (a
    manifest chain A -> B -> A) has one such entry per return, the same replica, and is not a second first copy.
    Optional: ``forward`` (FORWARD_FIELDS, for known
    answers at a forwarded copy's direct endpoint), ``alias`` (ALIAS_FIELDS, a pinned STORE_ALIASES pair; never with a
    forward; with ``further`` points, FURTHER_FIELDS: the graph's pinned reference read from that copy, once, and at
    least one other identity check), and ``identity_format`` "ident.py" (IDENT_PY) with ``not_checked`` exactly
    IDENT_PY_NOT_CHECKED's keys, in order (neither without the other)."""
    raw = Path(path).read_bytes()
    doc = json.loads(raw)
    entries = doc.get("proofs") if isinstance(doc, dict) and doc.get("schema") == STORE_PROOFS_SCHEMA else None
    if not isinstance(entries, list):
        raise Refused(f"{path}: not a {STORE_PROOFS_SCHEMA} file with a proofs list")
    seen, first = set(), set()
    for e in entries:
        # an entry is named by its replica and OUT, never by its position (a number)
        who = (f"the entry of {e.get('replica')} for {e.get('out')}" if isinstance(e, dict)
               else "an entry that is not an object")
        if not isinstance(e, dict) or not all(isinstance(e.get(k), str) and e[k] for k in STORE_PROOF_FIELDS):
            raise Refused(f"{path}: {who} lacks one of {STORE_PROOF_FIELDS}")
        if not any(isinstance(e.get(k), str) and e[k].strip() for k in ADMISSION_CITATIONS):
            raise Refused(f"{path}: {who} cites no ruling or change-log entry ({' or '.join(ADMISSION_CITATIONS)})")
        for k in ("endpoint_sha256", "storecheck_sha256", "known_answers_sha256"):
            if not re.fullmatch(r"[0-9a-f]{64}", e[k]):
                raise Refused(f"{path}: {who}: {k} is not a sha256")
        behind = e.get("behind_endpoint", False)
        if not isinstance(behind, bool):
            raise Refused(f"{path}: {who}: behind_endpoint is not true or false")
        second = e.get("second_period", False)
        if not isinstance(second, bool):
            raise Refused(f"{path}: {who}: second_period is not true or false")
        if "change_at" in e and parse_utc(e["change_at"]) is None:
            raise Refused(f"{path}: {who}: change_at is not a UTC time")
        if behind and "change_at" not in e:
            raise Refused(f"{path}: {who}: a copy behind an endpoint needs its switch instant (change_at)")
        if second and behind:
            raise Refused(f"{path}: {who}: a later period of a copy (second_period) is not also a copy behind an "
                          f"endpoint (behind_endpoint)")
        if second and "change_at" not in e:
            raise Refused(f"{path}: {who}: a later period of a copy (second_period) needs its change_at, the "
                          f"start of the first record of that period")
        fwd = e.get("forward")
        if "forward" in e and not (isinstance(fwd, dict) and set(fwd) == set(FORWARD_FIELDS) and all(
                isinstance(fwd[k], str) and re.fullmatch(r"[0-9a-f]{64}", fwd[k]) for k in FORWARD_FIELDS)):
            raise Refused(f"{path}: {who}: forward is not the sha256s of its direct endpoint and of its record "
                          f"({', '.join(FORWARD_FIELDS)})")
        ali = e.get("alias")
        if "alias" in e and not (isinstance(ali, dict) and set(ali) == set(ALIAS_FIELDS) and all(
                isinstance(ali[k], str) and re.fullmatch(r"[0-9a-f]{64}", ali[k]) for k in ALIAS_FIELDS)):
            raise Refused(f"{path}: {who}: alias is not the sha256s of its alias endpoint and of its copy table "
                          f"({', '.join(ALIAS_FIELDS)})")
        if "alias" in e and "forward" in e:
            raise Refused(f"{path}: {who}: an entry names a forward or an alias, not both")
        fur = e.get("further")
        if "further" in e and not (isinstance(fur, list) and fur and all(
                isinstance(f, dict) and set(f) == set(FURTHER_FIELDS) and isinstance(f["point"], str)
                and f["point"].strip() and all(isinstance(f[k], str) and re.fullmatch(r"[0-9a-f]{64}", f[k])
                                               for k in FURTHER_FIELDS[1:]) for f in fur)):
            raise Refused(f"{path}: {who}: further is not a list of points ({', '.join(FURTHER_FIELDS)})")
        named = [e["storecheck_sha256"], e["known_answers_sha256"],
                 *(f[k] for f in fur or () for k in FURTHER_FIELDS[1:])]
        if len(set(named)) != len(named):
            raise Refused(f"{path}: {who}: its further points name files distinct from its pair and from one "
                          f"another")
        if "alias" in e and not (isinstance(fur, list) and len(fur) >= 2 and sum(
                f["storecheck_sha256"] == REFERENCE_STATS_SHA256.get(e["graph"]) for f in fur) == 1):
            raise Refused(f"{path}: {who}: an alias entry needs its further points: the graph's pinned reference "
                          f"read from that copy, once, and another identity check")
        if ("identity_format" in e or "not_checked" in e) and (
                e.get("identity_format") != IDENT_PY or e.get("not_checked") != list(IDENT_PY_NOT_CHECKED)):
            raise Refused(f"{path}: {who}: an identity_format other than the storecheck's is {IDENT_PY!r} only, "
                          f"with not_checked exactly {list(IDENT_PY_NOT_CHECKED)}")
        key = (e["out"], e["endpoint_sha256"])
        if second:  # a later period of the copy its ordinary entry proves: the same replica, a distinct change_at
            if ("second_period", *key, e["replica"], parse_utc(e["change_at"])) in seen:
                raise Refused(f"{path}: {who} lists {e['replica']} at that OUT's endpoint twice for one later "
                              f"period (the same change_at)")
            seen.add(("second_period", *key, e["replica"], parse_utc(e["change_at"])))
            continue
        if (*key, e["replica"]) in seen:
            raise Refused(f"{path}: {who} lists {e['replica']} at that OUT's endpoint twice")
        seen.add((*key, e["replica"]))
        if not behind:
            if key in first:
                raise Refused(f"{path}: {who}: a second first copy of one OUT's endpoint (a copy behind it is "
                              f"behind_endpoint)")
            first.add(key)
    return entries, sha256_bytes(raw)


def read_proof_files(paths: Sequence[Path]) -> dict[str, tuple[str, Any]]:
    """The recorded check files given (--store-proof), by sha256: (path, content). A file that is not JSON must be a
    forward's watchdog log (forward_log)."""
    files = {}
    for p in paths:
        raw = Path(p).read_bytes()
        try:
            content = json.loads(raw)
        except ValueError:
            content = forward_log(raw)
            if content is None:
                raise Refused(f"{p}: a store proof that is neither JSON nor a forward's watchdog log") from None
        files[sha256_bytes(raw)] = (str(p), content)
    return files


@dataclasses.dataclass(frozen=True)
class ForwardLog:
    """A spoke forward's watchdog log (spoke_forward_watchdog.sh): its watchdog lines in order, each ("start", instant,
    ssh pid, listen address, target) for an ssh it started, ("exit", instant, ssh pid) when that ssh exited, or ("end",
    instant) when the watchdog ended, and the watchdog names they carry. Other lines (ssh's own messages) are not
    read."""

    events: tuple[tuple[str, ...], ...]
    names: frozenset[str]


def forward_log(raw: bytes) -> ForwardLog | None:
    """The watchdog log in ``raw``, or None when it is not text or names no ssh started."""
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return None
    events, names = [], set()
    for line in text.splitlines():
        if m := FORWARD_START.fullmatch(line):
            events.append(("start", m["at"], m["pid"], m["listen"], m["target"]))
        elif m := FORWARD_EXIT.fullmatch(line):
            events.append(("exit", m["at"], m["pid"]))
        elif m := FORWARD_END.fullmatch(line):
            events.append(("end", m["at"]))
        else:
            continue
        names.add(m["name"])
    return ForwardLog(tuple(events), frozenset(names)) if any(e[0] == "start" for e in events) else None


def check_forward(log: Any, endpoint: str, direct_sha256: str, change_at: datetime | None) -> tuple[str | None,
                                                                                                    list[str]]:
    """The forward record of an entry's ``forward``: one watchdog's log, every ssh it started listening on the OUT's
    endpoint and forwarding to one target, the direct endpoint "http://<target>/" whose sha256 the entry names; with
    ``change_at``, an ssh of it started by then and not exited by then (so the forward ran when the copy took the OUT's
    attempts), the watchdog not ended by then, its lines in time order. An end must be its last watchdog line. Returns
    that direct endpoint when all hold."""
    if not isinstance(log, ForwardLog):
        return None, ["its forward record is not a forward's watchdog log"]
    problems = []
    if len(log.names) != 1:
        problems.append("its forward record is of more than one watchdog")
    ends = [e for e in log.events if e[0] == "end"]
    if ends and (len(ends) > 1 or log.events[-1][0] != "end"):
        problems.append("its forward record starts an ssh after its watchdog ended")
    starts = [e for e in log.events if e[0] == "start"]
    mappings = {(listen, target) for *_, listen, target in starts}
    if len(mappings) != 1:
        why = "its forward record names more than one mapping (a forward whose target changed is a copy behind it)"
        return None, [*problems, why]
    ((listen, target),) = mappings
    direct = f"http://{target}/"
    if f"http://{listen}" != str(endpoint).rstrip("/"):
        problems.append("its forward record listens on another address than the OUT's endpoint")
    if sha256_text(direct) != direct_sha256:
        problems.append("its forward record forwards to another endpoint than the entry's direct endpoint")
    if direct.rstrip("/") == str(endpoint).rstrip("/"):
        problems.append("its forward record forwards the OUT's endpoint to itself")
    times = [parse_utc(e[1]) for e in log.events]
    if None in times or any(a > b for a, b in itertools.pairwise(times)):
        problems.append("its forward record has a line without its time, or lines out of time order")
    elif change_at is not None:
        running: set[str] = set()  # the ssh pids running at change_at: started, not exited, by then
        for e, t in zip(log.events, times, strict=True):
            if t > change_at:
                break
            if e[0] == "start":
                running.add(e[2])
            elif e[0] == "exit":
                running.discard(e[2])
            else:
                running.clear()
                problems.append("its forward ended before the copy took the OUT's attempts (change_at)")
        if times[0] > change_at:
            problems.append("its forward came up after the copy took the OUT's attempts (change_at)")
        elif not running:
            problems.append("no ssh of its forward was running when the copy took the OUT's attempts (change_at)")
    return (None if problems else direct), problems


def check_alias(table: Any, endpoint: str, alias_sha256: str, replica: str) -> tuple[str | None, list[str]]:
    """The copy table of an entry's ``alias`` (COPY_TABLE_SCHEMA): its "aliases" map the OUT endpoint's host:port and
    one other, the alias "http://<host:port>/" whose sha256 the entry names, to the entry's replica, and that copy's
    "copies" entry lists the alias among its endpoints. Returns the alias endpoint when all hold."""
    if not (isinstance(table, dict) and table.get("schema") == COPY_TABLE_SCHEMA
            and isinstance(table.get("aliases"), dict) and isinstance(table.get("copies"), dict)):
        return None, [f"its alias record is not a {COPY_TABLE_SCHEMA} copy table"]
    mine = urllib.parse.urlsplit(str(endpoint)).netloc
    hits = [k for k in table["aliases"] if sha256_text(f"http://{k}/") == alias_sha256]
    if len(hits) != 1:
        return None, ["its alias record does not hold exactly one key of the entry's alias endpoint"]
    if hits[0] == mine:
        return None, ["its alias is the OUT's endpoint itself"]
    (other,) = hits
    problems = []
    if table["aliases"].get(mine) != replica or table["aliases"][other] != replica:
        problems.append(f"its alias record does not map the OUT's endpoint and the alias to {replica}")
    copy = table["copies"].get(replica)
    if not isinstance(copy, dict) or not isinstance(copy.get("endpoints"), list) or other not in copy["endpoints"]:
        problems.append(f"its alias record's {replica} does not list the alias among its endpoints")
    return (None if problems else f"http://{other}/"), problems


@dataclasses.dataclass(frozen=True)
class Reference:
    """A graph's recorded reference for its copies' proofs: its stats file's sha256 and the index it names (None when
    not recorded), and where the pin came from (the committed constants, or a private pin file by its sha256)."""

    stats_sha256: str | None
    name_index: str | None
    source: str


def reference_pin(graph: str, path: Path | None) -> dict[str, str] | None:
    """A privately pinned reference (PRIVATE_REFERENCE_PINS: a15's): the pin file given (--reference-pin), outside git,
    readable by its owner alone, with its pinned sha256, naming the graph and its reference's sha256. None without a
    file: that graph then has no recorded reference here, so several endpoints of it are refused."""
    if path is None:
        return None
    if graph not in PRIVATE_REFERENCE_PINS:
        raise Refused(f"{graph}: --reference-pin is given only for a privately pinned reference "
                      f"({', '.join(sorted(PRIVATE_REFERENCE_PINS))})")
    pin = outside_git(Path(path))
    if pin.stat().st_mode & 0o777 != 0o600:
        raise Refused(f"{graph}: its reference pin must be readable and writable by its owner alone (owner-only mode)")
    raw = pin.read_bytes()
    if sha256_bytes(raw) != PRIVATE_REFERENCE_PINS[graph]:
        raise Refused(f"{graph}: the reference pin given is not its pinned one (sha256 "
                      f"{PRIVATE_REFERENCE_PINS[graph][:12]})")
    doc = json.loads(raw)
    if (not isinstance(doc, dict) or doc.get("schema") != REFERENCE_PIN_SCHEMA or doc.get("graph") != graph
            or not isinstance(doc.get("reference_sha256"), str)
            or not re.fullmatch(r"[0-9a-f]{64}", doc["reference_sha256"])):
        raise Refused(f"{graph}: its reference pin is not a {REFERENCE_PIN_SCHEMA} file of {graph} naming a sha256")
    return {"graph": graph, "reference_sha256": doc["reference_sha256"]}


def _hostname() -> str:
    return socket.gethostname()


def require_private_a15_path(path: Path, what: str) -> Path:
    """_private_a15_path, with any error that is not its own Refused (a NUL byte in the path: ValueError; a symlink
    loop: RuntimeError; an unreadable parent: OSError, whose text names the path) turned into the guard's own fixed
    refusal: a path the guard cannot resolve is not a path it can accept."""
    try:
        return _private_a15_path(path, what)
    except Refused:
        raise
    except Exception:
        raise Refused(PRIVATE_TREE_REFUSAL.format(what=what)) from None


PRIVATE_TREE_REFUSAL = ("{what} must be under a private a15 tree (the spoke's or SiteB' private a15 directory) on its "
                        "host")


def _private_a15_path(path: Path, what: str) -> Path:
    """``path`` made absolute and fully resolved, or Refused unless it is under one of PRIVATE_A15_ROOTS whose host
    predicate accepts this host's name, that exists as a directory only its owner can enter and is not itself a
    symlink. The path as given (``..`` folded) must be under the root as configured or under the root resolved (the
    kernel names an opened file by its physical path, which differs from the configured one when a directory above the
    root is a symlink), and the path through every symlink must be under the resolved root; so neither a link out of
    the tree nor a link in from outside passes (a path that leaves the tree and comes back through a link is judged by
    where it ends). The refusal names no path. This checks path names and the host's name: a bind mount that puts a
    private path on other storage is not detected."""
    given = Path(os.path.normpath(Path(path).absolute()))
    real = Path(path).absolute().resolve()
    host = _hostname()
    for root, on_host in PRIVATE_A15_ROOTS:
        try:
            if not on_host(host) or root.is_symlink():
                continue
            top = root.resolve(strict=True)
            if not top.is_dir() or top.stat().st_mode & 0o077:
                continue
        except OSError:
            continue
        if (given.is_relative_to(root) or given.is_relative_to(top)) and real.is_relative_to(top):
            return real
    raise Refused(PRIVATE_TREE_REFUSAL.format(what=what))


def check_private_out(path: Path) -> Path:
    """``--out`` of a call with a private table, checked before anything is read: under a private a15 tree of this host
    (require_private_a15_path), outside any git work tree (not inside one, not one itself, not within a .git), and either an existing directory or a missing one whose
    parent is an existing directory (the writer makes one level only), so a result is not lost to an output that could
    never be written. Fixed refusals, no path."""
    what = "--out of a call with a private table"
    out = require_private_a15_path(path, what)
    try:
        outside_git(out)  # (the directories above it)
        if ".git" in out.parts or (out / ".git").exists() or (out / ".git").is_symlink():  # (itself a work tree, or in .git)
            raise Refused("inside git")
    except Refused:
        raise Refused(f"{what} must be outside any git work tree") from None
    except OSError:
        raise Refused(f"{what} cannot be checked for a git work tree") from None
    try:
        usable = out.is_dir() or (not out.exists() and not out.is_symlink() and out.parent.is_dir())
    except OSError:
        usable = False
    if not usable:
        raise Refused(f"{what} must be an existing directory or a new one in an existing directory")
    return out


def _check_open_dir(fd: int) -> None:
    """Refused unless the name the kernel gives the open directory ``fd`` is under a private a15 tree (checked again at
    each step: a directory moved after an earlier check is caught at the next one, not defended against in between)."""
    try:
        where = Path(os.readlink(f"/proc/self/fd/{fd}"))
    except OSError:
        raise Refused("where the output directory of a call with a private table was opened cannot be verified "
                      "here") from None
    require_private_a15_path(where, "the output directory of a call with a private table as opened")


def _open_private_dir(directory: Path, create: bool) -> int:
    """A descriptor of ``directory`` (a private a15 directory the guard resolved), opened without following a link and
    named, as opened, inside the private trees again (/proc/self/fd). With ``create`` a missing directory is made, one
    level only (its parent must exist), inside a descriptor of that parent, which is itself opened without following
    a link and checked the same way: no directory is made through a link, and none above the checked parent; a made
    directory that then does not check out (its parent was moved in between) has its removal attempted (it is empty;
    a failure is ignored)."""
    cannot = "the output directory of a call with a private table cannot be opened or made"

    def opened(name: str, dir_fd: int | None) -> int:
        try:
            fd = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=dir_fd)
        except OSError:
            raise Refused(cannot) from None
        try:
            _check_open_dir(fd)
        except Refused:
            os.close(fd)
            raise
        return fd

    try:
        return opened(str(directory), None)
    except Refused:
        if not create or directory.exists() or directory.is_symlink():
            raise
    parent = opened(str(directory.parent), None)  # the directory is missing: make it, one level, inside its parent
    try:
        try:
            os.mkdir(directory.name, 0o700, dir_fd=parent)
        except OSError:
            raise Refused(cannot) from None
        try:
            return opened(directory.name, parent)
        except Refused:
            try:
                os.rmdir(directory.name, dir_fd=parent)  # the directory just made, empty: not left where it should not be
            except OSError:
                pass
            raise
    finally:
        try:
            os.close(parent)
        except OSError:  # (nothing more can be done with it; the directory returned above is not affected)
            pass


def write_in_private_dir(directory: Path, name: str, text: str, *, create: bool = False) -> None:
    """_write_in_private_dir, with any error that is not its own Refused (a close that fails while a refusal is being
    raised, a path that cannot be read: their text names the path) turned into one fixed refusal."""
    try:
        _write_in_private_dir(directory, name, text, create=create)
    except Refused:
        raise
    except Exception:
        raise Refused("the output file of a call with a private table could not be written") from None


def _write_in_private_dir(directory: Path, name: str, text: str, *, create: bool = False) -> None:
    """``text`` as the file ``name`` (exactly mode 600, whatever the umask) of ``directory``, a private a15 directory
    the guard resolved (created, one level, with ``create``). The text is encoded first (UTF-8; text that cannot be
    encoded refuses before anything is opened). The directory is opened without following a link and its own name is
    checked against the private trees again, here and again just before the rename (_open_private_dir,
    _check_open_dir). The file is never opened in place: it is written, by its descriptor, under a new exclusive
    temporary name in that directory (no link followed) and renamed onto ``name``, so an existing ``name`` that is a
    symlink, a hard link to a file elsewhere, a FIFO or a file of another mode is replaced, never written through,
    waited on or kept (a directory of that name refuses). Every failure (encoding, opening, making, naming, writing,
    syncing, closing the file, renaming) is a Refused with a fixed text that names no path (write_in_private_dir turns
    any other error into one more fixed text); a failed write tries to remove its temporary file, and a failure to close
    the directory's own descriptor after the rename is ignored (nothing more can be done with it). Not defended against: a process of this user that moves the opened directory
    between two of those checks (the guard keeps a private result off the hub by accident, not against a process with
    the owner's rights)."""
    try:
        data = text.encode("utf-8")
    except UnicodeError:
        raise Refused("the output of a call with a private table cannot be encoded as UTF-8") from None
    dfd = _open_private_dir(directory, create)
    made = False
    temporary = ""
    try:
        try:
            temporary = f".{name}.{secrets.token_hex(8)}.tmp"
            fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=dfd)
        except (OSError, NotImplementedError):
            raise Refused("the output file of a call with a private table cannot be created there") from None
        made = True
        try:
            try:
                os.fchmod(fd, 0o600)  # (the creation mode is cut by the umask: set it exactly)
                view = memoryview(data)
                while view:
                    view = view[os.write(fd, view):]
                os.fsync(fd)
            finally:
                os.close(fd)
        except OSError:
            raise Refused("the output file of a call with a private table cannot be written there") from None
        _check_open_dir(dfd)  # just before the rename: the directory is still where it was
        try:
            os.replace(temporary, name, src_dir_fd=dfd, dst_dir_fd=dfd)
        except OSError:
            raise Refused("the output file of a call with a private table cannot replace the file there") from None
        made = False
    finally:
        if made:
            try:
                os.unlink(temporary, dir_fd=dfd)
            except OSError:
                pass
        try:
            os.close(dfd)
        except OSError:
            pass


@dataclasses.dataclass(frozen=True)
class PrivateTable:
    """A private table of a privately pinned graph, read at run time: the basename and sha256 of its file (all the
    results name of it) and its parsed document."""
    name: str
    sha256: str
    doc: Any


def read_private_table(graph: str, flag: str, path: Path | None, sha: str | None,
                       parse: Callable[[ModuleType, bytes], Any],
                       names: Callable[[Any], Iterable[str]]) -> PrivateTable | None:
    """_read_private_table, with any error that is not its own Refused (an OS error from a close, a parent that cannot be
    read, whose text names the path) turned into one fixed refusal."""
    try:
        return _read_private_table(graph, flag, path, sha, parse, names)
    except Refused:
        raise
    except Exception:
        raise Refused(f"{graph}: the table given by {flag} could not be read") from None


def _read_private_table(graph: str, flag: str, path: Path | None, sha: str | None,
                        parse: Callable[[ModuleType, bytes], Any],
                        names: Callable[[Any], Iterable[str]]) -> PrivateTable | None:
    """A table the call gives for a privately pinned graph alone (PRIVATE_REFERENCE_PINS), named by ``flag`` (its path)
    and ``flag``-sha256 (its pin, on the command line: the table changes with the rulings it records, so its hash is
    not kept in the code as the reference pin's is). None when neither is given. The two are given together; the
    graph is a privately pinned one (any other refuses); the file is outside git (checked on the path given and again
    on where the opened file really is: /proc/self/fd, so a directory swapped after the first check cannot redirect
    the open into a work tree; the name the kernel gives the opened file must be that file, the same device and inode,
    and must not end " (deleted)", so an unlinked file is refused; a bind mount showing a work-tree file under another
    name is not detected), a
    regular file (opened non-blocking, so a FIFO is refused, not waited on), owner-only
    mode (the reference pin's rules), and its bytes, read once from the file as opened, have the pinned sha256 before
    ``parse`` (the loader of copy_classes.py) validates those same bytes; every copy name the table gives (``names``)
    is a plain name (PLAIN_COPY_NAME), because a call with the table prints these names as its copies' labels; the file
    is under a private a15 tree of its host (require_private_a15_path), checked on the path given and again on the
    opened file's own name. No
    refusal names a path, a directory or
    anything the table says: the table is private, and its loader's refusal (which names copies) is not repeated, so
    an operator who needs it runs copy_classes.loads / loads_endpoint_copies on the file."""
    if path is None and sha is None:
        return None
    if path is None or sha is None:
        raise Refused(f"{flag} and {flag}-sha256 are given together or not at all")
    if graph not in PRIVATE_REFERENCE_PINS:
        raise Refused(f"{graph}: {flag} is given only for a privately pinned graph "
                      f"({', '.join(sorted(PRIVATE_REFERENCE_PINS))})")
    if re.fullmatch(r"[0-9a-f]{64}", sha) is None:
        raise Refused(f"{flag}-sha256 is not a sha256 in lowercase hex")
    require_private_a15_path(path, f"{graph}: the table given by {flag}")
    outside = f"{graph}: the table given by {flag} must be outside any git work tree and not a symlink"

    def checked_outside_git(candidate: Path) -> Path:
        try:
            return outside_git(candidate)
        except Refused:  # outside_git names the path (and the work tree): not here
            raise Refused(outside) from None
        except OSError:  # an unreadable .git file or directory on the way up names the path too
            raise Refused(f"{graph}: the table given by {flag} cannot be checked for a git work tree") from None

    given = checked_outside_git(Path(path))
    try:
        fd = os.open(given, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except OSError:
        raise Refused(f"{graph}: the table given by {flag} cannot be opened") from None
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_mode & 0o777 != 0o600:
            raise Refused(f"{graph}: the table given by {flag} must be a regular file readable and writable by its "
                          f"owner alone (owner-only mode)")
        try:
            where_text = os.readlink(f"/proc/self/fd/{fd}")
            if where_text.endswith(" (deleted)"):  # how the kernel names an unlinked file (a hard link of that name too)
                raise OSError("unlinked")
            where = Path(where_text)
            named = os.stat(where)
            if (named.st_dev, named.st_ino) != (info.st_dev, info.st_ino):
                raise OSError("the name is not the opened file")
        except OSError:
            raise Refused(f"{graph}: where the table given by {flag} was opened cannot be verified here") from None
        checked_outside_git(where)  # the file really opened, not the path as it was a moment before
        require_private_a15_path(where, f"{graph}: the table given by {flag} as opened")  # and in a private tree too
        with os.fdopen(fd, "rb", closefd=False) as fh:
            raw = fh.read()
    finally:
        os.close(fd)
    if sha256_bytes(raw) != sha:
        raise Refused(f"{graph}: the table given by {flag} is not its pinned one (sha256 {sha[:12]})")
    module = copy_classes_module()
    try:
        doc = parse(module, raw)
    except module.Refused:
        raise Refused(f"{graph}: the table given by {flag} is not a valid table (its loader refused it)") from None
    if not all(PLAIN_COPY_NAME.fullmatch(name) for name in names(doc)):
        raise Refused(f"{graph}: the table given by {flag} names a copy that is not a plain name")
    return PrivateTable(given.name, sha, doc)


def reference_of(graph: str, pin: Mapping[str, str] | None = None) -> Reference:
    """The graph's recorded reference: the committed pins (REFERENCE_STATS_SHA256, REFERENCE_NAME_INDEX), or for a
    privately pinned graph the pin reference_pin read (the results name that pin's sha256, never its reference's)."""
    if graph in PRIVATE_REFERENCE_PINS:
        if pin is None or pin["graph"] != graph:
            return Reference(None, None, f"a private pin (sha256 {PRIVATE_REFERENCE_PINS[graph]}), not given")
        return Reference(pin["reference_sha256"], REFERENCE_NAME_INDEX.get(graph),
                         f"a private pin (sha256 {PRIVATE_REFERENCE_PINS[graph]})")
    return Reference(REFERENCE_STATS_SHA256.get(graph), REFERENCE_NAME_INDEX.get(graph), "REFERENCE_STATS_SHA256")


def check_storecheck(doc: Any, graph: str, endpoint: str, replica: str, change_at: datetime | None,
                     ref: Reference | None = None) -> list[str]:
    """partition_run.py storecheck's file: of this graph's replica at this endpoint, against the graph's recorded
    reference stats (``ref``, by default the committed pin), identical in every field, the index name among them (so
    the copy serves the reference's index), and checked no later than the copy took the OUT's attempts
    (``change_at``)."""
    ref = ref if ref is not None else reference_of(graph)
    if not isinstance(doc, dict) or doc.get("schema") != STORECHECK_SCHEMA:
        return [f"its storecheck is not a {STORECHECK_SCHEMA} file"]
    problems = []
    if doc.get("graph") != graph or doc.get("replica") != replica:
        problems.append(f"its storecheck is of another graph or replica than {graph}'s {replica}")
    if doc.get("endpoint") != endpoint or doc.get("url") != str(endpoint).rstrip("/") + "/?cmd=stats":
        problems.append("its storecheck is of another endpoint than the OUT's")
    if ref.stats_sha256 is None:
        problems.append(f"{graph} has no recorded reference stats here (REFERENCE_STATS_SHA256, or its private pin)")
    elif doc.get("reference_sha256") != ref.stats_sha256:
        problems.append(f"its storecheck compares with another reference than {graph}'s recorded stats")
    if doc.get("identical_to_reference") is not True or doc.get("differ") != []:
        problems.append("its storecheck is not identical to the reference stats")
    if not isinstance(doc.get("fields"), list) or "name-index" not in doc["fields"]:
        problems.append("its storecheck does not compare the index (name-index)")
    checked = parse_utc(doc.get("checked_utc"))
    if checked is None:
        problems.append("its storecheck has no check time")
    elif change_at is not None and checked > change_at:
        problems.append("its storecheck was made after the copy took the OUT's attempts (change_at)")
    return problems


def _same(a: Any, b: Any) -> bool:
    """Equal and of one JSON type (True == 1 in Python, not in a store's stats)."""
    return type(a) is type(b) and a == b


def check_ident_py(doc: Any, graph: str, endpoint: str, change_at: datetime | None, ref: Reference | None,
                   files: Mapping[str, tuple[str, Any]], alias: str | None = None) -> list[str]:
    """The store agent's ident.py file (IDENT_PY) in place of a storecheck: its one copy at this endpoint reachable,
    identical and without mismatches, its recorded stats equal, field by field in type and value, to the pinned
    reference file's reference_stats (the file given by --store-proof under the graph's pinned sha256, holding exactly
    IDENT_PY_FIELDS), and so serving the reference's index (name-index); the file's own reference stats that same
    reference; checked no later than ``change_at``. Its file's own verdict is never taken alone. What the format does
    not compare is the entry's not_checked (IDENT_PY_NOT_CHECKED). With an entry's proven ``alias`` (check_alias), its
    one copy may be at that alias instead."""
    ref = ref if ref is not None else reference_of(graph)
    if not (isinstance(doc, dict) and isinstance(doc.get("reference_stats"), dict)
            and isinstance(doc.get("copies"), dict)):
        return [f"its identity check is not an {IDENT_PY} file"]
    problems = []
    pinned_stats = None
    if ref.stats_sha256 is None:
        problems.append(f"{graph} has no recorded reference stats here (REFERENCE_STATS_SHA256, or its private pin)")
    elif ref.stats_sha256 not in files:
        problems.append(f"its reference file (sha256 {ref.stats_sha256[:12]}) was not given (--store-proof)")
    else:
        pinned = files[ref.stats_sha256][1]
        pinned_stats = pinned.get("reference_stats") if isinstance(pinned, dict) else None
        if not isinstance(pinned_stats, dict) or sorted(pinned_stats) != sorted(IDENT_PY_FIELDS):
            problems.append(f"{graph}'s reference file does not hold exactly the fields {IDENT_PY} compares")
            pinned_stats = None
    theirs = doc["reference_stats"]
    if pinned_stats is not None and (sorted(theirs) != sorted(IDENT_PY_FIELDS)
                                     or not all(_same(theirs[f], pinned_stats[f]) for f in IDENT_PY_FIELDS)):
        problems.append(f"its identity check compares with other stats than {graph}'s recorded reference")
    if doc.get("reference_missing_fields") != []:
        problems.append("its identity check's reference lacks fields it compares")
    at = {str(endpoint).rstrip("/"), *([alias.rstrip("/")] if alias else [])}
    mine = [c for url, c in doc["copies"].items() if str(url).rstrip("/") in at]
    if len(mine) != 1:
        return [*problems, "its identity check does not have exactly one entry of the OUT's endpoint"]
    c = mine[0] if isinstance(mine[0], dict) else {}
    if c.get("reachable") is not True or c.get("identical") is not True or c.get("mismatches") != {}:
        problems.append("its identity check's copy is not identical to the reference stats")
    stats = c.get("stats") if isinstance(c.get("stats"), dict) else {}
    if pinned_stats is not None:
        differ = [f for f in IDENT_PY_FIELDS if f not in stats or not _same(stats[f], pinned_stats[f])]
        if differ:
            problems.append(f"its identity check's copy differs from {graph}'s recorded reference in {differ}")
    if ref.name_index is None or stats.get("name-index") != ref.name_index:
        problems.append(f"its identity check's copy does not serve {graph}'s reference index (name-index)")
    checked = parse_utc(doc.get("checked_utc"))
    if checked is None:
        problems.append("its identity check has no check time")
    elif change_at is not None and checked > change_at:
        problems.append("its identity check was made after the copy took the OUT's attempts (change_at)")
    return problems


def check_known_answers(doc: Any, graph: str, endpoint: str, replica: str, change_at: datetime | None,
                        reference: Reference | None = None, direct: str | None = None,
                        alias: str | None = None) -> list[str]:
    """The copy's known answers, KNOWN_ANSWERS of KNOWN_ANSWERS passed, at this endpoint: load10.py's output, or the
    store agent's check-copies file's one entry of this replica (which also records the index: it must be the
    reference's, by name and index-list hash), made no later than ``change_at``. A copy reached through a forward is
    proven by load10.py's output through that forward, as the store moves run it, or by a check-copies entry at its
    ``direct`` endpoint, the one its entry's forward record proves (check_forward). load10.py's output may be at an
    entry's proven ``alias`` (check_alias)."""
    problems = []
    if isinstance(doc, dict) and isinstance(doc.get("copies"), list):  # check-copies.sh
        mine = [c for c in doc["copies"] if isinstance(c, dict) and c.get("replica") == replica]
        if len(mine) != 1:
            return [f"its known-answer file (check-copies) does not have exactly one entry of {replica}"]
        c = mine[0]
        load = c.get("load") if isinstance(c.get("load"), dict) else {}
        ref = (reference if reference is not None else reference_of(graph)).name_index
        at = str(c.get("endpoint") or "").rstrip("/")
        if at != str(endpoint).rstrip("/") and (direct is None or at != direct.rstrip("/")):
            problems.append("its check-copies entry is of another endpoint than the OUT's"
                            + ("" if direct is None else " or its forward's direct endpoint"))
        if c.get("store") != graph or not all(c.get(k) is True for k in ("ok", "identical_stats", "index_identity_ok")):
            problems.append(f"its check-copies entry is not a passed check of a {graph} copy")
        if ref is None or c.get("name_index") != ref or c.get("index_list_sha256") != ref.split("@", 1)[1]:
            problems.append(f"its check-copies entry's index is not {graph}'s reference index")
        if not (_exact(load.get("passed"), KNOWN_ANSWERS) and _exact(load.get("of"), KNOWN_ANSWERS)):
            problems.append("its known answers did not all pass")
        made = parse_utc(doc.get("checked_utc"))
    elif isinstance(doc, dict) and "passed" in doc and "results" in doc:  # load10.py
        if str(doc.get("endpoint") or "").rstrip("/") not in {str(endpoint).rstrip("/"),
                                                               *([alias.rstrip("/")] if alias else [])}:
            problems.append("its known answers were run at another endpoint than the OUT's")
        results = doc.get("results")
        if not (_exact(doc.get("concurrent"), KNOWN_ANSWERS) and _exact(doc.get("passed"), KNOWN_ANSWERS)
                and isinstance(results, list) and len(results) == KNOWN_ANSWERS):
            problems.append("its known answers did not all pass")
        made = parse_utc(doc.get("started_utc"))
    else:
        return ["its known-answer file is neither load10.py's output nor a check-copies file"]
    if made is None:
        problems.append("its known-answer file has no time")
    elif change_at is not None and made > change_at:
        problems.append("its known answers were run after the copy took the OUT's attempts (change_at)")
    return problems


def period_identity(entry: Mapping[str, Any], endpoint: Any, files: Mapping[str, tuple[str, Any]]) -> tuple[
        tuple[str, tuple] | None, str | None]:
    """What one period's identity file records of the copy, to set beside another period's: for an
    entry180-store-check/2 file its ``reference_sha256`` and ``stats_sha256``, for an ident.py file the copy's
    IDENT_PY_FIELDS values at the endpoint (or the entry's proven alias), each with the kind of file. Or, in place of
    that pair, why the file cannot be read for it."""
    given = files.get(entry["storecheck_sha256"])
    doc = given[1] if given is not None else None
    if entry.get("identity_format") == IDENT_PY:
        if not (isinstance(doc, dict) and isinstance(doc.get("copies"), dict)):
            return None, f"its identity check is not an {IDENT_PY} file"
        at = {str(endpoint).rstrip("/")}
        table = files.get(entry["alias"]["record_sha256"]) if "alias" in entry else None
        if table is not None:
            alias, _ = check_alias(table[1], str(endpoint), entry["alias"]["endpoint_sha256"], entry["replica"])
            if alias is not None:
                at.add(alias.rstrip("/"))
        mine = [c for url, c in doc["copies"].items() if str(url).rstrip("/") in at]
        stats = mine[0].get("stats") if len(mine) == 1 and isinstance(mine[0], dict) else None
        if not isinstance(stats, dict) or any(f not in stats for f in IDENT_PY_FIELDS):
            return None, f"its identity check's copy does not record the fields {IDENT_PY} compares"
        return (IDENT_PY, tuple(stats[f] for f in IDENT_PY_FIELDS)), None
    if not isinstance(doc, dict) or not all(isinstance(doc.get(k), str) and doc[k]
                                            for k in ("reference_sha256", "stats_sha256")):
        return None, "its storecheck does not record both its reference_sha256 and its stats_sha256"
    return ("storecheck", (doc["reference_sha256"], doc["stats_sha256"])), None


def _period_name(entry: Mapping[str, Any]) -> str:
    """A copy's period as a refusal names it: by the instant it began (never by its number)."""
    return f"the period from {entry['change_at']}" if "change_at" in entry else "the period without a change_at"


def periods_identity(used: Sequence[Mapping[str, Any]], endpoint: Any,
                     files: Mapping[str, tuple[str, Any]]) -> list[str]:
    """The identity checks of one copy's periods (its ordinary entry, then each second_period entry) set beside one
    another (science, 25 Oct 00:10Z, point iii: a copy used in two periods is accepted provided both periods' identity
    checks are identical). Each period's file is already held to the SAME pinned reference by check_storecheck or
    check_ident_py, field by field, so each period equals the reference; in addition the periods' own records must be
    equal to one another where they are of one format (period_identity): the same reference_sha256 and stats_sha256 for
    storecheck files, the same IDENT_PY_FIELDS values, by type and value, for ident.py files. Periods of different
    formats share no field beyond the reference, so they are compared through it alone (both identical to it)."""
    problems, seen = [], {}
    for e in used:
        found, why = period_identity(e, endpoint, files)
        if found is None:
            problems.append(f"{_period_name(e)}: {why}, so its identity cannot be set beside the other periods'")
            continue
        kind, values = found
        if kind not in seen:
            seen[kind] = (_period_name(e), values)
        elif not all(_same(a, b) for a, b in zip(seen[kind][1], values, strict=True)):
            problems.append(f"the copy's identity differs between its periods ({seen[kind][0]} and {_period_name(e)}, "
                            f"both {kind} files)")
    return problems


PERIOD_WORDS = ("two", "three", "four", "five", "six", "seven", "eight", "nine")  # a copy's label by its periods
RECORDED_CHECK = re.compile(r"\(store check sha256 (?P<sha>[0-9a-f]{64})\)$")


@dataclasses.dataclass
class StoreCopies:
    """The call's store copies: each records/ record's row (its copy's label), each OUT's manifest store changes (OUTs
    with several endpoints of a model) and their timelines, every (OUT, endpoint)'s label, the copies as results
    describe them, the (graph, model)s with several endpoints and the config read."""

    run_row: dict[str, str]
    changes: dict[str, list[StoreChange]]
    labels: dict[tuple[str, Any], str]
    copies: dict[str, dict]
    several: list[dict]
    config: dict
    timelines: dict[str, StoreTimeline] = dataclasses.field(default_factory=dict)  # per OUT with manifest store changes
    # per (OUT, endpoint), the instant each period of the copy began as its entry gives it (None: no change_at, the
    # chain's start or a copy no entry proves), in period order
    period_starts: dict[tuple[str, Any], list[str | None]] = dataclasses.field(default_factory=dict)
    # per OUT with manifest store changes and a branch phase, sorted by OUT: the copies its branch rows record (their
    # parents') and the copy their live retry ran on, each with its speed class (science, 25 Oct ~04:55Z)
    branch_phase_copies: list[dict] = dataclasses.field(default_factory=list)
    # the copy classes read (replica -> entry) and, per (OUT, copy label), the replicas behind the label (none: an
    # endpoint label): what a retried output's class label reads (label_class)
    classes: dict[str, dict] = dataclasses.field(default_factory=dict)
    replicas_of: dict[tuple[str, str], list[str]] = dataclasses.field(default_factory=dict)
    # per (OUT, manifest change number), the moves for which the entry's recorded instant is not the first record on the
    # new copy (FIRST_RECORD_RULE); empty where every entry's instant is the first record
    first_record_points: dict[tuple[str, int], FirstRecordPoint] = dataclasses.field(default_factory=dict)


def check_stores(at: ModuleType, outs: Sequence[Out], graph: str, proof_paths: Sequence[Path] = (),
                 config_path: Path | None = None, pin: Mapping[str, str] | None = None,
                 classes_path: Path | None = None, endpoint_copies_path: Path | None = None,
                 private_classes: PrivateTable | None = None,
                 private_endpoints: PrivateTable | None = None, *, plain_union: bool = False) -> StoreCopies:
    """The store endpoints of every record (attempts and set-aside tries), read by STORE_FIELDS alone. A (graph, model)
    whose records name one endpoint runs as before. With several, each (OUT, endpoint) of them is refused unless it is
    listed in the OUT (its deployment identity's store blocks, store_chain; or its manifest store changes' from/to) and
    its first copy is proven by a STORE_PROOFS entry (check_storecheck and check_known_answers on the files given, the
    storecheck the one the identity records for that endpoint where it records one, and change_at for a copy a manifest
    store change moved to: the change point is the start of the first record on that copy, store_timeline, and an
    ordinary entry whose change_at is another instant is accepted with both instants and their gap disclosed
    (FIRST_RECORD_RULE, StoreCopies.first_record_points; science, 25 Oct, Gemma e190 K=3), its proof checks held to the
    first record, where its OUT name is one OUT's alone in the call; a second_period entry's change_at, and that of an
    entry whose OUT name is also another OUT's, must be that start to the millisecond); the records' own
    endpoints and times must follow its store changes' chain in order, no two copies interleaving (rows made once the
    OUT's runner entered the branch phase, BranchPhase, are outside that order, and each one's live retry is attributed
    to the chain's last copy: branch_copy_endpoint; science, 25 Oct ~04:55Z).
    A branch-only OUT (``is_branch_only``; science, 25 Oct ~09:5xZ) is followed by its manifest's change instants
    (``branch_only_timeline``): every copy of its chain is examined and held to what a needed copy is held to, named by
    a record or not, an entry's change_at is the instant of its manifest change, its identity's store is the chain's
    last copy, and a branch row's label is that of the copy current at its start (``branch_only_copy``) in place of
    the chain's last.
    Any entry for an (OUT, endpoint) the records name is checked whatever the count; a copy behind an endpoint
    (behind_endpoint)
    needs the endpoint listed and its first copy's entry, and makes that endpoint's row unsplittable. The proofs compare
    with the graph's recorded reference (reference_of: the committed pins, or a privately pinned graph's ``pin``).

    A copy the OUT used, left and came back to (science, 25 Oct 00:10Z, point iii: a manifest chain A -> B -> A) is one
    copy in several periods: the endpoint's occurrences in the chain are numbered; the first needs its ordinary entry
    (with a change_at unless the endpoint is the chain's start), each later one exactly one ``second_period`` entry of
    the same replica whose change_at is the start of the first record of that period (the timeline's), and an endpoint
    used once admits none; an endpoint has one ordinary entry among all the paths the OUT is known by. The periods'
    identity checks must be identical (periods_identity). prove()'s tests of what
    the OUT's identity records (its replica, its storecheck) are those of the endpoint's last period, so they are made
    for its last entry alone. The periods are one row of the report's copies, labelled "<replica> (two periods)".

    The endpoint copies (ENDPOINT_COPIES; science, 25 Oct ~05:3xZ: every copy a look reads has a class) name the copy
    behind an endpoint that no proof entry and no identity names a replica for (the OUTs with a single store endpoint,
    every Claude OUT and some corporate ones): its label and class only, in the one place below that labels an endpoint
    without a proof entry. (A replica the table names for an endpoint must have a copy-classes entry wherever the
    endpoint is used, proven or not: that check is the table's one other read.) They are never a proof: ``needed``, the entries (``first``, ``behind``, ``second``), the order
    check, the timelines and the branch-phase attribution do not read them, so an OUT whose records name several
    endpoints still needs its STORE_PROOFS entries. A table entry that disagrees with a store-proofs entry or with an
    identity's replica for its endpoint refuses (endpoint_copy_conflicts).

    A privately pinned graph's own copy-class table (``private_classes``, read_private_table: --private-copy-classes)
    is merged into the committed copy classes by copy_classes.merge_classes (every copy of either; a copy in both is
    refused unless its two entries are identical), so each check below that reads a copy's class reads the merged
    table. The results name the private table's file basename and sha256 under ``config.private_copy_classes`` and
    nothing else of it; without the table the results are those of a call that never had the option.
    Its endpoint-to-copy table (``private_endpoints``: --private-endpoint-copies) is the private counterpart of the
    endpoint copies: a privately pinned graph's OUTs name no replica (no deployment identity and no committed
    endpoint-copies row), so without it their endpoint labels have no replica behind them and the private class table
    is never reached. Its entries must be of the call's graph; it is merged into the committed table as one document
    (copy_classes.merge_endpoint_copies: no endpoint twice, a replica for one endpoint only) and read exactly as the
    committed one is: a label and a class only, never a proof. The results name its basename and sha256 under
    ``config.private_endpoint_copies``.

    ``plain_union`` (a plain partition union of a privately pinned graph; design ruling (b), 25 Oct, extended to every
    (OUT, endpoint)): an (OUT, endpoint) the several-endpoints rule would hold to a STORE_PROOFS entry is instead held to the
    pinned description: it must be listed by its OUT (as for every pair) and be the private endpoint copies' endpoint of the
    replica the OUT's own identity names for it; no entry is required for it (one given is still checked as always). A pair the
    private endpoint copies do not list for that replica is refused. Without it every pair needs its entry, as ever."""
    config_path = Path(config_path) if config_path is not None else HERE / STORE_PROOFS
    entries, config_sha = read_store_proofs(config_path)
    classes_path = Path(classes_path) if classes_path is not None else HERE / COPY_CLASSES
    classes, classes_sha = read_copy_classes(classes_path)
    if private_classes is not None:  # a privately pinned graph's own table: one fixed rule (copy_classes.merge_classes)
        module = copy_classes_module()
        try:
            merged = module.merge_classes(classes, private_classes.doc["copies"])
        except module.Refused as exc:
            raise Refused(str(exc)) from None
        # the results print a copy's class, label, basis and source (copy_object): a copy only the private table
        # describes keeps its class and label and shows fixed words for the table's basis and source, so nothing of the
        # private table but those two (and its basename and sha256 in the config) reaches the results
        classes = {copy: (entry if copy in classes else {**entry, "basis": PRIVATE_BASIS, "source": PRIVATE_SOURCE})
                   for copy, entry in merged.items()}
    endpoint_copies_path = Path(endpoint_copies_path) if endpoint_copies_path is not None else HERE / ENDPOINT_COPIES
    table, table_sha = read_endpoint_copies(endpoint_copies_path)
    conflicts = endpoint_copy_conflicts(table, entries)  # the committed table's, named as ever
    if conflicts:
        raise Refused("the endpoint copies disagree with the store proofs: " + " | ".join(conflicts[:10]))
    if private_endpoints is not None:  # a privately pinned graph's own endpoint-to-copy table, merged as one document
        if any(e["graph"] != graph for e in private_endpoints.doc):
            raise Refused(f"{graph}: its private endpoint copies name another graph")
        if endpoint_copy_conflicts(private_endpoints.doc, entries):  # (its conflicts would name its replica and graph)
            raise Refused("the private endpoint copies disagree with the store proofs")
        module = copy_classes_module()
        try:
            table = module.merge_endpoint_copies(table, private_endpoints.doc)
        except module.Refused as exc:
            raise Refused(str(exc)) from None
    table_replica = {e["endpoint_sha256"]: e["replica"] for e in table if e["graph"] == graph}  # this call's graph
    private_replicas = {e["replica"] for e in private_endpoints.doc} if private_endpoints is not None else set()
    if plain_union and private_endpoints is None:  # (prepare's require_plain_pinned_copies refuses this first)
        raise Refused("a plain partition union's endpoints are held to the private endpoint copies, which are not given")
    pinned_replica = ({e["endpoint_sha256"]: e["replica"] for e in private_endpoints.doc}  # (of this graph, checked above)
                      if plain_union else {})

    def shown(replica: str) -> str:  # a replica only the private endpoint table names is not printed
        return "a replica of the private endpoint copies" if replica in private_replicas else replica
    ref = reference_of(graph, pin)
    files = read_proof_files(proof_paths)
    rows = {str(o.path): read_store_rows(at, o.path) for o in outs}
    endpoints: dict[tuple[Any, Any], dict[Any, set[str]]] = defaultdict(lambda: defaultdict(set))
    for out, rs in rows.items():
        for r in rs:
            endpoints[(r.store, r.model)][r.endpoint].add(out)
    several = {key: eps for key, eps in endpoints.items() if len(eps) > 1}
    needed = {(out, ep) for eps in several.values() for ep, where in eps.items() for out in where}
    problems: list[str] = []
    changes: dict[str, list[StoreChange]] = {}
    timelines: dict[str, StoreTimeline] = {}
    branches: dict[str, BranchPhase] = {}  # per OUT with manifest store changes and a branch phase
    branch_only_outs: set[str] = set()  # of them, the branch-only OUTs (is_branch_only)
    labels: dict[tuple[str, Any], str] = {}
    label_replicas: dict[tuple[str, Any], list[str]] = {}  # the replicas behind a label (none: an endpoint label)
    period_starts: dict[tuple[str, Any], list[str | None]] = {}
    first_record_points: dict[tuple[str, int], FirstRecordPoint] = {}
    copies: dict[str, dict] = {}
    matched = 0
    # how many OUTs of the call each name is one of (Out.names): an entry is selected for an OUT by its name alone
    name_count = Counter(name for o in outs for name in o.names())
    for o in outs:
        out = str(o.path)
        chain, broken = store_chain(o.identity or {})  # a union's broken chain is refused by check_union too
        if broken is not None and any(o2 == out for o2, _ in needed):
            problems.append(f"{out}: {broken}")
        parents = (o.identity or {}).get("parents_store")  # a GLM partition's parents' copy: named here, never listed
        named = [*chain, *([parents] if isinstance(parents, dict) else [])]
        replica_of = {b.get("endpoint"): b.get("replica") for b in named if b.get("endpoint")}
        for named_ep, named_replica in replica_of.items():  # an identity's replica against the endpoint copies'
            table_says = table_replica.get(sha256_text(str(named_ep)))
            if named_replica and table_says and named_replica != table_says:
                problems.append(f"{out} {endpoint_label(named_ep)}: its deployment identity names {named_replica}, the "
                                f"endpoint copies {shown(table_says)}")
        recorded = {}  # endpoint -> the storecheck sha256 the identity records for it
        for b in named:
            m = RECORDED_CHECK.search(str(b.get("identity_check") or ""))
            if b.get("endpoint") and m:
                recorded[b["endpoint"]] = m.group("sha")
        listed: dict[Any, list[str]] = defaultdict(list)
        for i, b in enumerate(chain):
            if b.get("endpoint"):
                listed[b["endpoint"]].append("its deployment identity's store" if i == 0
                                             else f"its deployment identity's store change {i}")
        moved_to: set[str] = set()
        mine, why = manifest_store_changes(o, plain_union=plain_union)  # every OUT: a declared change is followed whether or not a model of the
        problems += why  # call names two endpoints in it (an entry is still needed only for a needed pair, below)
        for c in mine:
            listed[c.from_endpoint].append(f"its manifest store change {c.index} (from)")
            listed[c.to_endpoint].append(f"its manifest store change {c.index} (to)")
            moved_to.add(c.to_endpoint)
        # the branch phase is read from every OUT's manifest, with or without store changes: a malformed one refuses
        branch, branch_why = manifest_branch_phase(o)
        problems += branch_why
        if mine:
            moved_to.discard(mine[0].from_endpoint)  # the chain's start is not moved to in its first period
            changes[out] = mine
            timelines[out] = store_timeline(rows[out], mine, at.ATTEMPT_DIR, branch)
            if is_branch_only(o, branch, mine):  # science, 25 Oct ~09:5xZ: the periods are the manifest's (BRANCH_ONLY_RULE)
                timelines[out] = branch_only_timeline(timelines[out], rows[out], mine, at.ATTEMPT_DIR, branch)
                branch_only_outs.add(out)
                identity_store = (o.identity or {}).get("store")
                if not (isinstance(identity_store, dict) and identity_store.get("endpoint") == timelines[out].chain[-1]):
                    problems.append(f"{out}: a branch-only OUT's deployment identity must name, as its store, the last "
                                    f"copy of its manifest's store changes (no record names it)")
                if broken is not None and not any(o2 == out for o2, _ in needed):  # (as for an OUT whose records need it)
                    problems.append(f"{out}: {broken}")
                if broken is None and len(chain) > 1 and [b.get("endpoint") for b in chain] != list(timelines[out].chain):
                    problems.append(f"{out}: a branch-only OUT's deployment identity records store changes that do not "
                                    f"follow the copies of its manifest's store changes, in the same order")
                problems += [f"{out}: {p}" for p in copied_parents_problems(rows[out], mine, branch, at.ATTEMPT_DIR)]
            if branch is not None:
                branches[out] = branch
            if not branch_why:  # without the branch instant the timeline's own verdict would only mislead
                problems += timeline_problems(out, timelines[out])
        line = timelines.get(out)
        chain0 = line.chain[0] if line is not None else None  # the manifest chain's start, not the identity's ``chain``
        moves: dict[Any, list[ChangeTimeline]] = defaultdict(list)  # an endpoint -> the changes that moved the OUT to it
        for ct in line.changes if line is not None else ():
            moves[ct.to_endpoint].append(ct)
        ours = [e for e in entries if e["out"] in o.names()]
        # a branch-only OUT's retries ran on copies no record names: every copy of its chain is listed and proven
        chain_copies = set(line.chain) if out in branch_only_outs else set()
        for ep in sorted({r.endpoint for r in rows[out]} | chain_copies, key=str):
            h = sha256_text(str(ep)) if ep else None
            first = [e for e in ours if e["endpoint_sha256"] == h and not e.get("behind_endpoint")
                     and not e.get("second_period")]
            behind = sorted((e for e in ours if e["endpoint_sha256"] == h and e.get("behind_endpoint")),
                            key=lambda e: parse_utc(e["change_at"]))
            second = sorted((e for e in ours if e["endpoint_sha256"] == h and e.get("second_period")),
                            key=lambda e: parse_utc(e["change_at"]))
            # the OUT's periods on this endpoint, in order: None for the chain's start (no change moved the OUT there),
            # else the change that moved it there; the first is the ordinary entry's, each later one a second_period's
            periods = ([None] if ep == chain0 and line is not None else []) + moves.get(ep, [])
            n_periods = max(len(periods), 1)
            where = f"{out} {endpoint_label(ep)}"
            named_here = table_replica.get(h) if h else None  # the endpoint copies' replica for this endpoint
            if named_here and named_here not in classes:  # whatever else names the copy: a proof, an identity (external reviewer r1, r2)
                problems.append(f"{where}: the endpoint copies name {shown(named_here)}, which has no entry in the copy "
                                f"classes (rebuild copy-classes.json)")
            if ((out, ep) in needed or ep in chain_copies or behind) and ep not in listed:
                problems.append(f"{where}: an endpoint the OUT does not list (its deployment identity's store blocks or "
                                f"its manifest's store_changes)")
            if plain_union and (out, ep) in needed:  # the pinned description in place of an entry (ruling (b), every endpoint)
                if ep in listed and (h is None or pinned_replica.get(h) is None or pinned_replica[h] != replica_of.get(ep)):
                    problems.append(f"{where}: its endpoint is not the pinned endpoint copies' endpoint of the copy its "
                                    f"identity names")
            if (((out, ep) in needed and not plain_union) or ep in chain_copies or behind) and not first:
                problems.append(f"{where}: no {STORE_PROOFS} entry proves its copy")
            if len(first) > 1:  # the OUT is known by several paths (Out.names), and read_store_proofs keys on one string
                problems.append(f"{where}: more than one ordinary entry for the OUT's endpoint (its first period has one "
                                f"entry; a later period is a second_period entry)")
            if second and n_periods == 1:
                problems.append(f"{where}: a second_period entry for an endpoint the OUT uses in one period only")
            elif len(second) > n_periods - 1:
                problems.append(f"{where}: more second_period entries than the OUT has later periods of the copy")
            elif len(second) < n_periods - 1:
                problems.append(f"{where}: a later period of the copy without its entry (a second_period entry for "
                                f"each return to it)")
            if second and behind:
                problems.append(f"{where}: a copy used in two periods and also behind an endpoint (second_period with "
                                f"behind_endpoint): its label and split are not defined")
            # the later periods' entries and the changes that moved the OUT back, paired in order when they match up
            paired = dict(zip(map(id, second), periods[1:], strict=True)) if len(second) == len(periods) - 1 else {}
            # (unpaired when the counts differ: that is the refusal above, and no entry is then tested against a change)
            # prove()'s identity tests (the replica the OUT's identity names, the storecheck it records) are of the
            # endpoint's last period: its ordinary entry when it has no later period, else the last second_period's
            last = id(second[-1]) if second else (id(first[0]) if first else None)
            proofs, errors = [], {}
            began: dict[int, str] = {}  # id(entry) -> the first record's text, for a period the first-record rule timed
            held_to: dict[int, datetime] = {}  # id(entry) -> the change point its proof checks are held to (``point``)
            for kind, e in [*(("first", x) for x in first), *(("behind", x) for x in behind),
                            *(("second", x) for x in second)]:
                matched += 1
                is_second = kind == "second"
                # every ordinary entry as before, unless a later period exists (then the last period's alone)
                tested = id(e) == last or kind == "behind" or (kind == "first" and not second)
                move = paired.get(id(e)) if is_second else (periods[0] if periods and kind == "first" else None)
                # FIRST_RECORD_RULE (science, 25 Oct, Gemma e190 K=3), for the ordinary entry of a period a change
                # began: the change point is the first record on the copy, which the proofs' time bounds are held to,
                # whatever instant the entry records. That holds only where the entry's OUT name is one OUT's alone in
                # the call (``relaxed``): an entry is selected for an OUT by its name, and the exact equality below was
                # what kept one entry from proving the copies of two OUTs that share a name (external reviewer r1). A second_period
                # entry keeps the exact equality too: the entries of a copy's later periods are paired with their
                # changes by sorting on those instants, so an entry that is off could bind its proof files to another
                # period's first record. A chain start (no move), a behind_endpoint entry and an entry of an unclean
                # timeline (already refused) keep their own change_at.
                relaxed = kind == "first" and name_count[e["out"]] == 1 and out not in branch_only_outs
                point = (move.first_to_started if relaxed and move is not None and line.clean
                         and "change_at" in e and isinstance(move.first_to_started, datetime) else None)
                found = prove(e, graph, ep, files, replica_of.get(ep) if tested else None,
                              recorded.get(ep) if tested else None, True if is_second else ep in moved_to, ref,
                              change_point=point)
                errors[id(e)] = bool(found)
                problems += [f"{where} ({e['replica']}): {p}" for p in found]
                if point is not None:
                    held_to[id(e)] = point
                if point is not None and parse_utc(e["change_at"]) != point:
                    began[id(e)] = move.first_to_started_text
                    manifest, why_not = manifest_instant_of(mine, move)  # the ruling's instant, beside the first
                    if manifest is not None and not (EXACT_INSTANT.fullmatch(e["change_at"])
                                                     and EXACT_INSTANT.fullmatch(move.first_to_started_text)):
                        why_not = ("the entry's change_at or the first record's start is not written as "
                                   "YYYY-MM-DDTHH:MM:SS[.ffffff] with Z or an offset, so the gaps cannot be disclosed "
                                   "exactly")
                        manifest = None
                    if manifest is None:
                        problems.append(f"{where} ({e['replica']}): {why_not}")
                    else:
                        first_record_points[(out, move.index)] = FirstRecordPoint(
                            out, move.index, move.to_endpoint, move.first_to_started_text, manifest[0],
                            (point - manifest[1]).total_seconds(), e["change_at"],
                            (point - parse_utc(e["change_at"])).total_seconds())
                if move is not None and line.clean and "change_at" in e and out in branch_only_outs:
                    if parse_utc(e["change_at"]) != parse_utc(move.manifest_at):  # no record names the copy (BRANCH_ONLY_RULE)
                        problems.append(f"{where} ({e['replica']}): its change_at is not the instant of the manifest's "
                                        f"store change (a branch-only OUT's records do not name the copy)")
                elif (not relaxed and move is not None and line.clean and "change_at" in e
                        and parse_utc(e["change_at"]) != move.first_to_started):
                    problems.append(f"{where} ({e['replica']}): its change_at is not the first record on the copy "
                                    f"(the records' own endpoint is the ground truth)")
                if is_second and first and e["replica"] != first[0]["replica"]:
                    problems.append(f"{where} ({e['replica']}): its replica is not the first period's "
                                    f"({first[0]['replica']}): a different replica is a different copy")
                if is_second and first and e["out"] != first[0]["out"]:  # (its graph and endpoint follow from the call)
                    problems.append(f"{where} ({e['replica']}): its OUT is not the first period's (the same OUT under "
                                    f"another path it is known by)")
                proofs.append({"replica": e["replica"], "behind_endpoint": e.get("behind_endpoint", False),
                               "change_at": e.get("change_at"), "storecheck_sha256": e["storecheck_sha256"],
                               "known_answers_sha256": e["known_answers_sha256"],
                               **({"second_period": True} if is_second else {}),
                               **{k: e[k] for k in ADMISSION_CITATIONS if k in e},
                               **({"forward": {"direct_endpoint": "endpoint sha256:"
                                                                 + e["forward"]["direct_endpoint_sha256"][:12],
                                               "record_sha256": e["forward"]["record_sha256"]}}
                                  if "forward" in e else {}),
                               **({"alias": {"endpoint": "endpoint sha256:" + e["alias"]["endpoint_sha256"][:12],
                                             "record_sha256": e["alias"]["record_sha256"]}} if "alias" in e else {}),
                               **({"further": [dict(f) for f in e["further"]]} if "further" in e else {}),
                               **({"identity_format": IDENT_PY,
                                   "not_checked": {k: IDENT_PY_NOT_CHECKED[k] for k in e["not_checked"]}}
                                  if e.get("identity_format") == IDENT_PY else {})})
            used = [*first[:1], *second]  # the copy's periods, in order
            if len(used) > 1 and not any(errors[id(e)] for e in used):  # a failing period is already a refusal
                problems += [f"{where}: {p}" for p in periods_identity(used, ep, files)]
            # the switch instants of a copy behind this endpoint follow the instant the first copy's checks are held to: the
            # change point (the first record) for an entry the first-record rule timed, else its own change_at (Opus review)
            times = [held_to.get(id(e), parse_utc(e["change_at"])) for e in [*first[:1], *behind] if "change_at" in e]
            if behind and any(a >= b for a, b in itertools.pairwise(times)):
                problems.append(f"{where}: its copies' switch instants are not strictly after one another")
            if first and behind:
                label = " then ".join(e["replica"] for e in [*first, *behind]) + " (one endpoint, unsplittable)"
                label_replicas[(out, ep)] = [e["replica"] for e in [*first, *behind]]
            elif first:
                label = first[0]["replica"]
                label_replicas[(out, ep)] = [label]
                if second:  # one copy in several periods: one label, one row of the report's copies
                    n = len(second) + 1  # the copy's periods, named in words where there is one
                    label += f" ({PERIOD_WORDS[n - 2] if n - 2 < len(PERIOD_WORDS) else n} periods)"
            else:  # no proof entry: the replica its identity names, else the endpoint copies' (agreeing, checked above)
                replica = replica_of.get(ep) or named_here
                label = replica or endpoint_label(ep)
                label_replicas[(out, ep)] = [replica] if replica else []
            labels[(out, ep)] = label
            period_starts[(out, ep)] = [began.get(id(e), e.get("change_at")) for e in used] or [None]
            info = copies.setdefault(label, {"endpoints": [], "outs": [], "listed_by": {}, "proofs": []})
            info["endpoints"] = sorted({*info["endpoints"], endpoint_label(ep)})
            info["outs"] = sorted({*info["outs"], out})
            info["listed_by"][out] = listed.get(ep, [])
            info["proofs"] += proofs
            if second:
                info.setdefault("periods", []).extend(
                    {"period": k, **({"change_at": began.get(id(e), e["change_at"])} if "change_at" in e else {}),
                     "storecheck_sha256": e["storecheck_sha256"], "known_answers_sha256": e["known_answers_sha256"]}
                    for k, e in enumerate(used, 1))
            unchecked = sorted({k for p in proofs for k in p.get("not_checked", {})} | set(info.get("not_checked", [])))
            if unchecked:  # what a copy's identity file does not compare, carried into the report with its shares
                info["not_checked"] = unchecked
            if behind:
                info["split"] = UNSPLITTABLE
                info["switch_instants"] = [e["change_at"] for e in behind]
    if problems:
        raise Refused("store copies not listed or not proven (several store endpoints per graph and model are analysed "
                      "only listed in their OUT and identity-proven): " + " | ".join(problems[:10]))
    # A branch row records its parent's endpoint, but its live retry ran on the copy current at its start: the chain's
    # last (science, 25 Oct ~04:55Z). The membership test matters: a row the timeline did not place is not a branch row.
    # The timeline places a row by its run id and leaves out one whose run id is not a string, which would then keep
    # its recorded (parent's) copy unnoticed: such a row in the branch phase is a refusal (External review of T4, r1).
    for out, branch in sorted(branches.items()):
        nameless = sorted(r.name for r in rows[out] if r.directory == at.ATTEMPT_DIR and not isinstance(r.run_id, str)
                          and (start := parse_utc(r.started)) is not None and start >= branch.at)
        if nameless:
            raise Refused(f"{out}: records/ rows made in the branch phase have no run id that is a string, so their "
                          f"copy cannot be attributed: {nameless[:5]}")
    # (a branch-only OUT's rows are attributed by its manifest's change instants instead: branch_only_copy)
    branch_label = {out: labels.get((out, branch_copy_endpoint(out, timelines[out], branch)))
                    for out, branch in branches.items() if out not in branch_only_outs}
    for out, label in branch_label.items():
        if label is None:  # a clean timeline has a record on every endpoint of its chain, so this is defensive
            raise Refused(f"{out}: the chain's last copy has no label, so the branch-phase rows cannot be attributed")
    # the instants of a branch-only OUT's changes (its timeline is clean here, so each parses and they increase)
    change_instants = {out: [parse_utc(c.at) for c in changes[out]] for out in branch_only_outs}

    def row_label(out: str, r: StoreRow) -> str:
        if out in branch_label and isinstance(r.run_id, str):
            period = timelines[out].period_of
            if r.run_id in period and period[r.run_id] is None:
                return branch_label[out]
        if out in change_instants and isinstance(r.run_id, str):
            period = timelines[out].period_of
            if r.run_id in period and period[r.run_id] is None:  # a branch row: the copy current at its start
                return labels[(out, branch_only_copy(timelines[out], change_instants[out], r.started))]
        return labels[(out, r.endpoint)]

    run_row = {r.run_id: row_label(out, r) for out, rs in rows.items() for r in rs if r.directory == at.ATTEMPT_DIR}

    def copy_of(out: str, endpoint: Any) -> dict[str, Any]:
        return copy_object(out, labels[(out, endpoint)], label_replicas[(out, endpoint)], classes)

    for out in sorted(branch_only_outs):  # every copy of the chain has a class, whether a row is attributed to it or not
        for ep in timelines[out].chain:
            copy_of(out, ep)
    branch_phase_copies = []
    for out, branch in sorted(branches.items()):
        line = timelines[out]
        parents = {labels[(out, ep)]: copy_of(out, ep) for ep in line.branch_row_endpoints or ()}
        if out in branch_only_outs:  # the copies the branch rows' starts fall in, with the instant each period began
            branch_phase_copies.append({
                "out": out, "branch_phase_start": line.branch_phase_start,
                "parent_copies": [parents[label] for label in sorted(parents)],
                "branch_copies": [{**copy_of(out, line.chain[i]), "period_start": changes[out][i - 1].at}
                                  for i in line.branch_row_periods],
                "attribution": BRANCH_ONLY_ATTRIBUTION, "branch_only": True, "branch_only_rule": BRANCH_ONLY_RULE})
            continue
        branch_phase_copies.append({
            "out": out, "branch_phase_start": line.branch_phase_start,
            "parent_copies": [parents[label] for label in sorted(parents)],
            "branch_copy": copy_of(out, branch_copy_endpoint(out, line, branch)),
            "attribution": BRANCH_ATTRIBUTION})
    return StoreCopies(run_row, changes, labels, copies,
                       [{"graph": g, "model": m, "endpoints": len(eps), "outs": len(set().union(*eps.values()))}
                        for (g, m), eps in sorted(several.items(), key=lambda kv: tuple(map(str, kv[0])))],
                       {"path": str(config_path), "sha256": config_sha, "entries": len(entries), "matched": matched,
                        "reference": {"graph": graph, "source": ref.source,
                                      **({} if graph in PRIVATE_REFERENCE_PINS else
                                         {"stats_sha256": ref.stats_sha256, "name_index": ref.name_index})},
                        "copy_classes": {"path": str(classes_path), "sha256": classes_sha},
                        "endpoint_copies": {"path": str(endpoint_copies_path), "sha256": table_sha},
                        **({"private_copy_classes": {"name": private_classes.name, "sha256": private_classes.sha256}}
                           if private_classes is not None else {}),
                        **({"private_endpoint_copies": {"name": private_endpoints.name,
                                                        "sha256": private_endpoints.sha256}}
                           if private_endpoints is not None else {})},
                       timelines, period_starts, branch_phase_copies, classes,
                       {(out, label): list(label_replicas[(out, ep)]) for (out, ep), label in labels.items()},
                       first_record_points)


def prove(entry: Mapping[str, Any], graph: str, endpoint: Any, files: Mapping[str, tuple[str, Any]],
          identity_replica: str | None, identity_check: str | None, moved_to: bool,
          ref: Reference | None = None, change_point: datetime | None = None) -> list[str]:
    """One STORE_PROOFS entry against the (OUT, endpoint) it names: its graph is the call's; a first copy is the replica
    the OUT's identity names at that endpoint (where it names one), its storecheck the one the identity records there,
    and with ``change_at`` when a manifest store change moved the OUT to it; both files given and passing, against the
    graph's recorded reference ``ref`` (by default the committed pin); ``moved_to`` is true for an entry of a period
    that began with a change (a second_period entry always, an ordinary one unless its endpoint is the chain's start),
    and ``identity_replica`` and ``identity_check`` are None for a period that is not the endpoint's last. An ident.py
    entry's identity file is read by
    check_ident_py in place of a storecheck; an entry's ``forward`` record (check_forward) admits check-copies known
    answers at the copy's direct endpoint; an entry's ``alias``, the pinned pair of this graph's endpoint
    (STORE_ALIASES) with its copy table (check_alias), admits ident.py and load10.py files at the alias. Each of an
    entry's ``further`` points is checked as its pair is, with no time bound (corroboration before or after), the
    graph's pinned reference file among them only as read from this copy (its "reference" the endpoint or the
    alias).

    ``change_point`` (FIRST_RECORD_RULE), given for the ordinary entry of a period a change began on a clean
    timeline, is the start of the first record on the copy: the instant the proofs' time bounds are held to in place
    of the entry's ``change_at`` (which must still be present for such an entry). None: the entry's own ``change_at``."""
    ref = ref if ref is not None else reference_of(graph)
    problems = []
    behind = entry.get("behind_endpoint", False)
    if entry["graph"] != graph:
        problems.append(f"an entry of graph {entry['graph']}, not the call's {graph}")
    if not behind and identity_replica is not None and entry["replica"] != identity_replica:
        problems.append(f"the OUT's identity names {identity_replica} at this endpoint")
    if not behind and identity_check is not None and entry["storecheck_sha256"] != identity_check:
        problems.append("its storecheck is not the one the OUT's identity records for this endpoint")
    if not behind and moved_to and "change_at" not in entry:
        problems.append("a copy a store change moved the OUT to, without its change_at")
    change_at = change_point if change_point is not None else parse_utc(entry.get("change_at"))
    direct = None
    if "forward" in entry:
        record = files.get(entry["forward"]["record_sha256"])
        if record is None:
            problems.append(f"its forward record (sha256 {entry['forward']['record_sha256'][:12]}) was not given "
                            f"(--store-proof)")
        else:
            direct, why = check_forward(record[1], str(endpoint), entry["forward"]["direct_endpoint_sha256"],
                                        change_at)
            problems += why
    alias = None
    if "alias" in entry:
        pinned = STORE_ALIASES.get((graph, sha256_text(str(endpoint))))
        if pinned != (entry["alias"]["endpoint_sha256"], entry["alias"]["record_sha256"]):
            problems.append("its alias is not the one pinned for this graph's endpoint (STORE_ALIASES)")
        elif (table := files.get(entry["alias"]["record_sha256"])) is None:
            problems.append(f"its alias record (sha256 {entry['alias']['record_sha256'][:12]}) was not given "
                            f"(--store-proof)")
        else:
            alias, why = check_alias(table[1], str(endpoint), entry["alias"]["endpoint_sha256"], entry["replica"])
            problems += why
    for kind in ("storecheck", "known_answers"):
        given = files.get(entry[f"{kind}_sha256"])
        if given is None:
            problems.append(f"its {kind} file (sha256 {entry[f'{kind}_sha256'][:12]}) was not given (--store-proof)")
        elif kind == "known_answers":
            problems += check_known_answers(given[1], graph, str(endpoint), entry["replica"], change_at, ref, direct,
                                            alias)
        elif entry.get("identity_format") == IDENT_PY:
            problems += check_ident_py(given[1], graph, str(endpoint), change_at, ref, files, alias)
        else:
            problems += check_storecheck(given[1], graph, str(endpoint), entry["replica"], change_at, ref)
    at = {str(endpoint).rstrip("/"), *([alias.rstrip("/")] if alias else [])}
    for f in entry.get("further", ()):
        where = f"its further point (storecheck sha256 {f['storecheck_sha256'][:12]})"
        sc, ka = files.get(f["storecheck_sha256"]), files.get(f["known_answers_sha256"])
        if sc is None or ka is None:
            problems.append(f"{where}: its files were not given (--store-proof)")
            continue
        if f["storecheck_sha256"] == ref.stats_sha256:  # the pinned reference itself, read from this copy
            if not isinstance(sc[1], dict) or str(sc[1].get("reference") or "").rstrip("/") not in at:
                problems.append(f"{where}: the graph's pinned reference was not read from this copy")
        elif entry.get("identity_format") == IDENT_PY:
            problems += [f"{where}: {p}" for p in check_ident_py(sc[1], graph, str(endpoint), None, ref, files, alias)]
        else:
            problems += [f"{where}: {p}" for p in check_storecheck(sc[1], graph, str(endpoint), entry["replica"], None,
                                                                   ref)]
        problems += [f"{where}: {p}" for p in check_known_answers(ka[1], graph, str(endpoint), entry["replica"], None,
                                                                  ref, direct, alias)]
    return problems


def store_section(F: Frozen, loaded: Loaded, dep: Deployments, stores: StoreCopies) -> dict[str, Any]:
    """Per arm, each store copy's share of its attempts over the call (a union's: the union) and per OUT. A checker
    arm's retried output is one unit row (store_unit_row): its copy when its attempts all ran on it, else a row naming
    its copies, "<parent copy> → <retry copies>" (a branch row's live retry is the chain's last copy's: science, 25 Oct
    ~06:2xZ), where the deployment section says 'mixed'. Beside ``shares``, ``class_shares`` count the units' class
    labels and ``attempt_shares`` the attempts, each for its own copy; per manifest store change, each of its OUT's
    arms before and after it (by the periods of the records' own timeline, store_timeline: before is the records of the
    runs up to the change's old copy, after those of the runs after it), with the manifest's placement of records
    beside it as disclosed booleans."""
    arms = attempt_arms(F, loaded)
    by_copy = dataclasses.replace(dep, served=stores.run_row)

    def copy_class(out: str | None) -> Callable[[str], tuple[str | None, str]]:
        def of(label: str) -> tuple[str | None, str]:
            if (out, label) not in stores.replicas_of:  # every row label is a copy of its OUT: a refusal, not a guess
                raise Refused(f"{out}: an analysed attempt's copy {label} is not one of the OUT's copies")
            return label_class(stores.replicas_of[(out, label)], stores.classes)
        return of

    units = {key: [store_unit_row(u, key, by_copy, F.metrics.PROMPT, copy_class(dep.out_of.get(u.attempt.run_id)))
                   for u in us] for key, us in arms.items()}
    rows = {key: [x.row for x in xs] for key, xs in units.items()}

    def units_of_out(key: Any, out: str, keep: Callable[[Any], bool] = lambda u: True) -> list[StoreUnit]:
        return [x for u, x in zip(arms[key], units[key], strict=True)
                if dep.out_of.get(u.attempt.run_id) == out and keep(u)]

    def of_out(key: Any, out: str, keep: Callable[[Any], bool] = lambda u: True) -> list[str | None]:
        return [x.row for x in units_of_out(key, out, keep)]

    def class_labels(xs: Sequence[StoreUnit]) -> list[str | None]:
        return [x.class_label for x in xs]

    def attempt_copies(xs: Sequence[StoreUnit]) -> list[str | None]:
        return [a for x in xs for a in x.attempts]

    outs = sorted(set(dep.out_of.values()))
    by_out = {out: {key.label: share_of(of_out(key, out)) for key in sorted(arms) if of_out(key, out)} for out in outs}
    class_by_out = {out: {key.label: share_of(class_labels(units_of_out(key, out))) for key in sorted(arms)
                          if units_of_out(key, out)} for out in outs}
    attempts_by_out = {out: {key.label: share_of(attempt_copies(units_of_out(key, out))) for key in sorted(arms)
                             if units_of_out(key, out)} for out in outs}
    changes = []
    for out, mine in sorted(stores.changes.items()):
        line = stores.timelines[out]

        def period(u: Any, line: StoreTimeline = line) -> int | None:
            if u.attempt.run_id not in line.period_of:  # an attempt the timeline did not place would otherwise fall
                raise Refused(f"{out}: an analysed attempt has no place in the OUT's store timeline")  # into one side
            return line.period_of[u.attempt.run_id]  # None: a branch-phase row, in no period

        for k, c in enumerate(mine):
            ct = line.changes[k]
            split = {}
            for key in sorted(arms):
                before = of_out(key, out, lambda u, k=k: (p := period(u)) is not None and p <= k)
                after = of_out(key, out, lambda u, k=k: (p := period(u)) is not None and p > k)
                branch = of_out(key, out, lambda u: period(u) is None)
                if before or after or branch:
                    split[key.label] = {"before": share_of(before), "after": share_of(after)}
                    if line.branch_phase_start is not None:  # the third bucket only for an OUT with a branch phase
                        split[key.label]["branch_phase"] = share_of(branch)
            changes.append({"out": out, "index": c.index, "at": c.at, "manifest_at": ct.manifest_at,
                            "branch_phase_start": line.branch_phase_start,
                            "from": stores.labels.get((out, c.from_endpoint), endpoint_label(c.from_endpoint)),
                            "to": stores.labels.get((out, c.to_endpoint), endpoint_label(c.to_endpoint)),
                            "first_record_on_new_endpoint": ct.first_to_started_text,
                            "change_at_used": ct.manifest_at if line.branch_only else ct.first_to_started_text,
                            "first_precedes_manifest_at": ct.first_precedes_manifest_at,
                            "manifest_lists_to_endpoint_record_as_before":
                                ct.manifest_lists_to_endpoint_record_as_before,
                            "manifest_omits_from_endpoint_record": ct.manifest_omits_from_endpoint_record,
                            "setaside_changes_first_instant": ct.setaside_changes_first_instant,
                            "manifest_pre_change_records": len(c.before), "shares_by_arm": split,
                            # only a move whose entry records another instant than the first record (FIRST_RECORD_RULE)
                            **({"first_record_change_point": stores.first_record_points[(out, c.index)].disclosure()}
                               if (out, c.index) in stores.first_record_points else {})})
    return {
        "rule": "science, ~09:50Z (change-log entry 150's in-OUT store changes, the analysis side): several identity-"
                "checked store copies per graph and model are analysed only where each is listed in its OUT and proven "
                f"by {STORE_PROOFS} (its recorded storecheck and known answers); each copy's share of attempts per "
                f"arm; {STORE_CHANGE_RULE}. {BRANCH_PHASE_RULE}",
        "reader": {"fields": list(STORE_FIELDS)}, "config": stores.config, "several_endpoints": stores.several,
        "copies": stores.copies, "branch_phase_copies": stores.branch_phase_copies,
        "not_checked": {label: {k: IDENT_PY_NOT_CHECKED[k] for k in info["not_checked"]}
                        for label, info in sorted(stores.copies.items()) if "not_checked" in info},
        "retried_unit_rule": STORE_UNIT_RULE,
        "shares": {key.label: share_of(r) for key, r in sorted(rows.items())},
        "shares_by_out": by_out,
        "class_shares": {key.label: share_of(class_labels(units[key])) for key in sorted(units)},
        "class_shares_by_out": class_by_out,
        "attempt_shares": {key.label: share_of(attempt_copies(units[key])) for key in sorted(units)},
        "attempt_shares_by_out": attempts_by_out, "store_changes": changes}


# --- every replay row on a copy of its record's own class (science, 25 Oct 11:04Z, on the class cells) ---------------------

REPLAY_ROUTING_SCHEMA = "replay-routing/5"  # driver/replay_matrices.py check writes it: run id -> the class of the replay copy
UNVERIFIABLE = "unverifiable"
NOT_RECORDED = "not_recorded"  # accepted only for older pinned routing files
REPLAY_CLASS_RULE = (
    "Each replay row is checked against the routing written by driver/replay_matrices.py: the replay copy class comes from "
    "the shard's first-line worker launch record, whose bytes match the producer's pinned time-file manifest and whose endpoint "
    "and copy id are bound by cell-classes.tsv. A missing or unusable launch may use only an ok not_recorded shard row of a "
    "pinned t2-attest/1 operator attestation. The class is compared with the record's own copy class. Counts are per replay arm: "
    "matches by proven (recorded cell), recorded (main tree) or attested basis, plus other_class, record_class_unknown, "
    "replay_class_unverified and not_routed exceptions. no_launch_record and pending_attestation are subcounts of "
    "replay_class_unverified; pending attestation means the OUT-phase has no confirmed pinned attestation.")
# The recorded label names the time of the sha manifest that covered the shard's tree-phase: the producer's incremental indexes were
# taken at different times (12:32Z, 15:31Z, ...), and a consolidated index keeps each row's own time. It is never a constant: the
# label of a call lists the distinct times (UTC, to the minute, sorted) of the manifests that covered the shards of the rows it counts.
RECORDED_LABEL = "recorded (worker launch record, unsealed; sha manifest taken {taken})"
RECORDED_TAKEN_NONE = "<the time of the sha manifest row that covered the shard's tree-phase>"  # a call that counts no recorded row
REPLAY_CLASS_LABEL_TEMPLATES = {
    "same_class_proven": "proven",
    "same_class_recorded": RECORDED_LABEL.format(taken=RECORDED_TAKEN_NONE),
    "same_class_attested": "operator attestation (from glue records)",
    "pending_attestation": "pending attestation"}
REPLAY_CLASS_COUNTERS = ("rows", "same_class_proven", "same_class_recorded", "same_class_attested", "other_class",
                         "record_class_unknown", "replay_class_unverified", "no_launch_record", "pending_attestation", "not_routed")


@dataclasses.dataclass(frozen=True)
class ReplayRouting:
    classes: Mapping[str, str]  # run id -> the class of the copy its record was replayed on, or UNVERIFIABLE or NOT_RECORDED
    basis: Mapping[str, str]  # run id -> proven, recorded or attested (only for a run with a real class)
    files: tuple[dict, ...]  # each routing file read: sha256, OUT, records, the sha manifest's time, the attestations
    no_launch: frozenset[str] = frozenset()
    pending_attestation: frozenset[str] = frozenset()
    recorded_taken: Mapping[str, str] = dataclasses.field(default_factory=dict)  # run id -> "YYYY-MM-DDTHH:MMZ" of the manifest that covered its shard (basis recorded only)


def _unique_keys(pairs: list) -> dict:
    """object_pairs_hook: a JSON object that repeats a key is ambiguous (json.loads keeps the last), so it refuses."""
    out: dict = {}
    for k, v in pairs:
        if k in out:
            raise ValueError("repeated key")
        out[k] = v
    return out


def _manifest_minute(text: object) -> str | None:
    """A sha manifest's taken-at time to the minute in the label's form ("2026-10-25T12:32Z"; the producer writes "12:32Z" for
    12:32:38Z, so the seconds are dropped, not rounded), from an index tag (YYYYMMDDTHHMMSSZ) or an ISO UTC time
    (YYYY-MM-DDTHH:MM:SSZ); None for anything else or a date that does not exist."""
    if not isinstance(text, str):
        return None
    m = re.fullmatch(r"([0-9]{4})([0-9]{2})([0-9]{2})T([0-9]{2})([0-9]{2})([0-9]{2})Z", text)
    if m:
        iso = f"{m[1]}-{m[2]}-{m[3]}T{m[4]}:{m[5]}:{m[6]}Z"
    else:
        iso = text
    if not re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z", iso):
        return None
    try:
        datetime.strptime(iso, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError:
        return None
    return iso[:16] + "Z"


def _consolidated_manifest_ok(man: dict) -> bool:
    """The ``sources`` of a routing file written from the consolidated time-file index, and its entries' own times: each source is
    {tag, file_sha256, verified}, tags are index tags without repeat; each manifests entry names one of them and carries the taken-at
    time of the row that covered its tree-phase, one time per source tag, every tree-phase once."""
    sources, entries = man["sources"], man["manifests"]
    sha_ok = lambda x: isinstance(x, str) and re.fullmatch(r"[0-9a-f]{64}", x) is not None  # noqa: E731
    if not (isinstance(sources, list) and sources
            and all(isinstance(x, dict) and set(x) == {"tag", "file_sha256", "verified"} and isinstance(x["tag"], str)
                    and re.fullmatch(r"[0-9]{8}T[0-9]{6}Z", x["tag"]) and _manifest_minute(x["tag"]) and sha_ok(x["file_sha256"])
                    and type(x["verified"]) is bool for x in sources)):
        return False
    tags = [x["tag"] for x in sources]
    time_of: dict = {}
    for m in entries:
        if (not isinstance(m["source_index"], str) or m["source_index"] not in tags or not isinstance(m["taken_at"], str)
                or not re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z", m["taken_at"]) or _manifest_minute(m["taken_at"]) is None
                or time_of.setdefault(m["source_index"], m["taken_at"]) != m["taken_at"]):
            return False
    return len(set(tags)) == len(tags) and len({(m["tree"], m["phase"]) for m in entries}) == len(entries)


def read_replay_routing(paths: Sequence[Path], pins: Sequence[str] = ()) -> ReplayRouting:
    """The replay routing files of a call (``--replay-routing``), strictly: a JSON object of the schema REPLAY_ROUTING_SCHEMA without a
    repeated key (older schemas, which inferred the copy or lacked the basis, refuse), the OUT's name, ``shards`` ("tree/phase/shard" ->
    {class, basis, copy, runs, optional subkind}) and ``inputs``. A shard has a real class (fast, medium, slow) with a basis, or unverifiable/not_recorded
    with none; basis ``proven`` only for a class cell (tree name ``<OUT>-cell-<class>`` of that class), ``recorded`` only for the OUT's own
    tree, and both need the shard's time file sha256 in ``inputs.time_files_sha256`` (and every listed time file a classed or unverifiable
    shard); ``attested`` needs an attestation in ``inputs.attestations`` of this OUT whose phase and per-shard row name it, and no time file.
    Every run is in exactly one shard; a run id in two files refuses. ``pins`` (from the command line, one sha256 per file) are the
    sha256 the checker's output had when the job wrote it: a file that is not byte for byte that refuses, so an edited routing is not
    read (the driver pins every routing it passes)."""
    if pins and len(pins) != len(paths):
        raise Refused("the replay routing files and their sha256 pins differ in number")
    classes: dict[str, str] = {}
    basis_of: dict[str, str] = {}
    seen_global: set[str] = set()
    no_launch: set[str] = set()
    pending_attestation: set[str] = set()
    recorded_taken: dict[str, str] = {}
    files: list[dict] = []
    sha_ok = lambda x: isinstance(x, str) and re.fullmatch(r"[0-9a-f]{64}", x) is not None  # noqa: E731
    real = ("fast", "medium", "slow")
    for i, path in enumerate(paths):
        raw = Path(path).read_bytes()
        if pins and sha256_bytes(raw) != pins[i]:
            raise Refused(f"{path}: the replay routing is not the file the checker wrote (its sha256 is not the pin)")
        try:
            doc = json.loads(raw, object_pairs_hook=_unique_keys)
        except ValueError:
            raise Refused(f"{path}: the replay routing is not JSON (or repeats a key)") from None
        d = doc if isinstance(doc, dict) else {}
        shards, inputs = d.get("shards"), d.get("inputs")
        bad = f"{path}: not a {REPLAY_ROUTING_SCHEMA} file (schema, out, shards and inputs)"
        if not isinstance(doc, dict) or set(d) != {"schema", "out", "shards", "inputs"} or d["schema"] != REPLAY_ROUTING_SCHEMA \
                or not isinstance(d["out"], str) or not isinstance(shards, dict) or not isinstance(inputs, dict):
            raise Refused(bad)
        man, atts, tfs = inputs.get("timefiles_manifest"), inputs.get("attestations"), inputs.get("time_files_sha256")
        consolidated = isinstance(man, dict) and "sources" in man  # written from the consolidated index: per-entry times, all or none
        man_keys = {"file_sha256", "taken_at", "manifests"} | ({"sources"} if consolidated else set())
        entry_keys = {"tree", "phase", "file_sha256"} | ({"source_index", "taken_at"} if consolidated else set())
        copy_ok = lambda c: isinstance(c, dict) and set(c) == {"endpoint", "copy"} and all(isinstance(c[k], str) and c[k] for k in c)  # noqa: E731
        shard_ok = lambda t: (isinstance(t, dict) and set(t) == {"tree", "shard", "class", "copy", "endpoint"}  # noqa: E731
                              and all(isinstance(t[k], str) and t[k] for k in t) and t["class"] in real)
        if (set(inputs) != {"cell_classes_sha256", "time_files_sha256", "timefiles_manifest", "attestations"}
                or not sha_ok(inputs["cell_classes_sha256"]) or not isinstance(tfs, dict)
                or not all(isinstance(k, str) and sha_ok(v) for k, v in tfs.items())
                or not isinstance(man, dict) or set(man) != man_keys
                or not sha_ok(man["file_sha256"])
                or not (isinstance(man["taken_at"], str) and re.fullmatch(r"\d{8}T\d{6}Z", man["taken_at"]))
                or not isinstance(man["manifests"], list)
                or not all(isinstance(m, dict) and set(m) == entry_keys
                           and isinstance(m["tree"], str) and m["phase"] in ("natural", "prompted")
                           and sha_ok(m["file_sha256"]) for m in man["manifests"])
                or not isinstance(atts, list)
                or not all(isinstance(a, dict) and set(a) == {"out", "phase", "shards", "file_sha256", "content_sha256", "agent"}
                           and isinstance(a["out"], str) and a["phase"] in ("natural", "prompted") and sha_ok(a["file_sha256"])
                           and sha_ok(a["content_sha256"]) and isinstance(a["agent"], str)
                           and isinstance(a["shards"], list) and all(shard_ok(t) for t in a["shards"]) for a in atts)):
            raise Refused(bad)
        if _manifest_minute(man["taken_at"]) is None or (consolidated and not _consolidated_manifest_ok(man)):
            raise Refused(bad)
        if any(a["out"] != d["out"] for a in atts):
            raise Refused(f"{path}: an attestation names another OUT")
        covered = {(a["phase"], t["tree"], t["shard"]): (t["class"], t["copy"], t["endpoint"]) for a in atts for t in a["shards"]}
        manifest_pairs = {(m["tree"], m["phase"]) for m in man["manifests"]}
        # the time of the manifest that covered each tree-phase: its own row's (consolidated index), else the single index's
        # taken-at; the consolidated file's own tag is a consolidation time, never read as a label time
        covered_at = {(m["tree"], m["phase"]): _manifest_minute(m["taken_at"] if consolidated else man["taken_at"]) for m in man["manifests"]}
        seen_runs: set[str] = set()
        for key, info in shards.items():
            parts = key.split("/") if isinstance(key, str) else []
            if len(parts) != 3 or not isinstance(info, dict) or not {"class", "basis", "copy", "runs"} <= set(info) or set(info) - {"class", "basis", "copy", "runs", "subkind"}:
                raise Refused(bad)
            tree, phase, _shard = parts
            if (tree, phase) not in manifest_pairs:
                raise Refused(bad)
            subkind = info.get("subkind")
            if subkind is not None and (subkind not in ("no_launch_record", "pending_attestation", "unlisted_time_file", "record_rejected_by_consumer")
                                        or info["class"] != UNVERIFIABLE):
                raise Refused(bad)
            if subkind == "pending_attestation" and any(a["phase"] == phase for a in atts):
                raise Refused(bad)
            cell = re.fullmatch(r"(.+)-cell-(fast|medium|slow)", tree)
            runs, cls, basis, copy = info["runs"], info["class"], info["basis"], info["copy"]
            if (phase not in ("natural", "prompted") or not (tree == d["out"] or (cell and cell.group(1) == d["out"]))
                    or not isinstance(runs, list) or not runs or not all(isinstance(r, str) for r in runs)
                    or len(set(runs)) != len(runs) or seen_runs & set(runs)):
                raise Refused(bad)
            seen_runs.update(runs)
            if cls in real:
                if basis not in ("proven", "recorded", "attested") or not copy_ok(copy):
                    raise Refused(bad)
                if basis == "proven" and not (cell and cell.group(2) == cls):
                    raise Refused(f"{path}: a shard is 'proven' without being a class cell of its own class")
                if basis == "recorded" and cell:
                    raise Refused(f"{path}: a class cell's shard is 'recorded', not proven")
                if basis == "attested":
                    if (cell and cell.group(2) != cls) or covered.get((phase, tree, _shard)) != (cls, copy["copy"], copy["endpoint"]) or key in tfs:
                        raise Refused(f"{path}: an attested shard is not tied to an attestation of this OUT-phase")
                elif key not in tfs:
                    raise Refused(f"{path}: a classed shard has no time file sha256")
            elif cls in (UNVERIFIABLE, NOT_RECORDED):
                if basis is not None or copy is not None:
                    raise Refused(bad)
            else:
                raise Refused(bad)
            for r in runs:
                classes[r] = cls
                if basis:
                    basis_of[r] = basis
                if basis == "recorded":
                    recorded_taken[r] = covered_at[(tree, phase)]
                if subkind in ("no_launch_record", "pending_attestation"):
                    no_launch.add(r)
                if subkind == "pending_attestation":
                    pending_attestation.add(r)
        if not set(tfs) <= set(shards):
            raise Refused(f"{path}: a time file sha256 names no shard")
        twice = seen_runs & seen_global
        if twice:
            raise Refused(f"{path}: {len(twice)} run ids are in two replay routing files")
        seen_global.update(seen_runs)
        files.append({"sha256": sha256_bytes(raw), "out": d["out"], "records": len(seen_runs), "timefiles_manifest": man,
                      "attestations": atts})
    return ReplayRouting(classes, basis_of, tuple(files), frozenset(no_launch), frozenset(pending_attestation), recorded_taken)


def replay_class_check(loaded: Loaded, dep: Deployments, stores: StoreCopies, routing: ReplayRouting) -> dict[str, Any]:
    """The store section's replay-class counts (REPLAY_CLASS_RULE): per replay arm, how many of its rows were replayed on a
    copy of their record's own class (by basis), and the exceptions by kind. Empty without a replay row. Counts only."""
    arms: dict[str, Counter] = {}
    recorded_times: set[str] = set()  # the manifest times of the rows counted as same_class_recorded
    for r in loaded.ledger:
        if r.use != REPLAY_USE or r.cohort == "bank":
            continue
        c = arms.setdefault(f"{r.graph}|{r.model}|{r.cohort}|{REPLAY_USE}|{r.source}", Counter())
        c["rows"] += 1
        out, label = dep.out_of.get(r.run_id), stores.run_row.get(r.run_id)
        own = None if out is None or label is None else label_class(stores.replicas_of.get((out, label), []), stores.classes)[0]
        got = routing.classes.get(r.run_id)
        if got is None:
            c["not_routed"] += 1
        elif own is None:
            c["record_class_unknown"] += 1
        elif got == NOT_RECORDED:
            c["replay_class_unverified"] += 1
            c["no_launch_record"] += 1
        elif got == UNVERIFIABLE:
            c["replay_class_unverified"] += 1
            if r.run_id in routing.no_launch:
                c["no_launch_record"] += 1
            if r.run_id in routing.pending_attestation:
                c["pending_attestation"] += 1
        elif got == own:
            c[f"same_class_{routing.basis[r.run_id]}"] += 1
            if routing.basis[r.run_id] == "recorded":
                recorded_times.add(routing.recorded_taken[r.run_id])
        else:
            c["other_class"] += 1
    labels = dict(REPLAY_CLASS_LABEL_TEMPLATES)
    if recorded_times:  # the time(s) of the manifest row(s) that covered the shards of the rows counted as recorded
        labels["same_class_recorded"] = RECORDED_LABEL.format(taken=", ".join(sorted(recorded_times)))
    matched = ("same_class_proven", "same_class_recorded", "same_class_attested")
    return {"rule": REPLAY_CLASS_RULE, "labels": labels, "routing": list(routing.files) if arms else [],  # empty without a replay row
            "arms": {k: {**{n: c.get(n, 0) for n in REPLAY_CLASS_COUNTERS},
                         "exceptions": c["rows"] - sum(c[m] for m in matched)} for k, c in sorted(arms.items())}}


# --- the store timeline tool (science, 25 Oct 00:10Z, point iv) ---------------------------------------------------

# One fixed line per refusal class on stderr (no counts, no names, no endpoints)
STOP_LINES = {
    "interleaved": "STOP interleaved: records of the copies alternate in time, a mix of copies, not a late timestamp",
    "overlap": "STOP overlap: an attempt on the old copy ran past the first on the new one, a mix of copies, not a "
               "late timestamp",
    "chain_incomplete": "STOP chain_incomplete: no record names a copy of the chain, or the last ones of it",
    "endpoint_outside_chain": "STOP endpoint_outside_chain: a record names an endpoint the store changes do not list",
    "rows_without_endpoint": "STOP rows_without_endpoint: a record names no store endpoint",
    "unusable_times": "STOP unusable_times: a record lacks a usable started_at or ended_at",
    "run_id_in_two_runs": "STOP run_id_in_two_runs: one run id names records in different runs of the endpoints",
    "manifest_store_changes_malformed": "STOP manifest_store_changes_malformed: the manifest's store_changes do not "
                                        "join up",
    "branch_phase_malformed": "STOP branch_phase_malformed: the manifest's additions do not name the branch phase "
                              "once, with an ISO time and zone",
}


def timeline_report(at: ModuleType, path: Path) -> tuple[dict[str, Any], list[str]]:
    """One OUT's store timeline as the tool prints it, and the refusal classes (STOP_LINES keys) it names, none when the
    OUT is clean or has no manifest store changes. Reads the manifest (its sha256 and store_changes) and each record's
    STORE_FIELDS; prints no count, run id, outcome or host: endpoints are labels."""
    raw = (Path(path) / "manifest.json").read_bytes()
    manifest = json.loads(raw)
    if not isinstance(manifest, dict):
        raise ValueError("the manifest is not an object")
    out = Out(Path(path), None, manifest)
    changes, problems = manifest_store_changes(out)
    branch, branch_problems = manifest_branch_phase(out)  # of every OUT, with or without store changes
    rows = read_store_rows(at, Path(path))
    report: dict[str, Any] = {"out": str(path)}
    if problems or branch_problems:
        stops = [*(["manifest_store_changes_malformed"] if problems else []),
                 *(["branch_phase_malformed"] if branch_problems else [])]
        report |= {"manifest_sha256": sha256_bytes(raw), **{stop: True for stop in stops},
                   "verdict": "STOP: " + ", ".join(stops)}
        return report, stops
    if not changes:
        report |= {"store_changes": False, "several_endpoints": len({r.endpoint for r in rows if r.endpoint}) > 1,
                   "branch_phase_start": None if branch is None else branch.text}
        return report, []
    line = store_timeline(rows, changes, at.ATTEMPT_DIR, branch)
    flags = [f for f in TIMELINE_FLAGS if getattr(line, f)]
    report |= {
        "manifest_sha256": sha256_bytes(raw), "chain": [endpoint_label(e) for e in line.chain],
        "changes": [{"index": ct.index, "from": endpoint_label(ct.from_endpoint), "to": endpoint_label(ct.to_endpoint),
                     "manifest_at": ct.manifest_at, "first_record_on_new_endpoint": ct.first_to_started_text,
                     "first_precedes_manifest_at": ct.first_precedes_manifest_at,
                     "manifest_lists_to_endpoint_record_as_before": ct.manifest_lists_to_endpoint_record_as_before,
                     "manifest_omits_from_endpoint_record": ct.manifest_omits_from_endpoint_record,
                     "setaside_changes_first_instant": ct.setaside_changes_first_instant} for ct in line.changes],
        "branch_phase_start": line.branch_phase_start,
        "branch_row_endpoints": (None if line.branch_row_endpoints is None
                                 else sorted(endpoint_label(e) for e in line.branch_row_endpoints)),
        "clean": line.clean, "interleaved": line.interleaved, "overlap": line.overlap,
        "chain_incomplete": line.chain_incomplete, "endpoint_outside_chain": line.endpoint_outside_chain,
        "rows_without_endpoint": line.rows_without_endpoint, "unusable_times": line.unusable_times,
        "run_id_in_two_runs": line.run_id_in_two_runs, "unusable_rows": list(line.unusable_rows),
        "runs": None if line.runs is None else [
            {"endpoint": endpoint_label(r.endpoint), "first_started": r.first_started, "last_started": r.last_started,
             "last_ended": r.last_ended, "in_records_dir": r.in_records_dir, "in_setaside": r.in_setaside}
            for r in line.runs],
        "verdict": "CLEAN" if line.clean else "STOP: " + ", ".join(flags)}
    return report, [] if line.clean else flags


def store_timeline_main(code_root: Path, outs: Sequence[Path]) -> int:
    """``store-timeline``: one JSON object per OUT on stdout (sorted keys), a fixed line per refusal class on stderr.
    Exit 0 when every OUT is CLEAN or has no manifest store changes, 2 when any is STOP (all are still printed), 3 when
    the code root, the attribution check or an OUT cannot be read (3 wins over 2; the message is the path and the
    exception class only)."""
    try:
        load_frozen(code_root)
        at = load_attribution()
    except Exception as exc:  # 2 means STOP here, so a refused code root is 3
        print(f"store-timeline: the frozen analysis or the attribution check cannot be loaded ({type(exc).__name__})",
              file=sys.stderr)
        return 3
    code, classes = 0, set()
    for path in outs:
        try:
            report, flags = timeline_report(at, Path(path))
        except Exception as exc:
            print(json.dumps({"out": str(path), "unreadable": type(exc).__name__}, sort_keys=True))
            print(f"unreadable OUT {path}: {type(exc).__name__}", file=sys.stderr)
            code = 3
            continue
        print(json.dumps(report, sort_keys=True))
        if flags:
            classes.update(flags)
            code = max(code, 2)
    for name in STOP_LINES:
        if name in classes:
            print(STOP_LINES[name], file=sys.stderr)
    return code


def check_stores_main(code_root: Path, graph: str, batches: Sequence[Path], store_proofs: Sequence[Path] = (),
                      store_config: Path | None = None, copy_classes: Path | None = None,
                      endpoint_copies: Path | None = None) -> int:
    """``check-stores``: the store check of a call (check_stores, nothing else: no questions, voided record, partition
    record or replay check is read) and its verdict as one JSON object on stdout, so that a refusal reaches the operator
    whose stderr is private. The graph is validated as analyse validates it (graph_rule: GRAPHS and the frozen config),
    its store named as prepare() names it, the OUTs read as prepare() reads them (Out.read). A privately pinned graph
    (PRIVATE_REFERENCE_PINS: a15) is refused: its reference pin and records stay in the private tree, and this command
    takes no pin. Success: see _check_stores_object. Refused by the graph's validation, that rule or check_stores:
    ``ok`` false and ``refused``, the message, exit 2. Anything else, the code root and the formatting of the verdict
    included: ``ok`` false and ``error``, the exception class only, exit 3. ``store_config``, ``copy_classes`` and
    ``endpoint_copies`` are for tests (another store-proofs.json, copy-classes.json and endpoint-copies.json than the
    committed ones)."""
    try:
        F = load_frozen(code_root)
        cfg, _ = frozen_config(F)
        at = load_attribution()
    except Exception as exc:  # a refused code root is 3 here: 2 is the store check's own verdict
        return _check_stores_failed(exc)
    code = 0
    wide = False  # a disclosed instant more than WIDE_GAP_SECONDS from the first record (FIRST_RECORD_RULE)
    try:
        spec, rule = graph_rule(graph, cfg)
        if rule is None:
            raise unknown_graph(graph, cfg)
        store = spec["store"] if spec else graph
        if store in PRIVATE_REFERENCE_PINS:
            raise Refused(f"{graph}: check-stores does not run for a privately pinned graph "
                          f"({', '.join(sorted(PRIVATE_REFERENCE_PINS))}): its reference pin and its records stay in "
                          f"the private tree")
        outs = [Out.read(b) for b in batches]
        stores = check_stores(at, outs, store, tuple(store_proofs), store_config, None, copy_classes,
                              endpoint_copies)
        text = json.dumps(_check_stores_object(stores), sort_keys=True)  # inside the try: a failure here is a 3
        wide = any(abs(gap) > WIDE_GAP_SECONDS for p in stores.first_record_points.values()
                   for gap in (p.manifest_gap_seconds, p.entry_gap_seconds))
    except Refused as exc:
        text, code = json.dumps({"ok": False, "refused": str(exc)}, sort_keys=True), 2
    except Exception as exc:
        return _check_stores_failed(exc)
    print(text)
    if wide:  # an operator's note outside the verdict (which stays as it is), one fixed line
        print(WIDE_GAP_NOTE, file=sys.stderr)
    return code


def _check_stores_object(stores: StoreCopies) -> dict[str, Any]:
    """The success verdict of check-stores: ``ok`` true; ``copies`` per label, with ``endpoints`` as labels, ``outs`` as
    directory names and ``periods``, one element per period of the copy: the instant it began (its change_at, as the
    entry writes it, or the first record's start where FIRST_RECORD_RULE timed the period) or null where it has none
    (the chain's start); and ``outs_with_store_changes`` (directory names); and, only where an entry's recorded
    instant is not the first record on its copy, ``first_record_change_points``: the rule and, per such move, the OUT's
    directory name, the copy it moved to, the change point, the manifest store change's recorded instant and the
    entry's, each with its gap to the change point in seconds (a float).
    No count of records or attempts, share, run id, host, file hash or integer. OUTs that give one label different
    periods are told apart, one entry per OUT and endpoint ("<label> [<OUT>, <endpoint>]"), so none reads as
    another's. A call with a branch-only OUT (``is_branch_only``) adds ``branch_only_outs``, their directory names; a call
    without one has no such key."""
    groups: dict[Any, list[tuple[str, str, list]]] = defaultdict(list)
    for (out, ep), label in stores.labels.items():
        groups[label].append((Path(out).name, endpoint_label(ep), stores.period_starts.get((out, ep), [None])))
    copies = {}
    for label in sorted(groups):
        rows = sorted(groups[label], key=lambda r: (r[0], r[1]))
        if all(r[2] == rows[0][2] for r in rows):
            copies[label] = {"endpoints": sorted({r[1] for r in rows}), "outs": sorted({r[0] for r in rows}),
                             "periods": list(rows[0][2])}
        else:
            for name, endpoint, periods in rows:
                copies[f"{label} [{name}, {endpoint}]"] = {"endpoints": [endpoint], "outs": [name],
                                                          "periods": list(periods)}
    verdict: dict[str, Any] = {"ok": True, "copies": copies,
                               "outs_with_store_changes": sorted(Path(o).name for o in stores.changes)}
    only = sorted(Path(o).name for o, line in stores.timelines.items() if line.branch_only)
    if only:  # a branch-only OUT (is_branch_only) is named; a call without one has no such key
        verdict["branch_only_outs"] = only
    if stores.first_record_points:  # FIRST_RECORD_RULE: only where an entry's recorded instant is not the first record
        verdict["first_record_change_points"] = {
            "rule": FIRST_RECORD_RULE,
            "moves": [{"out": Path(p.out).name, "to": stores.labels.get((p.out, p.to_endpoint),
                                                                        endpoint_label(p.to_endpoint)),
                       **p.values()}
                      for p in sorted(stores.first_record_points.values(), key=lambda p: (Path(p.out).name, p.index))]}
    return verdict


def _check_stores_failed(exc: Exception) -> int:
    print(json.dumps({"ok": False, "error": type(exc).__name__}, sort_keys=True))
    print(f"check-stores: failed ({type(exc).__name__})", file=sys.stderr)
    return 3


# --- the calls ------------------------------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Call:
    batches: tuple[Path, ...]
    questions: tuple[Path, ...]
    graph: str
    split: str
    voided: tuple[Path, ...]
    partition_record: Path | None = None
    ledgers: tuple[Path, ...] = ()
    n_reps: int | None = None
    tier_if_missing: str | None = None
    cluster_map: Path | None = None  # a private map (a15)
    front_logs: tuple[Path, ...] = ()  # the e167 front's provenance log(s), for fronted records
    admissions: Path | None = None  # tests only: another admitted-fingerprints config than the committed one
    store_proofs: tuple[Path, ...] = ()  # the recorded check files STORE_PROOFS entries name (--store-proof)
    store_config: Path | None = None  # tests only: another store-proofs config than the committed one
    copy_classes: Path | None = None  # tests only: another copy-classes file than the committed one
    endpoint_copies: Path | None = None  # tests only: another endpoint-copies file than the committed one
    reference_pin: Path | None = None  # a privately pinned graph's reference pin (a15; --reference-pin)
    private_copy_classes: Path | None = None  # a privately pinned graph's own copy-class table (--private-copy-classes)
    private_copy_classes_sha256: str | None = None  # its sha256, pinned on the command line (--private-copy-classes-sha256)
    private_endpoint_copies: Path | None = None  # that graph's own endpoint-to-copy table (--private-endpoint-copies)
    private_endpoint_copies_sha256: str | None = None  # its sha256, pinned on the command line
    partitions: tuple[int, ...] | None = None  # --partitions: a union of only these partitions of the record
    partition_binding: Path | None = None  # --partition-binding: the DeepSeek dsh binding (the plans give the store copies)
    partition_plans: tuple[Path, ...] = ()  # --partition-plan: one plan per partition of the cited binding
    replay_routing: tuple[Path, ...] = ()  # --replay-routing: the classes of the copies the replay rows' records were replayed on
    replay_routing_pins: tuple[str, ...] = ()  # the sha256 pin of each (the command line always gives them)
    plain_partition_record_sha256: str | None = None  # producer record pinned on the command line
    plain_union_outs: str | None = None  # absolute OUT prefix and ordered manifest sha256 pins


def prepare(F: Frozen, call: Call, *, cluster_map: Any = None, map_record: dict | None = None,
            cfg: Any = None) -> tuple[Loaded, Union | None, dict]:
    """Everything before the resample matrix: the voided and partition refusals, the frozen loading (the union's stack
    check, or check_one_stack unchanged without a record), the graph and question set checks. ``cluster_map`` and
    ``map_record`` (tests) stand in for the graph's (``cluster_map`` None with a record: each question its own cluster)."""
    if call.partitions is not None and call.partition_record is None:
        raise Refused("--partitions names partitions of a partition record: it needs --partition-record")
    if (call.partition_binding is None) != (not call.partition_plans):
        raise Refused("--partition-binding and --partition-plan (one per partition) are given together or not at all")
    if call.partition_binding is not None and call.partition_record is None:
        raise Refused("--partition-binding binds the copies of a partition record: it needs --partition-record")
    if call.plain_partition_record_sha256 is not None and (call.partition_record is None or call.partitions is not None
                                                           or call.partition_binding is not None):
        raise Refused("a plain partition union needs --partition-record and all partitions, without a binding")
    if (call.plain_union_outs is None) != (call.plain_partition_record_sha256 is None):
        raise Refused("--plain-union-outs and --plain-partition-record-sha256 are required together")
    plain_prefix, manifest_pins = (parse_plain_union_outs(call.plain_union_outs, call.batches)
                                   if call.plain_union_outs is not None else (None, ()))
    if cfg is None:
        cfg, cfg_record = frozen_config(F)
    else:
        cfg_record = {"config": "given (a test)"}
    if map_record is None:
        cluster_map, map_record = cluster_map_for(F, cfg, call.graph, call.cluster_map)
    pin = reference_pin(map_record["store"], call.reference_pin)  # before any record is read, as the map
    private_classes = read_private_table(map_record["store"], "--private-copy-classes", call.private_copy_classes,
                                         call.private_copy_classes_sha256, lambda module, raw: module.loads(raw),
                                         lambda doc: doc["copies"])
    private_endpoints = read_private_table(map_record["store"], "--private-endpoint-copies",
                                           call.private_endpoint_copies, call.private_endpoint_copies_sha256,
                                           lambda module, raw: module.loads_endpoint_copies(raw),
                                           lambda doc: (e["replica"] for e in doc))
    outs = [Out.read(b, manifest_pins[i] if manifest_pins else None) for i, b in enumerate(call.batches)]
    if call.split == REGISTERED_SPLIT:  # its OUTs are matched to the voided record by their on-host paths too
        bare = [str(o.path) for o in outs if not (isinstance((o.identity or {}).get("out"), str) and o.identity["out"])]
        if bare:
            raise Refused(f"OUTs of the {REGISTERED_SPLIT} split without a deployment identity naming their on-host path "
                          f"(out): {bare}")
    voided, voided_files = read_voided(call.voided)
    check_not_voided(outs, voided)
    for b in call.batches:  # the frozen gate first, before any record is read (as analyse_batches orders it)
        F.gate.require_replay_check(b)
    union = None
    plain_paths: list[Path] = []
    if call.partition_record is not None:
        plain = call.plain_partition_record_sha256 is not None
        if plain:
            require_plain_private_graph(map_record["store"])
        record, record_sha = (read_plain_partition_record(call.partition_record, call.plain_partition_record_sha256)
                              if plain else read_record(call.partition_record))
        binding = (read_binding(call.partition_binding, call.partition_plans, record, record_sha)
                   if call.partition_binding is not None else None)
        if plain:
            if len(call.questions) != 1:
                raise Refused("a plain partition union needs exactly one registered questions file")
            qpath = call.questions[0]
            source_parts = record["partitions"]
            if record["questions_sha256"] != map_record["questions_sha256"]:
                raise Refused("the plain partition record is not for the graph's registered questions file")
            plain_paths, record = plain_partition_files(record, call.partition_record, qpath, call.split)
            record["graph"] = map_record["store"]
            require_plain_pinned_copies(record, map_record["store"], pin, private_classes, private_endpoints)
            outs_for_union = plain_union_outs(outs, record, source_parts, record_sha, plain_prefix)
        else:
            qpaths = {F.gate.select_questions(b, call.questions) for b in call.batches}
            if len(qpaths) != 1:
                raise Refused(f"a partition union runs on one questions file, not {sorted(map(str, qpaths))}")
            qpath = qpaths.pop()
            outs_for_union = outs
        all_rows = F.inputs.load_questions(qpath)
        design_ids = [qid for qid, q in all_rows.items() if q.get("split") == call.split]
        union = check_union(F, outs_for_union, record, record_sha, design_ids, map_record["store"], call.split,
                            sha256_file(qpath), call.partitions, binding,
                            identity_note=VIEW_STORE_CHECK_NOTE if plain else None)
        if union.subset is not None:
            check_bases_whole(all_rows, call.split, union)
        stack_check = union_stack_check(F)
    else:
        parts = [str(o.path) for o in outs if o.partition is not None]
        if parts:
            raise Refused(f"partition OUTs are analysed only in their union (--partition-record), never alone: {parts}")
        def stack_check(batches: list[tuple]) -> None:  # the frozen per-arm path's own check, unchanged
            F.inputs.check_one_stack(batches, require_route=False)
    keep = None
    if union is not None and union.subset is not None:  # the design is the named partitions' questions alone
        def keep(qid: str) -> bool:
            return partition_of(qid, union.k, union.weights) in union.subset
    loaded = load_inputs(F, call.batches, (*call.questions, *plain_paths), split=call.split, n_reps=call.n_reps, ledgers=call.ledgers,
                         tier_if_missing=call.tier_if_missing, stack_check=stack_check, design_keep=keep)
    check_replay_coverage(loaded)
    if loaded.graphs != {map_record["store"]}:
        raise Refused(f"the records' stores are {sorted(loaded.graphs)}, not {call.graph}'s {map_record['store']} alone "
                      f"(one map per call)")
    expected_questions = ({p["file_sha256"] for p in source_parts.values()} if plain_paths
                          else {map_record["questions_sha256"]})
    if loaded.questions_sha256 != expected_questions:
        raise Refused(f"the batches ran on questions {sorted(loaded.questions_sha256)}, not {call.graph}'s question set "
                      f"{map_record['questions_sha256']} (the one its cluster map was built from)")
    if cluster_map is None:  # SIB's rule: each question of the index its own cluster
        cluster_map = F.prereg.ClusterMap.singletons(loaded.index.questions)
    if union is not None:
        check_union_ledger(loaded.ledger, union)
    stores = check_stores(load_attribution(), outs, map_record["store"], call.store_proofs, call.store_config, pin,
                          call.copy_classes, call.endpoint_copies, private_classes, private_endpoints,
                          plain_union=call.plain_partition_record_sha256 is not None)
    context = {"cfg": cfg, "cluster_map": cluster_map, "map_record": map_record, "config": cfg_record, "stores": stores,
               "replay_routing": read_replay_routing(call.replay_routing, call.replay_routing_pins),
               "voided": voided_files,
               "outs": [{"path": str(o.path), "identity_out": (o.identity or {}).get("out"),
                         "partition": None if union is None else union.index_of[str(o.path)]} for o in outs]}
    if union is not None:
        context["partition"] = {"record": str(call.partition_record), "record_sha256": union.record_sha256,
                                "schema": union.record["schema"], "k": union.k, "graph": union.record["graph"],
                                "stack_check": "one stack per (graph, model) from the partitions' deployment "
                                               "identities, each store copy identity-checked (in place of "
                                               "check_one_stack's one-URL/one-endpoint refusals; entries 150, 184)"}
        if call.plain_partition_record_sha256 is not None:  # design ruling (b), 25 Oct: said in the results, here alone
            context["partition"]["stack_check"] = ("one stack per (graph, model) from the partitions' deployment "
                                                   "identities (in place of check_one_stack's one-URL/one-endpoint "
                                                   "refusals; entries 150, 184)")
            context["partition"]["copy_identity"] = PLAIN_UNION_COPY_IDENTITY
            context["partition"]["endpoint_proofs"] = PLAIN_UNION_ENDPOINT_PROOFS
            context["partition"]["out_manifest_sha256"] = list(manifest_pins)  # the pinned manifests, in partition order
        if union.binding is not None:
            context["partition"]["binding"] = union.binding
        if union.subset is not None:
            context["partition"]["subset"] = subset_block(union, loaded, cluster_map, call.split)
    return loaded, union, context


def run(F: Frozen, call: Call, *, K_clusters: np.ndarray | None = None, cluster_map: Any = None,
        map_record: dict | None = None, cfg: Any = None) -> dict[str, Any]:
    """One exploratory call: the per-arm numbers with every interval from the cluster bootstrap over the call's (or the
    union's) question index."""
    loaded, _union, context = prepare(F, call, cluster_map=cluster_map, map_record=map_record, cfg=cfg)
    K, clusters = cluster_counts(F, context["cluster_map"], loaded.index, context["cfg"], K_clusters)
    deployments = attribute(F, call.batches, call.front_logs, call.admissions)  # over the union for a partition union
    result = per_arm(F, loaded, K, resampling=RESAMPLING, status=STATUS)
    result["online_vs_replay"] = online_vs_replay(loaded.ledger)
    result["deployments"] = deployment_section(F, loaded, K, deployments)
    result["stores"] = store_section(F, loaded, deployments, context["stores"])
    result["stores"]["replay_class_check"] = replay_class_check(loaded, deployments, context["stores"], context["replay_routing"])
    result["exploratory"] = {
        "ruling": RULING, "graph": call.graph, "cluster_map": context["map_record"],
        "clusters": clusters.summary(), "resample_matrix": "given over the clusters (a test)" if K_clusters is not None
        else f"ClusterMap.resampling: {N_BOOT} unstratified draws of the clusters, seed {SEED}",
        "config": context["config"], "code": F.identity, "voided_records": context["voided"], "outs": context["outs"],
        "partition_union": context.get("partition")}
    return result


def parity(F: Frozen, call: Call, **kw: Any) -> dict[str, Any]:
    """Dev only: this script's loading and per-arm assembly with the frozen stratified matrix against the frozen per-arm
    path (pipeline.analyse_batches without a config) on the same inputs, section by section; True where identical."""
    loaded, union, _context = prepare(F, call, **kw)
    if union is not None and union.subset is not None:
        raise Refused("parity compares with the frozen per-arm path over the whole question set, which has no twin for "
                      "a strict subset of the record's partitions")
    K = F.stats.resample_matrix(loaded.index, N_BOOT, seed=SEED)  # the frozen per-arm matrix (stratified)
    mine = per_arm(F, loaded, K, resampling="", status="")
    frozen = F.pipeline.analyse_batches(list(call.batches), list(call.questions), split=call.split, n_reps=call.n_reps,
                                        ledgers=list(call.ledgers), checked_call=F.checked_call,
                                        tier_if_missing=call.tier_if_missing, seed=SEED, n_boot=N_BOOT)
    return compare_sections(frozen, mine, skip_settings=("resampling", "status"))


SECTIONS = ("arms", "real_output_ledger", "refusals", "scorer_changes", "model_settings", "output_cap_hits",
            "confirmatory")


def _canon(x: Any) -> str:
    return json.dumps(x, sort_keys=True, allow_nan=True)


def compare_sections(a: Mapping[str, Any], b: Mapping[str, Any], *,
                     skip_settings: Sequence[str] = ()) -> dict[str, Any]:
    """Section by section, whether two results are identical (booleans and key names only, never a number)."""
    out: dict[str, Any] = {s: _canon(a.get(s)) == _canon(b.get(s)) for s in SECTIONS}
    sa, sb = dict(a.get("settings") or {}), dict(b.get("settings") or {})
    keys = sorted((set(sa) | set(sb)) - set(skip_settings))
    out["settings_differing"] = [k for k in keys if _canon(sa.get(k)) != _canon(sb.get(k))]
    if not out["arms"]:
        arms_a, arms_b = a.get("arms") or {}, b.get("arms") or {}
        out["arms_differing"] = sorted(k for k in set(arms_a) | set(arms_b)
                                       if _canon(arms_a.get(k)) != _canon(arms_b.get(k)))
    return out


# --- command line ---------------------------------------------------------------------------------------------------


def _call(args: argparse.Namespace) -> Call:
    return Call(tuple(args.batch), tuple(args.questions), args.graph, args.split, tuple(args.voided),
                args.partition_record, tuple(args.ledger), args.n_reps, args.tier_if_missing, args.cluster_map,
                tuple(args.front_log), store_proofs=tuple(args.store_proof), reference_pin=args.reference_pin,
                partitions=args.partitions, partition_binding=args.partition_binding,
                partition_plans=tuple(args.partition_plan), private_copy_classes=args.private_copy_classes,
                private_copy_classes_sha256=args.private_copy_classes_sha256,
                private_endpoint_copies=args.private_endpoint_copies,
                private_endpoint_copies_sha256=args.private_endpoint_copies_sha256,
                replay_routing=tuple(p for p, _ in args.replay_routing),
                replay_routing_pins=tuple(h for _, h in args.replay_routing),
                plain_partition_record_sha256=getattr(args, "plain_partition_record_sha256", None),
                plain_union_outs=getattr(args, "plain_union_outs", None))


def _routing_pin(text: str) -> tuple[Path, str]:
    """``PATH:SHA256`` of a replay routing file: the path and the sha256 the checker's output had."""
    path, sep, pin = text.rpartition(":")
    if not sep or not path or re.fullmatch(r"[0-9a-f]{64}", pin) is None:
        raise argparse.ArgumentTypeError("a replay routing is given as PATH:SHA256")
    return Path(path), pin


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("clusters", help="build and hash Rhea 142's and opaque Rhea 142's cluster maps")
    c.add_argument("--code-root", type=Path, required=True)
    c.add_argument("--out-dir", type=Path, required=True)
    for name in ("analyse", "parity"):
        s = sub.add_parser(name)
        s.add_argument("--code-root", type=Path, required=True, help="a deploy of the frozen analysis")
        s.add_argument("--graph", required=True)
        s.add_argument("--split", required=True)
        s.add_argument("--batch", type=Path, action="append", required=True, help="a runner OUT directory")
        s.add_argument("--questions", type=Path, action="append", required=True)
        s.add_argument("--voided", type=Path, action="append", required=True, help="the voided-run record(s)")
        s.add_argument("--partition-record", type=Path, help="a partition union's record (entries 179, 180, 184)")
        s.add_argument("--plain-partition-record-sha256", help="sha256 pin for a privately pinned graph's plain-run partition record")
        s.add_argument("--plain-union-outs", metavar="PREFIX:SHA0,SHA1,...",
                       help="absolute OUT prefix and manifest.json sha256 pins in partition index order")
        s.add_argument("--partitions", type=parse_partitions, metavar="I,J,...",
                       help="analyse only the union of these partitions of the record (a strict subset of at least "
                            "two; the record is still checked in full; needs --partition-record)")
        s.add_argument("--partition-binding", type=Path,
                       help="a partition binding (DeepSeek's dsh union): its plans, not the record, name each OUT's store copy; "
                            "needs --partition-record and one --partition-plan per partition of the binding (also with "
                            "--partitions, a proper subset of at least two)")
        s.add_argument("--partition-plan", type=Path, action="append", default=[],
                       help="a plan of the cited binding (one per partition of the binding, a --partitions subset included)")
        s.add_argument("--ledger", type=Path, action="append", default=[], help="checker-ledger files")
        s.add_argument("--replay-routing", type=_routing_pin, action="append", default=[], metavar="PATH:SHA256",
                       help="the replay routing file of an OUT (driver/replay_matrices.py check): the class of the copy each "
                            "replay row's record was replayed on, checked against its record's own copy's class")
        s.add_argument("--n-reps", type=int)
        s.add_argument("--tier-if-missing")
        s.add_argument("--cluster-map", type=Path, help="a private map (a15), checked by its pinned sha256")
        s.add_argument("--front-log", type=Path, action="append", default=[],
                       help="the e167 front's provenance log as it is now (for fronted records)")
        s.add_argument("--store-proof", type=Path, action="append", default=[],
                       help=f"a store copy's recorded check file (storecheck or known answers) a {STORE_PROOFS} entry "
                            f"names")
        s.add_argument("--reference-pin", type=Path,
                       help="a privately pinned graph's reference pin (a15), checked by its pinned sha256")
        s.add_argument("--private-copy-classes", type=Path,
                       help="a privately pinned graph's own copy-class table (a15): outside git, mode 600, merged with "
                            "copy-classes.json (a copy in both is refused unless the entries are identical); needs "
                            "--private-copy-classes-sha256")
        s.add_argument("--private-copy-classes-sha256",
                       help="the sha256 the private copy-class table must have (checked before it is read as a table)")
        s.add_argument("--private-endpoint-copies", type=Path,
                       help="a privately pinned graph's own endpoint-to-copy table (a15): the e189-endpoint-copies/1 "
                            "shape, outside git, mode 600, merged with endpoint-copies.json (an endpoint in both is "
                            "refused); gives its OUTs' endpoints the replica the private copy classes describe; needs "
                            "--private-endpoint-copies-sha256")
        s.add_argument("--private-endpoint-copies-sha256",
                       help="the sha256 the private endpoint-copies table must have (checked before it is read)")
        if name == "analyse":
            s.add_argument("--out", type=Path, required=True)
    t = sub.add_parser("cluster-map", help="one graph's map from the frozen generator's anchors (corporate, a15)")
    t.add_argument("--code-root", type=Path, required=True)
    t.add_argument("--graph", required=True, choices=TEMPLATE_MAP_GRAPHS)
    t.add_argument("--questions", type=Path, required=True)
    t.add_argument("--manifest", type=Path, required=True, help="the generator's manifest of that question set")
    t.add_argument("--out", type=Path, required=True)
    t.add_argument("--record", type=Path, required=True, help="the build record (hashes and cluster counts only)")
    t.add_argument("--private", action="store_true", help="a15: mode 600, outside git")
    m = sub.add_parser("compare", help="two results files, section by section (booleans only)")
    m.add_argument("a", type=Path)
    m.add_argument("b", type=Path)
    w = sub.add_parser("store-timeline", help="each OUT's store changes against its records' own endpoints and times "
                                              "(one JSON object per OUT; no counts)")
    w.add_argument("--code-root", type=Path, required=True)
    w.add_argument("outs", type=Path, nargs="+", metavar="OUT", help="a runner OUT directory")
    k = sub.add_parser("check-stores", help="the call's store check alone, its verdict as one JSON object (no counts)")
    k.add_argument("--code-root", type=Path, required=True, help="a deploy of the frozen analysis")
    k.add_argument("--graph", required=True)
    k.add_argument("--batch", type=Path, action="append", required=True, help="a runner OUT directory")
    k.add_argument("--store-proof", type=Path, action="append", default=[],
                   help=f"a store copy's recorded check file (storecheck or known answers) a {STORE_PROOFS} entry names")
    args = ap.parse_args(argv)
    private_out = None
    if args.cmd in ("analyse", "parity") and any(option is not None for option in (
            args.private_copy_classes, args.private_copy_classes_sha256, args.private_endpoint_copies,
            args.private_endpoint_copies_sha256)):  # (an empty sha256 is a given option too)
        try:  # before anything is read: a call that uses a private table writes only into a private tree
            if args.cmd == "analyse":
                private_out = check_private_out(args.out)
        except Refused as exc:
            print(f"refused: {exc}", file=sys.stderr)
            return 2
    if args.cmd == "store-timeline":  # before the generic refusal handling: here exit 2 means STOP, not refused
        return store_timeline_main(args.code_root, args.outs)
    if args.cmd == "check-stores":  # likewise: its own exit codes, and its verdict on stdout
        return check_stores_main(args.code_root, args.graph, args.batch, args.store_proof)
    if args.cmd == "compare":
        print(json.dumps(compare_sections(read_json(args.a), read_json(args.b), skip_settings=("inputs", "code_digest")),
                         indent=1, sort_keys=True))
        return 0
    try:
        F = load_frozen(args.code_root)
    except Refused as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2
    try:
        if args.cmd == "clusters":
            record = build_clusters(F, args.out_dir)
            print(json.dumps({g: {k: m[k] for k in ("sha256", "all", "test")} for g, m in record["maps"].items()},
                             indent=1, sort_keys=True))
            return 0
        if args.cmd == "cluster-map":
            record = build_template_map(F, args.graph, args.questions, args.manifest, args.out, args.record,
                                        args.private)
            print(json.dumps({k: record[k] for k in ("graph", "map_sha256", "all", "test")}, indent=1, sort_keys=True))
            return 0
        if args.cmd == "parity":
            print(json.dumps(parity(F, _call(args)), indent=1, sort_keys=True))
            return 0
        result = run(F, _call(args))
    except F.refusals as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2
    text = json.dumps(result, sort_keys=True, indent=1, ensure_ascii=False, allow_nan=False) + "\n"
    if private_out is not None:  # into the directory the guard checked, by descriptor: no link can redirect the write
        try:
            write_in_private_dir(private_out, "results.json", text, create=True)
        except Refused as exc:
            print(f"refused: {exc}", file=sys.stderr)
            return 2
        except Exception as exc:  # anything else (a path would sit in its text): the class only, as check-stores does
            print(f"results of a call with a private table: not written ({type(exc).__name__})", file=sys.stderr)
            return 3
        path = Path("results.json")  # (the success line names the file, not where it is)
    else:
        args.out.mkdir(parents=True, exist_ok=True)
        path = args.out / "results.json"
        path.write_text(text, encoding="utf-8")
    ex = result["exploratory"]
    dep = result["deployments"]
    union = ex["partition_union"]
    of = f"{len(union['subset']['partitions'])} of " if union and "subset" in union else ""
    print(f"{path}: {len(result['arms'])} arms over {result['settings']['n_questions']} questions in "
          f"{ex['clusters']['clusters']} clusters ({ex['graph']}"
          + (f", a union of {of}{union['k']} partitions" if union else "")
          + f"); {dep['pairs_compared']} arm pairs compared, {len(dep['triggered_pairs'])} over 10 points"
          + ("" if private_out is not None else f"; {len(result['stores']['copies'])} store copies"))
    # (a private call's line leaves the store-copy count out: with its tables it is a count of private copies)
    return 0


if __name__ == "__main__":
    sys.exit(main())
