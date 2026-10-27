"""Part 13 tables: UniProt's further retry cells (the design log), Sonnet and Qwen3.8.
Usage: p13gen.py <look dir> <registered results.json>"""
import json, sys
from decimal import Decimal, ROUND_HALF_UP
LOOK = sys.argv[1]; REG = json.load(open(sys.argv[2]))
G = 'uniprot-sp-2026_03'
CELLS = [('Sonnet', 'claude-sonnet-5-5', 'uniprot-retry-claude-sonnet-5-5'), ('Qwen3.8', 'Qwen/Qwen3.8-27B', 'uniprot-retry-qwen3.8-27b')]
SRC = {'census': 'census', 'generated-1': 'generated draft 1', 'generated-2': 'generated draft 2', 'generated-3': 'generated draft 3',
       'union': 'union (curated or census)', 'curated': 'curated'}
CO = {'none': (['census', 'generated-1', 'generated-2', 'generated-3', 'union'], 'no ontology'),
      'census': (['curated', 'generated-1', 'generated-2', 'generated-3'], 'census prompt')}


def n(x):
    v = (Decimal(repr(x)).quantize(Decimal('1e-12')) * 100).quantize(Decimal('0.1'), rounding=ROUND_HALF_UP)
    if v == 0: v = Decimal('0.0')
    return f"{v:.1f}".replace('-', '−')
def f(d, ci=True):
    if d is None or d.get('estimate') is None: return 'n/a'
    s = n(d['estimate'])
    if ci and d.get('ci'): s += f" ({n(d['ci'][0])}, {n(d['ci'][1])})"
    return s

out = []; p = out.append
looks = {nm: json.load(open(f"{LOOK}/{d}/results.json")) for nm, m, d in CELLS}
for cohort, (srcs, cname) in CO.items():
    title = {"none": "## 2. The no-ontology outputs: the further sources, with one retry",
             "census": "## 3. The deployment row: the census-prompted outputs, curated and generated checks, with one retry"}[cohort]
    p(title + "\n")
    p("Per model and source: recall, false-flag rate and precision per output (%); outputs (attempts with an output, of 630), and the outputs retried, fixed and corrupted (counts); net change in points, with its 95% interval; accuracy before the retry (the cohort's own accuracy) and after it (%).\n")
    p("| Model | Source | Recall | False-flag rate | Precision | Outputs | Retried | Fixed | Corrupted | Net change | Accuracy before | Accuracy after |")
    p("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for nm, m, d in CELLS:
        L = looks[nm]
        before = L['arms'][f"{G}|{m}|{cohort}|prompt|-"]['metrics']['accuracy']
        rb = REG['arms'][f"{G}|{m}|{cohort}|prompt|-"]['metrics']['accuracy']
        assert before['estimate'] == rb['estimate'], (nm, cohort)
        for s in srcs:
            a = L['arms'][f"{G}|{m}|{cohort}|checker|{s}"]; mm = a['metrics']; c = a['counts']
            r = REG['arms'].get(f"{G}|{m}|{cohort}|replay|{s}")
            assert (r is not None) == (cohort == 'none'), (nm, cohort, s)
            if r is not None:
                for k in ('recall', 'false_flag_rate', 'flag_precision'):
                    assert mm[k]['estimate'] == r['metrics'][k]['estimate'], (nm, cohort, s, k)
            assert c['N_att'] == 630
            p(f"| {nm} | {SRC[s]} | {f(mm['recall'], False)} | {f(mm['false_flag_rate'], False)} | {f(mm['flag_precision'], False)} | {c['N_out']} | {c['retried']} | {c['fixed']} | {c['corrupted']} | {f(mm['net_change'])} | {f(before, False)} | {f(mm['accuracy_final'], False)} |")
    p("")

p("## 4. The ledger on real outputs, per attempt\n")
p("Per model, cohort and source, per attempt over the 630 attempts (%, with 95% intervals for catches and false flags): catches (wrong and flagged), false flags (correct and flagged), the same counting only non-empty outputs, fixes (wrong, flagged, then right after the retry) and corruptions (right, flagged, then wrong after the retry); unknown (% of outputs). Unknowns and vacuous passes count as not flagged.\n")
ne_lines = []
for nm, m, d in CELLS:
    for cohort, (srcs, cname) in CO.items():
        outs = {json.dumps(x['non_empty']['outputs'], sort_keys=True) for x in looks[nm]['real_output_ledger']['rows'] if x['cohort'] == cohort}
        assert len(outs) == 1
        o = json.loads(outs.pop())
        ne_lines.append(f"{nm}, {cname}: {o['returns_rows']} return rows, {o['empty']} empty, {o['no_row_count']} without a recorded row count")
p("The non-empty columns count only events on outputs that return rows, still per all 630 attempts (outputs without a recorded row count are left out): " + "; ".join(ne_lines) + ".\n")
p("| Model | Cohort | Source | Catches | False flags | Catches, non-empty | False flags, non-empty | Fixes | Corruptions | Unknown |")
p("|---|---|---|---|---|---|---|---|---|---|")
for nm, m, d in CELLS:
    rows = {(x['cohort'], x['source']): x for x in looks[nm]['real_output_ledger']['rows']}
    for cohort, (srcs, cname) in CO.items():
        for s in srcs:
            x = rows[(cohort, s)]; pa = x['per_attempt']; ne = x['non_empty']
            assert x['counts']['N_att'] == 630 and ne['estimable']
            p(f"| {nm} | {cname} | {SRC[s]} | {f(pa['catches'])} | {f(pa['false_flags'])} | {f(ne['catches'], False)} | {f(ne['false_flags'], False)} | {f(pa['fixes'], False)} | {f(pa['corruptions'], False)} | {f(x['unknown_rate'], False)} |")
p("")
print("\n".join(out))
