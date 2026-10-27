"""Tier 2 results part tables from design_log look cells (results.json per graph x model), section-major.
Usage: t2gen.py <cell-dir> [<cell-dir> ...]  (each holds results.json; one graph per call)"""
import json, sys
from decimal import Decimal, ROUND_HALF_UP

MN = {'claude-sonnet-5-5': 'Sonnet', 'claude-haiku-5-5': 'Haiku', 'Qwen/Qwen3.8-27B': 'Qwen3.8', 'qwen3.8-27b': 'Qwen3.8',
      'zai-org/GLM-5.3-Flash': 'GLM', 'google/gemma-4-31B': 'Gemma', 'deepseek-ai/DeepSeek-V4-Flash-0731': 'DeepSeek',
      'Qwen/Qwen3-Coder-30B-A3B-Instruct': 'Qwen3-Coder'}
ORDER = list(dict.fromkeys(MN.values()))
SRC = {'curated': 'curated', 'census': 'census (threshold 5)', 'census-t5': 'census (threshold 5)', 'census-t2': 'census (threshold 2)', 'census-t20': 'census (threshold 20)', 'generated-1': 'generated draft 1', 'generated-2': 'generated draft 2',
       'generated-3': 'generated draft 3', 'generic': 'generic re-check', 'union': 'union (curated or census)'}
SORD = list(SRC)
PROMPT = ['none', 'curated', 'census', 'generated-1']
PNAME = {'none': 'no ontology', 'curated': 'curated', 'census': 'census', 'generated-1': 'generated draft 1'}

def n(x):
    v = (Decimal(repr(x)).quantize(Decimal('1e-12')) * 100).quantize(Decimal('0.1'), rounding=ROUND_HALF_UP)
    if v == 0: v = Decimal('0.0')
    return f"{v:.1f}".replace('-', '−')
def f(d, ci=True):
    if d is None or d.get('estimate') is None: return 'n/a'
    s = n(d['estimate'])
    if ci and d.get('ci'): s += f" ({n(d['ci'][0])}, {n(d['ci'][1])})"
    return s
def diff(a, b):
    if a is None or b is None or a.get('estimate') is None or b.get('estimate') is None: return 'n/a'
    return n(a['estimate'] - b['estimate'])

def load(path):
    r = json.load(open(f"{path}/results.json"))
    k0 = next(iter(r['arms'].values()))['key']
    return {'r': r, 'g': k0['graph'], 'm': k0['model'], 'name': MN.get(k0['model'], k0['model']), 'dir': path}

def arm(c, cohort, use, src):
    return c['r']['arms'].get(f"{c['g']}|{c['m']}|{cohort}|{use}|{src}")

def render(cells):
    cells = sorted(cells, key=lambda c: ORDER.index(c['name']) if c['name'] in ORDER else 99)
    out = []; p = out.append
    p("## 2. H1, context: the ontology as prompt text\n")
    p("Per model: accuracy and the other per-question means (%), accuracy with its 95% interval; the difference from no ontology in points, from unrounded values, without an interval.\n")
    p("| Model | Prompt | Attempts | Accuracy | Difference from no ontology | Mean F1 | Output rate | Missing rate | Silent failure |")
    p("|---|---|---|---|---|---|---|---|---|")
    for c in cells:
        nat = arm(c, 'none', 'prompt', '-')
        for pr in PROMPT:
            a = arm(c, pr, 'prompt', '-')
            if not a: continue
            mm = a['metrics']
            p(f"| {c['name']} | {PNAME[pr]} | {a['counts']['N_att']:,} | {f(mm['accuracy'])} | {'—' if pr == 'none' else diff(mm['accuracy'], nat['metrics']['accuracy'])} | {f(mm['mean_f1'], False)} | {f(mm['output_rate'], False)} | {f(mm['missing_rate'], False)} | {f(mm['silent_failure'], False)} |")
    p("")
    def sord(s):
        if s in ('census-t2', 'census-t20'): return 1.5 + ['census-t2', 'census-t20'].index(s) / 10
        return SORD.index(s) if s in SORD else 99
    for cohort, title, rtitle in (('none', "## 3. Check time on the no-ontology outputs: every source's checks", "**Retries on the no-ontology outputs** (the retry cells that ran):"),
                          ('census', "## 4. Checks on the census-prompted outputs, and the deployment row", "**Retries on the census-prompted outputs** (the deployment row: the curated and generated checks with one retry):"),
                          ('curated', "## 5. Checks on the curated-prompted outputs", None),
                          ('generated-1', "## 6. Checks on the generated-prompted outputs", None)):
        rows = [(c, x) for c in cells for x in c['r']['real_output_ledger']['rows'] if x['cohort'] == cohort]
        if not rows: continue
        p(title + "\n")
        p("**Detection,** per model and source, from the offline replay of each source's checks on the cohort's initial outputs (no retry): catches and false flags per attempt (%, with 95% intervals); the same counting only outputs that return rows, still per all attempts (non-empty; outputs with no recorded row count are counted as not returning rows); the unknown rate (unknown verdicts over outputs, %); and recall, false-flag rate and precision per output (%). Unknowns and vacuous passes count as not flagged.\n")
        p("| Model | Source | Catches | False flags | Catches, non-empty | False flags, non-empty | Unknown rate | Recall | False-flag rate | Precision |")
        p("|---|---|---|---|---|---|---|---|---|---|")
        rrows = []
        for c in cells:
            by = {}
            for x in c['r']['real_output_ledger']['rows']:
                if x['cohort'] == cohort: by.setdefault(x['source'], {})[x['use']] = x
            for s in sorted(by, key=sord):
                d = by[s]
                if 'checker' in d: rrows.append((c, s, d['checker']))
                if s == 'generic': continue
                det = d.get('replay') or d.get('checker')
                da = arm(c, cohort, det['use'], s); dm = da['metrics'] if da else {}
                pa = det['per_attempt']; ne = det.get('non_empty') or {}
                nec = f(ne.get('catches'), False) if ne.get('estimable') else 'n/a'
                nef = f(ne.get('false_flags'), False) if ne.get('estimable') else 'n/a'
                p(f"| {c['name']} | {SRC.get(s, s)} | {f(pa['catches'])} | {f(pa['false_flags'])} | {nec} | {nef} | {f(det['unknown_rate'], False)} | {f(dm.get('recall'), False)} | {f(dm.get('false_flag_rate'), False)} | {f(dm.get('flag_precision'), False)} |")
        p("")
        if rtitle and rrows:
            p(rtitle + " per model and source: outputs retried (count); fixes (wrong, flagged, then right) and corruptions (right, flagged, then wrong) per attempt (%, with 95% intervals); net change in points (with its 95% interval); accuracy after the retry (%). The generic re-check retries the outputs the curated checks flag, with a generic message.\n")
            p("| Model | Source | Retried | Fixes | Corruptions | Net change | Accuracy after the retry |")
            p("|---|---|---|---|---|---|---|")
            for c, s, x in rrows:
                oa = arm(c, cohort, 'checker', s); om = oa['metrics']; oc = oa['counts']; pa = x['per_attempt']
                assert oc['retried'] == x['counts']['retried']
                p(f"| {c['name']} | {SRC.get(s, s)} | {oc['retried']} | {f(pa['fixes'])} | {f(pa['corruptions'])} | {f(om.get('net_change'))} | {f(om.get('accuracy_final'), False)} |")
            p("")
    p("## 9. The offline replay against the online checks\n")
    p("Per model: where a source has both an online retry cell and an offline replay on the same outputs, the number of outputs whose verdict (flag, pass, unknown) agrees and disagrees (the generic re-check has no replay twin and isn't compared).\n")
    p("| Model | Online arms compared | Agree | Disagree |")
    p("|---|---|---|---|")
    for c in cells:
        ov = {k: v for k, v in c['r'].get('online_vs_replay', {}).get('arms', {}).items() if v['agree'] + v['disagree'] > 0}
        ag = sum(v['agree'] for v in ov.values()); di = sum(v['disagree'] for v in ov.values())
        p(f"| {c['name']} | {len(ov)} | {ag:,} | {di:,} |")
    p("")
    return "\n".join(out)

if __name__ == '__main__':
    print(render([load(d) for d in sys.argv[1:]]))
