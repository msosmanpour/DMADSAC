#!/usr/bin/env python3
"""Public-safe read-only D330 anchor memory-model sensitivity for Article 2.

This script re-scores existing question-level standardized-assessment CSVs only.
It does NOT run the scheduler, train a policy, migrate/write a database, or modify
source data.

Required inputs are three user-supplied roots:
  --dsac-seed14040504-root
  --dsac-multiseed-root
  --mau-root

The script verifies the locked SHA256 of each of the 10 question-level input
files before analysis.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, math, os, sys, zipfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path

SEEDS = [14040504, 1123252065, 1529594671, 997781685, 1719214619]

EXPECTED = {
    ("DSAC",14040504): "365683507417C6C25B1464B18BA48CBFF646A9D7275EA575E5431A25BBFE21EE",
    ("DSAC",1123252065): "B74A069612D7740617A25B76D5687109CC98BBD067D20D5B7AA8B2DEE416C8B8",
    ("DSAC",1529594671): "7FA3B28517C25589A5977604E6C4BA46548F134A4B8417D3D968FCD966B745B8",
    ("DSAC",997781685): "C34607977F629286B583D07F636FB0AFD86097AEC2B06F490990A9D7982FE926",
    ("DSAC",1719214619): "585910552D37C77256455F37EAC9AC0021B87016B2EBC12077B90A59220280F7",
    ("MAU",14040504): "0E057B26203B77463A1845A746A043C50BD3EAECAAF0945A98839A5EFA2D0DD2",
    ("MAU",1123252065): "CFA4D97E7EF04DFD69D8CBB3C11D99CC73E9F72A622A8BBB96F9FB74918CA14A",
    ("MAU",1529594671): "F166B14F385F48B35510E57F2D2F3D3CC3D86B42F0EAE7DDA89F32515802F0ED",
    ("MAU",997781685): "760C0041EBD2910212C2B9BA022CDA94E44D3332A304F5F07496BAA56AAEC512",
    ("MAU",1719214619): "8FA4B91E6EC8EF1CBBE71DFF845CC5BF653A396547829798893ECE6EBE70F788",
}

SCENARIOS = [
 dict(name="LOCKED", hscale=1.0, shape=1.0, ew=(.50,.20,.20,.10), rh=.30, vh=.25),
 dict(name="SHORT_RETENTION_0p5x", hscale=.5, shape=1.0, ew=(.50,.20,.20,.10), rh=.30, vh=.25),
 dict(name="LONG_RETENTION_2x", hscale=2.0, shape=1.0, ew=(.50,.20,.20,.10), rh=.30, vh=.25),
 dict(name="LONG_TAIL_SHAPE_0p75", hscale=1.0, shape=.75, ew=(.50,.20,.20,.10), rh=.30, vh=.25),
 dict(name="STEEP_TAIL_SHAPE_1p25", hscale=1.0, shape=1.25, ew=(.50,.20,.20,.10), rh=.30, vh=.25),
 dict(name="LOW_REPETITION_REVIEW", hscale=1.0, shape=1.0, ew=(.65,.20,.10,.05), rh=.15, vh=.125),
 dict(name="HIGH_REPETITION_REVIEW", hscale=1.0, shape=1.0, ew=(.35,.20,.30,.15), rh=.45, vh=.375),
]

def clamp(x,a,b): return max(a,min(b,x))

def sha256(path: Path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):
            h.update(b)
    return h.hexdigest().upper()

def recalc(row, s):
    n=int(float(row['exposure_count']))
    if n <= 0: return 0.0
    q=float(row['source_quality'])
    days=max(0.0,float(row['days_since_last']))
    dd=int(float(row['distinct_study_dates']))
    rv=int(float(row['review_exposure_count']))
    rep=max(0,n-1)
    distinct=max(0,dd-1)
    rep_signal=clamp(math.log1p(rep)/math.log(6.0),0,1)
    review_signal=clamp(max(distinct,rv)/3.0,0,1)
    wq,wi,wr,wv=s['ew']
    enc=clamp(wq*q+wi+wr*rep_signal+wv*review_signal,0,1)
    qf=.70+.60*q
    rf=1+s['rh']*math.log1p(rep)
    vf=1+s['vh']*max(distinct,rv)
    stress=.875
    fatigue=.925
    base=2.0*s['hscale']
    lo=.75*s['hscale']
    hi=14.0*s['hscale']
    hl=clamp(base*qf*rf*vf*stress*fatigue,lo,hi)
    retention=2.0**(-((days/hl)**s['shape']))
    return 100.0*clamp(enc*retention,0,1)

def build_files(args):
    a = Path(args.dsac_seed14040504_root)
    m = Path(args.dsac_multiseed_root)
    h = Path(args.mau_root)
    files = []
    files.append((a / "STANDARDIZED_EVALUATION_FINAL330" / "standardized_exam_v1_question_level.csv",
                  EXPECTED[("DSAC",14040504)], 14040504, "DSAC"))
    for seed in SEEDS[1:]:
        files.append((m / f"SEED_{seed}" / "STANDARDIZED_EVALUATION_FINAL330" /
                      "standardized_exam_v1_question_level.csv",
                      EXPECTED[("DSAC",seed)], seed, "DSAC"))
    for seed in SEEDS:
        files.append((h / f"SEED_{seed}" / "STANDARDIZED_EVALUATION_MAU_VS_STATIC" /
                      "standardized_exam_v1_question_level.csv",
                      EXPECTED[("MAU",seed)], seed, "MAU"))
    return files

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dsac-seed14040504-root", required=True)
    ap.add_argument("--dsac-multiseed-root", required=True)
    ap.add_argument("--mau-root", required=True)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    if args.out_dir:
        out = Path(args.out_dir)
        out.mkdir(parents=True, exist_ok=False)
    else:
        desktop=Path(os.environ.get('USERPROFILE', str(Path.home()))) / 'Desktop'
        stamp=datetime.now().strftime('%Y%m%d_%H%M%S')
        out=desktop/f'ARTICLE2_MEMORY_SENSITIVITY_{stamp}'
        out.mkdir(parents=True,exist_ok=False)

    files = build_files(args)
    sums=defaultdict(lambda:[0.0,0])
    verification=[]
    inputs=[]

    for p, expected, seed, family in files:
        if not p.is_file():
            raise FileNotFoundError(p)
        got=sha256(p)
        if got != expected:
            raise RuntimeError(f'SHA256 mismatch: {p}\nexpected {expected}\ngot {got}')
        inputs.append({
            'seed':seed,'family':family,'logical_file':p.name,
            'sha256':got,'bytes':p.stat().st_size
        })
        maxdiff=0.0
        kept=0
        with p.open('r',encoding='utf-8-sig',newline='') as f:
            r=csv.DictReader(f)
            required={'method','username','checkpoint','scope','expected_recall_score',
                      'exposure_count','distinct_study_dates','review_exposure_count',
                      'days_since_last','source_quality'}
            miss=required-set(r.fieldnames or [])
            if miss:
                raise RuntimeError(f'{p}: missing columns {sorted(miss)}')
            for row in r:
                if row['checkpoint'] not in ('FINAL_D330','D330'): continue
                if row['scope'] != 'anchor': continue
                raw=row['method']
                if family=='DSAC':
                    if raw=='Full': method='Dynamic'
                    elif raw=='Static': method='Static'
                    else: continue
                else:
                    if raw!='Full': continue
                    method='MAU'
                user=row['username']
                kept+=1
                for s in SCENARIOS:
                    val=recalc(row,s)
                    key=(s['name'],seed,method,user)
                    sums[key][0]+=val
                    sums[key][1]+=1
                    if s['name']=='LOCKED':
                        maxdiff=max(maxdiff,abs(val-float(row['expected_recall_score'])))
        verification.append({
            'seed':seed,'family':family,'kept_d330_anchor_rows':kept,
            'locked_max_abs_question_diff':maxdiff
        })

    seed_vals=defaultdict(list)
    for (sc,seed,m,u),(tot,n) in sums.items():
        if n != 90:
            raise RuntimeError(f'Expected 90 anchor questions: {sc} {seed} {m} {u}: {n}')
        seed_vals[(sc,seed,m)].append(tot/n)

    seed_rows=[]
    for sc in [s['name'] for s in SCENARIOS]:
        for seed in SEEDS:
            means={m:sum(seed_vals[(sc,seed,m)])/len(seed_vals[(sc,seed,m)])
                   for m in ('Dynamic','Static','MAU')}
            seed_rows.append({
                'scenario':sc,'seed':seed,**means,
                'Dynamic_minus_Static':means['Dynamic']-means['Static'],
                'MAU_minus_Static':means['MAU']-means['Static'],
                'MAU_minus_Dynamic':means['MAU']-means['Dynamic']
            })

    agg=[]
    for sc in [s['name'] for s in SCENARIOS]:
        rr=[r for r in seed_rows if r['scenario']==sc]
        a={k:sum(r[k] for r in rr)/5
           for k in ('Dynamic','Static','MAU','Dynamic_minus_Static',
                     'MAU_minus_Static','MAU_minus_Dynamic')}
        order=' > '.join(sorted(('Dynamic','Static','MAU'),
                                key=lambda mm:a[mm], reverse=True))
        agg.append({
            'scenario':sc,**a,'aggregate_rank':order,
            'all_5_MAU_gt_Dynamic':all(r['MAU_minus_Dynamic']>0 for r in rr),
            'all_5_MAU_gt_Static':all(r['MAU_minus_Static']>0 for r in rr)
        })

    def write_csv(name,rows):
        path=out/name
        with path.open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    write_csv('INPUT_PROVENANCE_PUBLIC.csv',inputs)
    write_csv('BASELINE_REPRO_CHECK.csv',verification)
    write_csv('D330_ANCHOR_SEED_LEVEL_SENSITIVITY.csv',seed_rows)
    write_csv('D330_ANCHOR_AGGREGATE_SENSITIVITY.csv',agg)
    with (out/'SENSITIVITY_SCENARIOS.json').open('w',encoding='utf-8') as f:
        json.dump(SCENARIOS,f,indent=2)

    maxdiff=max(x['locked_max_abs_question_diff'] for x in verification)
    locked=next(x for x in agg if x['scenario']=='LOCKED')
    md=[
        '# Article 2 read-only memory-model sensitivity','',
        'This analysis re-scores already generated D330 anchor question-level records.',
        'It does not rerun any scheduler or policy.','',
        f'- Locked-model maximum absolute question-level reproduction difference: {maxdiff:.12g}',
        f"- Locked aggregate: Dynamic={locked['Dynamic']:.3f}, Static={locked['Static']:.3f}, MAU={locked['MAU']:.3f}",
        '',
        'Interpretation boundary: these are post-hoc engineering perturbations around the locked memory equation, not empirically validated cognitive models.'
    ]
    (out/'SUMMARY.md').write_text('\n'.join(md)+'\n',encoding='utf-8')
    (out/'RUN_STATUS.txt').write_text(
        f'SUCCESS\nGenerated: {datetime.now().isoformat()}\nLocked max abs question diff: {maxdiff:.12g}\n',
        encoding='utf-8'
    )
    z=Path(str(out)+'.zip')
    with zipfile.ZipFile(z,'w',zipfile.ZIP_DEFLATED) as zz:
        for fp in out.iterdir():
            zz.write(fp,fp.name)
    print('SUCCESS')
    print('Folder:',out)
    print('ZIP:',z)
    print('Locked max abs question diff:',maxdiff)

if __name__=='__main__':
    try:
        main()
    except Exception as e:
        print('ERROR:',e,file=sys.stderr)
        sys.exit(2)
