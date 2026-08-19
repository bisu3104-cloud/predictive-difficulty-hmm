#!/usr/bin/env python3
from pathlib import Path
import hashlib, json, re, sys
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[2]


def sha256(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for c in iter(lambda: f.read(1024 * 1024), b''): h.update(c)
    return h.hexdigest()


def check_npz():
    problems=[]; n=0
    for p in (ROOT/'data/hmms').rglob('*.npz'):
        with np.load(p, allow_pickle=False) as z:
            if not {'T','O','pi'}.issubset(z.files): continue
            T,O,pi=z['T'],z['O'],z['pi']; n+=1
            if not np.allclose(T.sum(1),1,atol=1e-10): problems.append(f'{p}: T rows')
            if not np.allclose(O.sum(1),1,atol=1e-10): problems.append(f'{p}: O rows')
            if not np.isclose(pi.sum(),1,atol=1e-10): problems.append(f'{p}: pi sum')
            if not np.allclose(pi@T,pi,atol=1e-9): problems.append(f'{p}: stationarity')
    return n,problems


def check_counts():
    expected={
      'development__main_training_raw.csv':2688,
      'confirmatory__confirmatory_training_raw.csv':2560,
      'predictor_specific__difficulty_neural_training_raw.csv':1440,
      'strict__strict_training_raw.csv':480,
      'external__external_training_raw.csv':576,
      'dataset_realization__training_with_metrics.csv':1152,
    }
    problems=[]
    for fn,n in expected.items():
        p=ROOT/'data/run_level'/fn
        if not p.exists(): problems.append(f'missing {fn}'); continue
        got=len(pd.read_csv(p))
        if got!=n: problems.append(f'{fn}: expected {n}, got {got}')
    return problems


def check_paths():
    problems=[]; pat=re.compile(r'/content/drive/MyDrive')
    for p in ROOT.rglob('*'):
        if p.resolve()==Path(__file__).resolve(): continue
        if p.is_file() and p.suffix.lower() in {'.csv','.json','.md','.txt','.py'}:
            if pat.search(p.read_text(encoding='utf-8', errors='ignore')):
                problems.append(str(p.relative_to(ROOT)))
    return problems


def check_prior_reconstructions():
    problems=[]
    table4=ROOT/'outputs/reconstructed_table4/table4_confirmatory_loho_reconstructed.csv'
    if not table4.exists(): problems.append('missing Table 4 reconstruction summary')
    else:
        df=pd.read_csv(table4).set_index('model')
        expected={'K-only':(1.049,-0.119,-0.489),'PLS-1 composite':(0.739,0.451,0.731),'PLS-2 composite':(0.729,0.466,0.735),'Full ridge':(0.720,0.482,0.733)}
        for model,vals in expected.items():
            got=tuple(round(float(df.loc[model,c]),3) for c in ['standardized_rmse','pooled_r2','mean_spearman'])
            if got!=vals: problems.append(f'Table 4 {model}: expected {vals}, got {got}')
    counts=ROOT/'outputs/reconstructed_table4/table4_full_ridge_selected_alpha_counts.csv'
    if counts.exists():
        cdf=pd.read_csv(counts); got={float(r.selected_alpha):int(r.outer_fold_count) for r in cdf.itertuples(index=False)}
        if got!={4.0:1,8.0:62,16.0:1}: problems.append(f'unexpected nested ridge alpha counts: {got}')
    else: problems.append('missing nested ridge selected-alpha counts')
    var=ROOT/'outputs/variance_components/seed_aware_variance_summary.csv'
    if var.exists():
        vdf=pd.read_csv(var).set_index('outcome')
        for outcome,value in {'Shape RMSE':0.18792931676453026,'Terminal excess CE':16.502929403964554}.items():
            got=float(vdf.loc[outcome,'interaction_share_percent'])
            if not np.isclose(got,value,rtol=0,atol=1e-10): problems.append(f'{outcome} share mismatch: {got}')
    else: problems.append('missing seed-aware variance summary')
    boots=ROOT/'outputs/variance_components/seed_aware_bootstrap_shares.csv'
    if not boots.exists() or len(pd.read_csv(boots))!=20000: problems.append('seed-aware bootstrap table must contain 20,000 rows')
    return problems


def check_dataset_realization():
    problems=[]
    selected=pd.read_csv(ROOT/'data/selections/dataset_realization_selected_hmms.csv')
    if len(selected)!=16 or selected.hmm_id.nunique()!=16: problems.append('dataset-realization selection must contain 16 unique HMMs')
    if selected.groupby('K').size().to_dict()!={3:4,4:4,5:4,6:4}: problems.append('dataset-realization selection must contain four HMMs per K')
    run=pd.read_csv(ROOT/'data/run_level/dataset_realization__training_with_metrics.csv')
    keys=['dataset_replicate','hmm_id','architecture','dimension','seed']
    if run[keys].duplicated().any(): problems.append('duplicate dataset-realization run keys')
    expected=16*3*2*4*3
    if len(run)!=expected: problems.append(f'dataset-realization run grid: expected {expected}, got {len(run)}')
    if set(run.hmm_id.astype(str))!=set(selected.hmm_id.astype(str)): problems.append('run HMM IDs differ from selected subset')
    manifest=pd.read_csv(ROOT/'data/protocols/dataset_realization_replicate_manifest.csv')
    if len(manifest)!=48: problems.append('dataset replicate manifest must contain 48 rows')
    regenerated=manifest[manifest.dataset_replicate.astype(int).isin([1,2])]
    if len(regenerated)!=32 or regenerated.sha256.astype(str).str.len().lt(64).any(): problems.append('regenerated dataset checksum records are incomplete')
    t24=pd.read_csv(ROOT/'outputs/paper_tables/table24_dataset_realization_stability.csv')
    expected_icc={('GRU','A_width_shape'):0.794077,('Transformer','A_width_shape'):0.798620,('GRU','mean_excess_ce_H'):0.549954,('Transformer','mean_excess_ce_H'):0.807372}
    for key,val in expected_icc.items():
        row=t24[(t24.architecture==key[0])&(t24.outcome==key[1])]
        if len(row)!=1 or not np.isclose(float(row.process_icc.iloc[0]),val,atol=1e-6): problems.append(f'Table 24 ICC mismatch for {key}')
    t25=pd.read_csv(ROOT/'outputs/paper_tables/table25_dataset_realization_frozen_prediction.csv')
    expected_reductions=[34.13041526921563,44.39759428247818,46.37911294690804,30.81740854653249,34.75594513969212,27.02848369390255]
    got=t25.sort_values(['architecture','dataset_replicate']).augmented_rmse_reduction_percent.to_numpy(float)
    if not np.allclose(got,expected_reductions,atol=1e-5): problems.append(f'Table 25 reductions mismatch: {got.tolist()}')
    return problems


def check_public_hygiene():
    problems=[]
    forbidden_meta=('userId','displayName','authorship_tag','mount_file_id','executionInfo','outputId')
    for p in (ROOT/'code/notebooks').glob('*.ipynb'):
        try:
            nb=json.loads(p.read_text(encoding='utf-8'))
        except Exception as e:
            problems.append(f'invalid notebook JSON: {p.name}: {e}')
            continue
        raw=p.read_text(encoding='utf-8', errors='ignore')
        for key in forbidden_meta:
            if key in raw: problems.append(f'notebook contains public-release metadata {key}: {p.name}')
        for c in nb.get('cells',[]):
            if c.get('cell_type')=='code':
                if c.get('outputs'): problems.append(f'notebook has stored outputs: {p.name}')
                if c.get('execution_count') is not None: problems.append(f'notebook has execution count: {p.name}')
    cff=(ROOT/'CITATION.cff').read_text(encoding='utf-8', errors='ignore')
    for bad in ('TO_BE_ADDED','[DOI]'):
        if bad in cff: problems.append(f'CITATION.cff contains invalid placeholder: {bad}')
    # Public release must eventually add license files; absence is reported by the checklist, not
    # treated as a validator failure while this package is explicitly a release candidate.
    return problems


def check_manifest():
    problems=[]; p=ROOT/'ARTIFACT_MANIFEST.csv'
    if not p.exists(): return ['missing ARTIFACT_MANIFEST.csv']
    m=pd.read_csv(p)
    listed=set(m.path.astype(str))
    actual={x.relative_to(ROOT).as_posix() for x in ROOT.rglob('*') if x.is_file() and x.name!='ARTIFACT_MANIFEST.csv'}
    if listed!=actual:
        problems.append(f'manifest path mismatch: missing={sorted(actual-listed)[:5]}, extra={sorted(listed-actual)[:5]}')
    for r in m.itertuples(index=False):
        fp=ROOT/r.path
        if fp.exists() and (int(r.size_bytes)!=fp.stat().st_size or str(r.sha256)!=sha256(fp)):
            problems.append(f'manifest checksum mismatch: {r.path}')
            if len(problems)>20: break
    return problems

n,problems=check_npz()
problems += check_counts()+check_prior_reconstructions()+check_dataset_realization()+check_public_hygiene()
paths=check_paths(); problems += check_manifest()
result={'artifact_version':'0.5.0','hmm_npz_checked':n,'dataset_realization_runs_checked':1152,'validation_problems':problems,'absolute_path_files':paths}
print(json.dumps(result,indent=2))
sys.exit(1 if problems or paths else 0)
