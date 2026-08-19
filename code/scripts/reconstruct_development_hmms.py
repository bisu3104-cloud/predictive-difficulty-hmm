#!/usr/bin/env python3
"""Reconstruct the 24 development HMM matrices from public candidate metrics."""
from pathlib import Path
import numpy as np, pandas as pd
from scipy.linalg import eig
ROOT=Path(__file__).resolve().parents[2]

def stationary(T):
    vals,vecs=eig(T.T); v=np.abs(np.real(vecs[:,np.argmin(np.abs(vals-1))])); return v/v.sum()
def generate(K,V,alpha_t,self_bias,alpha_e,seed):
    rng=np.random.default_rng(int(seed)); T=np.zeros((int(K),int(K)))
    for i in range(int(K)):
        a=np.full(int(K),float(alpha_t)); a[i]+=float(self_bias); T[i]=rng.dirichlet(a)
    O=np.vstack([rng.dirichlet(np.full(int(V),float(alpha_e))) for _ in range(int(K))])
    return T,O,stationary(T)
sel=pd.read_csv(ROOT/'data/selections/development_selected_hmms.csv')
out=ROOT/'data/hmms/development'; out.mkdir(parents=True,exist_ok=True)
for _,r in sel.iterrows():
    T,O,pi=generate(r.K,r.vocab_size,r.alpha_t,r.self_bias,r.alpha_e,r.gen_seed)
    np.savez(out/f'{r.hmm_id}.npz',T=T,O=O,pi=pi)
print(f'Reconstructed {len(sel)} HMMs in {out}')
