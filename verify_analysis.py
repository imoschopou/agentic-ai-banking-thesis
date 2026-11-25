from pathlib import Path
import json,hashlib
import numpy as np,pandas as pd
from scipy import stats,linalg
R=Path(__file__).resolve().parent;r=json.loads((R/'results/analysis.json').read_text());d=pd.read_csv(R/'results/Coded_100.csv');raw=pd.read_excel(R/'Raw_100_Locked.xlsx')
assert raw.shape==(100,21)
assert hashlib.sha256((R/'Raw_100_Locked.xlsx').read_bytes()).hexdigest()==r['freeze']['source_sha256']
assert all(sum(i['counts'])+i['dk']==100 for i in r['items'])
assert r['total_selections']==sum(v['count'] for v in r['safeguards'])
reg=d[['INTENT','PU','TRUST','RISK']].dropna();y=reg.INTENT.to_numpy();X=np.c_[np.ones(len(reg)),reg[['PU','TRUST','RISK']].to_numpy()];b=linalg.lstsq(X,y)[0];e=y-X@b;inv=linalg.solve(X.T@X,np.eye(4));hat=np.einsum('ij,jk,ik->i',X,inv,X);cov=np.zeros((4,4))
for xi,ei,hi in zip(X,e,hat):
 v=inv@xi;cov+=np.outer(v,v)*(ei/(1-hi))**2
assert np.allclose(b,[v['b'] for v in r['regression']['rows']])
assert np.allclose(np.sqrt(np.diag(cov)),[v['se_hc3'] for v in r['regression']['rows']])
A=d[['Q17','Q18','Q19']].dropna().to_numpy();N,K=A.shape;ranks=np.array([stats.rankdata(row) for row in A]);q=12/(N*K*(K+1))*np.sum(ranks.sum(axis=0)**2)-3*N*(K+1);ties=sum(sum(v**3-v for v in np.unique(row,return_counts=True)[1]) for row in A);q/=1-ties/(N*(K**3-K));assert np.isclose(q,r['friedman']['chi2'])
for c in r['contrasts']:
 diff=A[:,c['b']-17]-A[:,c['a']-17];diff=diff[diff!=0];rk=stats.rankdata(abs(diff));wp=rk[diff>0].sum();wm=rk[diff<0].sum();assert np.isclose(min(wp,wm),c['w'])
# Flag a uniform response pattern but preserve it in all main analyses.
no49=d[d.ID!='R049'];sensitivity=[]
for p in ['PU','TRUST','RISK']:
 a=no49[[p,'INTENT']].dropna();z=stats.spearmanr(a[p],a.INTENT);sensitivity.append(dict(predictor=p,n=len(a),rho=float(z.statistic),p=float(z.pvalue)))
sigma2=(e@e)/(len(y)-4);cook=e**2/(4*sigma2)*hat/(1-hat)**2
checks=dict(independent_ols_and_hc3=True,independent_friedman=True,independent_wilcoxon_statistics=True,all_item_counts_reconcile=True,frozen_source_hash_verified=True,scenario_complete_n=N,max_cooks_distance=float(cook.max()),above_4_over_n=int(sum(cook>4/len(y))),fitted_range=[float(min(X@b)),float(max(X@b))],without_uniform_pattern=sensitivity)
(R/'results/verification.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2))
print(json.dumps(checks,ensure_ascii=False,indent=2))
