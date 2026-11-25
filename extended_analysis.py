"""Hierarchical OLS and exploratory factor analysis of the frozen 100 responses.

Run: python extended_analysis.py --input Raw_100_Locked.xlsx
No mediation, independent replication, or simulated survey responses are used.
Permutation samples are used only as a factor-retention reference distribution.
"""
from pathlib import Path
import argparse, hashlib, json, warnings, importlib.metadata
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
from statsmodels.stats.outliers_influence import variance_inflation_factor
from factor_analyzer import FactorAnalyzer, calculate_kmo, calculate_bartlett_sphericity
from factor_analyzer.rotator import Rotator
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SEED = 20261017
EXPECTED_SHA256 = 'ee7bf681fe135d265d47241ef1d06c6a7b8d05cf9f2710c0b3a783c00ecebdfd'
SCALES = {'PU':[6,7], 'TRUST':[8,9,10], 'RISK':[11,12,13], 'INTENT':[14,15,16]}

def serial(x):
    if isinstance(x, dict): return {str(k):serial(v) for k,v in x.items()}
    if isinstance(x, (list,tuple)): return [serial(v) for v in x]
    if isinstance(x, np.ndarray): return serial(x.tolist())
    if isinstance(x, np.generic): return serial(x.item())
    if isinstance(x, float) and not np.isfinite(x): return None
    return x

def code_data(path):
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != EXPECTED_SHA256:
        raise ValueError('Input differs from the frozen source. Review the file; do not bypass this check silently.')
    raw=pd.read_excel(path)
    assert raw.shape==(100,21)
    lik={'Διαφωνώ απόλυτα':1,'Διαφωνώ':2,'Ούτε συμφωνώ ούτε διαφωνώ':3,'Συμφωνώ':4,'Συμφωνώ απόλυτα':5}
    dk='Δεν γνωρίζω / Δεν μπορώ να κρίνω'
    num=pd.DataFrame(index=raw.index)
    for q in range(6,17):
        v=raw.iloc[:,q].astype(str).str.strip()
        assert set(v)<=set(lik)|{dk}
        num[f'Q{q}']=v.map(lik)
    for key,qs in SCALES.items():
        c=[f'Q{q}' for q in qs]
        num[key]=num[c].mean(axis=1).where(num[c].notna().all(axis=1))
    return raw,num,digest

def regression(num, out):
    # Same 90 observations in EVERY block: changes in R2 cannot reflect sample changes.
    reg=num[['INTENT','PU','RISK','TRUST']].dropna()
    y=reg.INTENT.to_numpy(float)
    blocks=[['PU'],['PU','RISK'],['PU','RISK','TRUST']]
    models=[]; rows=[]; coefficients=[]
    for i,terms in enumerate(blocks,1):
        X=add_constant(reg[terms],has_constant='add')
        ols=OLS(y,X).fit()
        robust=ols.get_robustcov_results(cov_type='HC3',use_t=True)
        models.append((ols,robust))
        ci=robust.conf_int()
        for j,t in enumerate(X.columns):
            coefficients.append(dict(model=i,term=t,b=robust.params[j],se_hc3=robust.bse[j],t=robust.tvalues[j],p=robust.pvalues[j],ci_low=ci[j,0],ci_high=ci[j,1],vif=variance_inflation_factor(X.to_numpy(),j) if j else None))
        # HC3 Wald F for the newly added coefficient (one variable per block).
        R=np.zeros((1,X.shape[1])); R[0,-1]=1
        w=robust.wald_test(R,use_f=True,scalar=True)
        previous_r2=models[i-2][0].rsquared if i>1 else 0.
        rows.append(dict(model=i,terms=terms,n=len(y),r2=ols.rsquared,adj_r2=ols.rsquared_adj,delta_r2=ols.rsquared-previous_r2,added=terms[-1],wald_f=float(w.statistic),df_num=float(w.df_num),df_den=float(w.df_denom),p_hc3=float(w.pvalue)))
    # Order sensitivity: contribution of RISK when it enters after PU and TRUST.
    alternate=OLS(y,add_constant(reg[['PU','TRUST']])).fit()
    full,robust=models[-1]
    risk_index=list(full.model.exog_names).index('RISK')
    R=np.zeros((1,len(robust.params)));R[0,risk_index]=1
    w=robust.wald_test(R,use_f=True,scalar=True)
    inf=full.get_influence()
    fig,axs=plt.subplots(1,2,figsize=(10,4))
    axs[0].scatter(full.fittedvalues,full.resid,s=20,alpha=.65)
    axs[0].axhline(0,color='black',lw=.8);axs[0].set(xlabel='Fitted intention score',ylabel='Residual')
    stats.probplot(full.resid,dist='norm',plot=axs[1]);fig.tight_layout();fig.savefig(out.parent/'figures'/'Regression_diagnostics.png',dpi=160);plt.close(fig)
    pd.DataFrame(rows).to_csv(out/'Hierarchical_models.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(coefficients).to_csv(out/'Hierarchical_coefficients.csv',index=False,encoding='utf-8-sig')
    reg.assign(ID=[f'R{i+1:03}' for i in reg.index]).to_csv(out/'Hierarchical_cases.csv',index=False,encoding='utf-8-sig')
    return dict(n=len(y),models=rows,coefficients=coefficients,alternative_risk_last=dict(reduced_r2=alternate.rsquared,delta_r2=full.rsquared-alternate.rsquared,wald_f=float(w.statistic),df_den=float(w.df_denom),p_hc3=float(w.pvalue)),diagnostics=dict(max_cook=float(inf.cooks_distance[0].max()),n_cook_over_4_n=int(sum(inf.cooks_distance[0]>4/len(y))),max_leverage=float(inf.hat_matrix_diag.max()),residual_min=float(full.resid.min()),residual_max=float(full.resid.max())))

def factor_solution(R,k):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        fa=FactorAnalyzer(n_factors=k,method='minres',rotation=None,is_corr_matrix=True)
        fa.fit(R)
    unrotated=fa.loadings_.copy()
    # Rotate explicitly to keep loading columns and Phi in the same order.
    # FactorAnalyzer 0.5.1 sorts rotated loadings but not Phi; avoid that path.
    if k>1:
        rotator=Rotator(method='oblimin',gamma=0)
        L=rotator.fit_transform(unrotated);phi=rotator.phi_.copy()
    else:
        L=unrotated.copy();phi=np.eye(1)
    signs=np.sign(L.sum(axis=0));signs[signs==0]=1
    L=L@np.diag(signs);phi=np.diag(signs)@phi@np.diag(signs)
    order=np.argsort(-np.sum(L**2,axis=0))
    L=L[:,order];phi=phi[np.ix_(order,order)]
    assert np.allclose(L@phi@L.T,unrotated@unrotated.T,atol=1e-7)
    # Oblique common variance is diag(L Phi L'), NOT row sums of squared pattern loadings.
    common=L@phi@L.T;h=np.diag(common);res=R-common
    ix=np.triu_indices_from(R,1)
    return dict(k=k,pattern=L,phi=phi,structure=L@phi,communalities=h,uniqueness=1-h,rmsr=np.sqrt(np.mean(res[ix]**2)),max_abs_residual=np.max(abs(res[ix])),warnings=[str(w.message) for w in caught if not issubclass(w.category,FutureWarning)])

def efa(num,out,B):
    cols=[f'Q{q}' for q in range(6,17)]
    complete=num[cols].dropna();a=complete.to_numpy(float);n,p=a.shape
    R=np.corrcoef(a,rowvar=False)
    kmo_item,kmo=calculate_kmo(a);bartlett,bp=calculate_bartlett_sphericity(a)
    eig=np.linalg.eigvalsh(R)[::-1]
    # Horn-style permutation parallel analysis using FULL correlation eigenvalues.
    # This is a component-eigenvalue retention aid, not a PCA extraction reported as EFA.
    rng=np.random.default_rng(SEED);null_eig=np.empty((B,p))
    for b in range(B):
        perm=np.column_stack([rng.permutation(a[:,j]) for j in range(p)])
        null_eig[b]=np.linalg.eigvalsh(np.corrcoef(perm,rowvar=False))[::-1]
    threshold=np.quantile(null_eig,.95,axis=0)
    retained=0
    for observed,cut in zip(eig,threshold):
        if observed>cut:retained+=1
        else:break
    solutions={str(k):factor_solution(R,k) for k in sorted(set([max(1,retained),2,3,4]))}
    Rs=stats.spearmanr(a,axis=0).statistic
    sensitivity=factor_solution(Rs,max(1,retained))
    pd.DataFrame({'component':np.arange(1,p+1),'observed_full_correlation_eigenvalue':eig,'permutation_p95':threshold,'above_threshold':eig>threshold}).to_csv(out/'Parallel_analysis.csv',index=False,encoding='utf-8-sig')
    for key,sol in solutions.items():
        pd.DataFrame(sol['pattern'],index=cols,columns=[f'F{i+1}' for i in range(sol['k'])]).assign(communality=sol['communalities'],uniqueness=sol['uniqueness']).to_csv(out/f'EFA_{key}_factors.csv',encoding='utf-8-sig')
        pd.DataFrame(sol['phi']).to_csv(out/f'EFA_{key}_phi.csv',index=False)
    pd.DataFrame(sensitivity['pattern'],index=cols).to_csv(out/'EFA_Spearman_sensitivity.csv',encoding='utf-8-sig')
    pd.DataFrame({'item':cols,'KMO':kmo_item}).to_csv(out/'EFA_KMO.csv',index=False)
    x=np.arange(1,p+1);fig,ax=plt.subplots(figsize=(7,4));ax.plot(x,eig,'o-',label='Observed correlation matrix');ax.plot(x,threshold,'s--',label='95th percentile of permutations');ax.axhline(1,color='gray',lw=.7);ax.set(xlabel='Ordered eigenvalue',ylabel='Eigenvalue',xticks=x);ax.legend();fig.tight_layout();fig.savefig(out.parent/'figures'/'Parallel_analysis.png',dpi=180);plt.close(fig)
    return dict(n=n,p=p,excluded_from_efa=100-n,items=cols,kmo=kmo,kmo_item=kmo_item,bartlett_chi2=bartlett,bartlett_df=p*(p-1)//2,bartlett_p=bp,eigenvalues=eig,permutation_p95=threshold,permutations=B,seed=SEED,retained_by_parallel=retained,solutions=solutions,spearman_sensitivity=sensitivity,correlation=R)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--input',default='Raw_100_Locked.xlsx');parser.add_argument('--permutations',type=int,default=2000);args=parser.parse_args()
    root=Path(__file__).resolve().parent;source=Path(args.input);source=source if source.is_absolute() else root/source
    out=root/'results';out.mkdir(exist_ok=True);(root/'figures').mkdir(exist_ok=True)
    raw,num,sha=code_data(source)
    result=dict(source_sha256=sha,seed=SEED,versions={n:importlib.metadata.version(n) for n in ['numpy','pandas','scipy','matplotlib','openpyxl','statsmodels','factor-analyzer','scikit-learn','patsy']},hierarchical=regression(num,out),efa=efa(num,out,args.permutations))
    (out/'extended_analysis.json').write_text(json.dumps(serial(result),ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(serial({ 'versions':result['versions'],'hierarchical':result['hierarchical'],'efa':{k:v for k,v in result['efa'].items() if k not in ['solutions','correlation','spearman_sensitivity']}}),ensure_ascii=False,indent=2))

if __name__=='__main__': main()
