from pathlib import Path
import argparse,json,hashlib,itertools,warnings
import numpy as np,pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def holm(ps):
 p=np.asarray(ps,float);order=np.argsort(p);out=np.empty_like(p);out[order]=np.minimum(1,np.maximum.accumulate((len(p)-np.arange(len(p)))*p[order]));return out.tolist()
def alpha(a):
 a=np.asarray(a,float);k=a.shape[1];v=np.var(a.sum(axis=1),ddof=1)
 return float(k/(k-1)*(1-np.var(a,axis=0,ddof=1).sum()/v)) if v>0 else np.nan
def bci(a,fn,B=4000,seed=20261017):
 a=np.asarray(a);rng=np.random.default_rng(seed);vals=[fn(a[rng.integers(0,len(a),len(a))]) for _ in range(B)];return np.nanpercentile(vals,[2.5,97.5]).tolist()
def cleanjson(o):
 if isinstance(o,dict):return {str(k):cleanjson(v) for k,v in o.items()}
 if isinstance(o,(list,tuple)):return [cleanjson(x) for x in o]
 if isinstance(o,np.generic):return cleanjson(o.item())
 if isinstance(o,float) and not np.isfinite(o):return None
 return o

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--input',default='Raw_100_Locked.xlsx');args=parser.parse_args();root=Path(__file__).resolve().parent;inp=Path(args.input);inp=inp if inp.is_absolute() else root/inp
 out=root/'results';out.mkdir(exist_ok=True);fig=root/'figures';fig.mkdir(exist_ok=True)
 raw=pd.read_excel(inp);assert raw.shape==(100,21),raw.shape
 df=raw.copy();df.columns=['timestamp']+[f'Q{i}' for i in range(1,21)]
 df.insert(0,'ID',[f'R{i:03}' for i in range(1,101)])
 lik={'Διαφωνώ απόλυτα':1,'Διαφωνώ':2,'Ούτε συμφωνώ ούτε διαφωνώ':3,'Συμφωνώ':4,'Συμφωνώ απόλυτα':5}
 scen={'Καθόλου πιθανό':1,'Λίγο πιθανό':2,'Μέτρια πιθανό':3,'Πολύ πιθανό':4,'Πάρα πολύ πιθανό':5}
 dk='Δεν γνωρίζω / Δεν μπορώ να κρίνω'
 num=pd.DataFrame({'ID':df.ID})
 for q in range(1,6):num[f'Q{q}']=df[f'Q{q}']
 for q in range(6,20):
  vals=df[f'Q{q}'].astype(str).str.strip();mp=lik if q<17 else scen
  assert set(vals)<=set(mp)|{dk},(q,set(vals)-set(mp)-{dk})
  num[f'Q{q}']=vals.map(mp)
 # Preserve observed risk direction: 5 denotes greater concern, with no reverse coding.
 specs={'PU':([6,7],'Χρησιμότητα'),'TRUST':([8,9,10],'Εμπιστοσύνη'),'RISK':([11,12,13],'Κίνδυνος'),'INTENT':([14,15,16],'Πρόθεση χρήσης')}
 reli=[];desc=[]
 for key,(qs,label) in specs.items():
  cols=[f'Q{i}' for i in qs];a=num[cols].dropna().to_numpy(float)
  num[key]=num[cols].mean(axis=1).where(num[cols].notna().all(axis=1))
  aci=bci(a,alpha,B=3000);sc=num[key].dropna().to_numpy()
  itemtotal=[]
  for j in range(len(qs)):itemtotal.append(float(stats.pearsonr(a[:,j],np.delete(a,j,axis=1).sum(axis=1)).statistic))
  reli.append(dict(scale=key,label=label,items=qs,n=len(a),alpha=alpha(a),alpha_ci=aci,corrected_item_total=itemtotal,interitem_pearson=float(stats.pearsonr(a[:,0],a[:,1]).statistic) if len(qs)==2 else None))
  desc.append(dict(scale=key,label=label,n=len(sc),mean=float(np.mean(sc)),sd=float(np.std(sc,ddof=1)),median=float(np.median(sc)),q1=float(np.quantile(sc,.25)),q3=float(np.quantile(sc,.75)),min=float(min(sc)),max=float(max(sc)),mean_ci=bci(sc,np.mean)))
 items=[]
 for q in range(6,20):
  x=num[f'Q{q}'];items.append(dict(q=q,text=raw.columns[q],n=int(x.notna().sum()),dk=int(x.isna().sum()),mean=float(x.mean()),median=float(x.median()),counts=[int((x==i).sum()) for i in range(1,6)],high=int((x>=4).sum()),low=int((x<=2).sum())))
 corrs=[]
 for pred in ['PU','TRUST','RISK']:
  a=num[[pred,'INTENT']].dropna().to_numpy();rho,p=stats.spearmanr(a[:,0],a[:,1]);ci=bci(a,lambda z:stats.spearmanr(z[:,0],z[:,1]).statistic)
  corrs.append(dict(predictor=pred,n=len(a),rho=float(rho),p=float(p),ci=ci))
 for r,p in zip(corrs,holm([x['p'] for x in corrs])):r['p_holm']=p
 matrix=num[list(specs)].corr(method='spearman')
 # OLS with HC3 robust covariance, complete cases, one model with three focal predictors.
 reg=num[['INTENT','PU','TRUST','RISK']].dropna();y=reg.INTENT.to_numpy();Z=reg[['PU','TRUST','RISK']].to_numpy();X=np.column_stack([np.ones(len(y)),Z]);N,K=X.shape
 inv=np.linalg.inv(X.T@X);beta=inv@X.T@y;res=y-X@beta;h=np.sum((X@inv)*X,axis=1);cov=inv@X.T@np.diag((res/(1-h))**2)@X@inv;se=np.sqrt(np.diag(cov));tv=beta/se;pv=2*stats.t.sf(abs(tv),N-K);crit=stats.t.ppf(.975,N-K)
 r2=1-(res@res)/np.sum((y-y.mean())**2);vifs=[]
 for j in range(3):
  v=Z[:,j];a=np.column_stack([np.ones(N),np.delete(Z,j,axis=1)]);r=v-a@np.linalg.lstsq(a,v,rcond=None)[0];vifs.append(float(np.sum((v-v.mean())**2)/(r@r)))
 regrows=[dict(term=t,b=float(b),se_hc3=float(s),ci=[float(b-crit*s),float(b+crit*s)],t=float(tval),p=float(p),vif=(vifs[j-1] if j else None),beta_standardized=(float(b*np.std(Z[:,j-1],ddof=1)/np.std(y,ddof=1)) if j else None)) for j,(t,b,s,tval,p) in enumerate(zip(['Intercept','PU','TRUST','RISK'],beta,se,tv,pv))]
 # Matched repeated ratings: same complete participants for omnibus and all contrasts.
 A=num[['Q17','Q18','Q19']].dropna().to_numpy();fr=stats.friedmanchisquare(*A.T);pairs=[]
 for i,j in itertools.combinations(range(3),2):
  d=A[:,j]-A[:,i];w=stats.wilcoxon(d,zero_method='wilcox',alternative='two-sided',method='asymptotic');nz=d[d!=0];ranks=stats.rankdata(abs(nz));wp=float(ranks[nz>0].sum());wm=float(ranks[nz<0].sum());rbc=(wp-wm)/(wp+wm)
  pairs.append(dict(a=i+17,b=j+17,n=len(d),nonzero=len(nz),ties=int(sum(d==0)),w=float(w.statistic),p=float(w.pvalue),rank_biserial=float(rbc),median_difference=float(np.median(d)),mean_difference=float(np.mean(d)),difference_ci=bci(d,np.mean)))
 for r,p in zip(pairs,holm([r['p'] for r in pairs])):r['p_holm']=p
 # High endorsement as proportions among all 100, preserving DK in denominators.
 scenario=[]
 for q in [17,18,19]:
  x=num[f'Q{q}'];k=int((x>=4).sum());z=stats.norm.ppf(.975);p=k/100;den=1+z*z/100;ctr=(p+z*z/200)/den;half=z*np.sqrt(p*(1-p)/100+z*z/(4*100**2))/den
  scenario.append(dict(q=q,n_valid=int(x.notna().sum()),high=k,denominator=100,high_ci=[ctr-half,ctr+half],mean=float(x.mean()),median=float(x.median())))
 opts=sorted({s for ans in df.Q20 for s in ans.split(', ')})
 selections=df.Q20.str.split(', ');counts={s:sum(s in a for a in selections) for s in opts};never=next(s for s in opts if s.startswith('Δεν θα'));unknown=next(s for s in opts if s.startswith('Δεν γνωρίζω'));mixed=df.ID[[never in a and len(a)>1 for a in selections]].tolist()
 safeguards=[dict(option=s,count=int(k),percent=float(k),without_mixed=int(sum(s in a for idx,a in enumerate(selections) if df.ID.iloc[idx] not in mixed)),den_without_mixed=100-len(mixed)) for s,k in sorted(counts.items(),key=lambda v:-v[1])]
 ageorder=['18–24','25–34','35–44','45–54','55–64','65 και άνω'];freqorder=['Σπανιότερα','Μερικές φορές τον μήνα','Μερικές φορές την εβδομάδα','Καθημερινά']
 num['AGE_ORD']=df.Q1.map({v:i+1 for i,v in enumerate(ageorder)});num['BANK_ORD']=df.Q4.map({v:i+1 for i,v in enumerate(freqorder)})
 profile=[]
 for pred in ['AGE_ORD','BANK_ORD','Q5']:
  for outcome in ['TRUST','INTENT']:
   a=num[[pred,outcome]].dropna().to_numpy();r,p=stats.spearmanr(a[:,0],a[:,1]);profile.append(dict(predictor=pred,outcome=outcome,n=len(a),rho=float(r),p=float(p),ci=bci(a,lambda z:stats.spearmanr(z[:,0],z[:,1]).statistic,B=3000)))
 for r,p in zip(profile,holm([r['p'] for r in profile])):r['p_holm']=p
 groups=[]
 for q in [1,2,3,4]:
  for category in df[f'Q{q}'].unique():
   m=df[f'Q{q}']==category
   groups.append(dict(q=q,category=category,n=int(m.sum()),**{key:dict(n=int(num.loc[m,key].notna().sum()),mean=float(num.loc[m,key].mean()) if num.loc[m,key].notna().any() else None,median=float(num.loc[m,key].median()) if num.loc[m,key].notna().any() else None) for key in ['TRUST','INTENT']}))
 # Sensitivity: available-item averages where at least 2 of 3, or both PU items, are observed.
 sensitivity=[]
 alt={key:num[[f'Q{i}' for i in qs]].mean(axis=1).where(num[[f'Q{i}' for i in qs]].notna().sum(axis=1)>=2) for key,(qs,_) in specs.items()}
 for key in ['PU','TRUST','RISK']:
  a=pd.concat([alt[key],alt['INTENT']],axis=1).dropna();r,p=stats.spearmanr(a.iloc[:,0],a.iloc[:,1]);sensitivity.append(dict(predictor=key,n=len(a),rho=float(r),p=float(p)))
 # At item level, audit response concentration without automatic exclusion.
 straight_ids=df.ID[num[[f'Q{i}' for i in range(6,17)]].nunique(axis=1,dropna=False)==1].tolist()
 codebook=[dict(column=f'Q{q}',question=raw.columns[q],coding=('1–10' if q==5 else ('1–5; DK=missing' if 6<=q<=19 else 'Original categories / multiple selection'))) for q in range(1,21)]
 result=dict(freeze=dict(n=100,first=str(df.timestamp.min()),last=str(df.timestamp.max()),source_sha256=hashlib.sha256(inp.read_bytes()).hexdigest(),blank_cells=int(raw.isna().sum().sum()),exact_duplicate_answers=int(raw.iloc[:,1:].duplicated().sum()),unique_timestamps=int(df.timestamp.nunique()),excluded=0,straightlining_ids=straight_ids),demographics={f'Q{q}':df[f'Q{q}'].value_counts().to_dict() for q in range(1,5)},ai=dict(mean=float(num.Q5.mean()),sd=float(num.Q5.std()),median=float(num.Q5.median()),counts=num.Q5.value_counts().sort_index().to_dict()),reliability=reli,scales=desc,items=items,correlations=corrs,spearman_matrix=matrix.to_dict(),regression=dict(n=N,r2=float(r2),adj_r2=float(1-(1-r2)*(N-1)/(N-K)),df=N-K,max_leverage=float(h.max()),rows=regrows),friedman=dict(n=len(A),chi2=float(fr.statistic),df=2,p=float(fr.pvalue),kendall_w=float(fr.statistic/(len(A)*2))),contrasts=pairs,scenarios=scenario,safeguards=safeguards,mixed_never_ids=mixed,total_selections=int(sum(map(len,selections))),profile_correlations=profile,group_descriptives=groups,sensitivity=sensitivity,codebook=codebook)
 (out/'analysis.json').write_text(json.dumps(cleanjson(result),ensure_ascii=False,indent=2))
 num['Q20']=df.Q20
 num.to_csv(out/'Coded_100.csv',index=False,encoding='utf-8-sig');raw.to_csv(out/'Raw_100_export.csv',index=False,encoding='utf-8-sig')
 for name,records in [('Items',items),('Scales',desc),('Reliability',reli),('Correlations',corrs),('Regression',regrows),('Scenario_contrasts',pairs),('Safeguards',safeguards),('Profile_correlations',profile),('Codebook',codebook)]:pd.DataFrame(records).to_csv(out/(name+'.csv'),index=False,encoding='utf-8-sig')
 matrix.to_csv(out/'Spearman_matrix.csv',encoding='utf-8-sig')
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'savefig.bbox':'tight'})
 colors=['#b64040','#df9473','#d3d7dd','#65a59a','#236d64','#ece6bc']
 def stacked(qs,labels,name):
  fig1,ax=plt.subplots(figsize=(10, max(2.6,.46*len(qs)+1.3)));left=np.zeros(len(qs))
  for j in range(6):
   vals=np.array([items[q-6]['counts'][j] if j<5 else items[q-6]['dk'] for q in qs]);ax.barh(range(len(qs)),vals,left=left,color=colors[j],label=(['1','2','3','4','5','Δεν γνωρίζω'][j]))
   for i,v in enumerate(vals):
    if v>=5:ax.text(left[i]+v/2,i,str(v)+'%',ha='center',va='center',color=('white' if j in [0,4] else '#152028'),fontsize=10)
   left+=vals
  ax.set(yticks=range(len(qs)),yticklabels=labels,xlim=(0,100),xlabel='Ποσοστό επί των 100 απαντήσεων');ax.invert_yaxis();ax.legend(ncol=6,loc='lower center',bbox_to_anchor=(.5,1.02),frameon=False);fig1.savefig(fig/name,dpi=200);plt.close(fig1)
 stacked(list(range(6,17)),['Χρόνος','Ευκολία διαχείρισης','Ορθή εκτέλεση','Βασίζομαι στον βοηθό','Όρια εξουσιοδότησης','Οικονομική ζημία','Έκθεση δεδομένων','Παραπλάνηση από απάτη','Δοκιμή υπηρεσίας','Χρήση για υποχρεώσεις','Συνήθης χρήση'],'01_Items.png')
 stacked([17,18,19],['Α Προτείνει και πληρώνω εγώ','Β Εκτελεί μετά την έγκρισή μου','Γ Εκτελεί εντός εξουσιοδότησης'],'02_Scenarios.png')
 fig1,ax=plt.subplots(figsize=(9,4.5));labels=['Έγκριση κάθε συναλλαγής','Άμεση διακοπή λειτουργίας','Ενημέρωση κάθε ενέργειας','Σαφείς όροι ευθύνης / ζημίας','Όρια ποσών και αποδεκτών','Επικοινωνία με άνθρωπο','Καμία χρήση ανεξαρτήτως δικλίδων','Κατανοητή εξήγηση','Δεν γνωρίζω']
 lookup={s['option']:s['count'] for s in safeguards};ordered=list(lookup);short=[]
 for s in ordered:
  short.append('Έγκριση κάθε συναλλαγής' if s.startswith('Να εγκρίνω') else 'Άμεση διακοπή λειτουργίας' if s.startswith('Να μπορώ να διακόπτω') else 'Ενημέρωση κάθε ενέργειας' if s.startswith('Να λαμβάνω') else 'Όροι ευθύνης και ζημίας' if s.startswith('Να υπάρχουν') else 'Όρια ποσών και αποδεκτών' if s.startswith('Να ορίζω') else 'Επικοινωνία με άνθρωπο' if s.startswith('Να μπορώ να επικοινωνώ') else 'Καμία χρήση ανεξαρτήτως δικλίδων' if s.startswith('Δεν θα') else 'Κατανοητή εξήγηση' if s.startswith('Να έχω') else 'Δεν γνωρίζω')
 ax.barh(range(len(ordered)),[lookup[s] for s in ordered],color='#236d64');ax.set(yticks=range(len(ordered)),yticklabels=short,xlim=(0,80),xlabel='Ποσοστό συμμετεχόντων — έως 3 επιλογές');ax.invert_yaxis()
 for i,s in enumerate(ordered):ax.text(lookup[s]+1,i,f'{lookup[s]}%',va='center')
 fig1.savefig(fig/'03_Safeguards.png',dpi=200);plt.close(fig1)
 fig1,ax=plt.subplots(figsize=(6,4.4));im=ax.imshow(matrix,vmin=-1,vmax=1,cmap='RdBu_r');labels=['Χρησιμότητα','Εμπιστοσύνη','Κίνδυνος','Πρόθεση'];ax.set(xticks=range(4),xticklabels=labels,yticks=range(4),yticklabels=labels);plt.setp(ax.get_xticklabels(),rotation=25,ha='right')
 for i in range(4):
  for j in range(4):ax.text(j,i,f'{matrix.iloc[i,j]:.2f}',ha='center',va='center',color='white' if abs(matrix.iloc[i,j])>.6 else 'black')
 fig1.colorbar(im,ax=ax,label='Spearman ρ');fig1.savefig(fig/'04_Correlations.png',dpi=200);plt.close(fig1)
 print(json.dumps(cleanjson({k:result[k] for k in ['freeze','reliability','scales','correlations','regression','friedman','contrasts','profile_correlations','mixed_never_ids','sensitivity']}),ensure_ascii=False,indent=2))
if __name__=='__main__':main()
