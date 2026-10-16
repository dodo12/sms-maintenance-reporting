"""Text-specific analyses on FMC-MWO2KG: official split, 5x5 CV model table, nested-free OOF analyses (reliability, top-k),
group codebooks (rate-distortion), SMS-style noise robustness, confusion matrix and per-class scores.

Output: results/text_study/*.json|npy
"""
import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from smsreport.paths import FMC_DIR, RESULTS
OUT = str(RESULTS / "text_study"); os.makedirs(OUT, exist_ok=True)
import numpy as np, json, random, warnings, collections, math
warnings.filterwarnings("ignore")
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline, make_union
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import ComplementNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from sklearn.cluster import AgglomerativeClustering
from scipy.stats import wilcoxon

def load(f):
    return [l.rstrip('\n').rsplit(',',1) for l in open(f,encoding='utf8') if l.strip()]
tr,dv,te=load(FMC_DIR/'train.txt'),load(FMC_DIR/'dev.txt'),load(FMC_DIR/'test.txt')
X=lambda d:[r[0].strip().lower() for r in d]; Y=lambda d:[r[1].strip() for r in d]
Xtr,Ytr,Xdv,Ydv,Xte,Yte=X(tr),Y(tr),X(dv),Y(dv),X(te),Y(te)
Xall,Yall=np.array(Xtr+Xdv+Xte,dtype=object),np.array(Ytr+Ydv+Yte)
classes=sorted(set(Yall)); print(len(Xall),len(classes))
json.dump({'n_train':len(tr),'n_dev':len(dv),'n_test':len(te),'n_all':len(Xall),
  'n_classes':len(classes),'counts':collections.Counter(Yall.tolist()),
  'dup_obs_conflict':sum(1 for k,v in collections.Counter(Xall.tolist()).items() if len(set(Yall[[i for i,x in enumerate(Xall) if x==k]]))>1),
  'unique_obs':len(set(Xall.tolist()))},open(OUT+'/data.json','w'),indent=1)

def feats(kind):
    if kind=='char': return TfidfVectorizer(analyzer='char_wb',ngram_range=(2,5),sublinear_tf=True)
    if kind=='word': return TfidfVectorizer(analyzer='word',ngram_range=(1,2),sublinear_tf=True)
    return make_union(TfidfVectorizer(analyzer='char_wb',ngram_range=(2,5),sublinear_tf=True),
                      TfidfVectorizer(analyzer='word',ngram_range=(1,2),sublinear_tf=True))
def mk(name,C=10):
    if name=='Majority': return DummyClassifier(strategy='most_frequent')
    f=feats('word') if name.endswith('word') else feats('char') if name.endswith('char') else feats('both')
    if name.startswith('LR'): m=LogisticRegression(C=C,max_iter=2000)
    elif name.startswith('SVM'): m=LinearSVC(C=C,random_state=0)
    elif name.startswith('CNB'): m=ComplementNB(alpha=0.3)
    elif name.startswith('kNN'): m=KNeighborsClassifier(n_neighbors=3,metric='cosine',weights='distance')
    return make_pipeline(f,m)
# ---- hyperparameters on official dev only
sel={}
for nm in ['LR-char','LR-both','SVM-char']:
    best=max([(accuracy_score(Ydv,mk(nm,C).fit(Xtr,Ytr).predict(Xdv)),C) for C in [1,10,100]]); sel[nm]=best[1]
print('selected C',sel)
models=['Majority','CNB-char','kNN-char','SVM-char','LR-word','LR-char','LR-both']
Cs=lambda n:sel.get(n,10)
# ---- official split
off={}
for nm in models:
    p=mk(nm,Cs(nm)).fit(Xtr+Xdv,Ytr+Ydv).predict(Xte)  # train+dev for final test
    off[nm]={'acc':accuracy_score(Yte,p),'macroF1':f1_score(Yte,p,average='macro'),'microF1':f1_score(Yte,p,average='micro'),'wF1':f1_score(Yte,p,average='weighted')}
json.dump(off,open(OUT+'/official.json','w'),indent=1)
# ---- repeated CV pooled
rskf=RepeatedStratifiedKFold(n_splits=5,n_repeats=5,random_state=0)
splits=list(rskf.split(Xall,Yall))
cv={nm:{'acc':[],'macroF1':[],'wF1':[]} for nm in models}
for tri,tei in splits:
    for nm in models:
        p=mk(nm,Cs(nm)).fit(list(Xall[tri]),Yall[tri]).predict(list(Xall[tei]))
        cv[nm]['acc'].append(accuracy_score(Yall[tei],p)); cv[nm]['macroF1'].append(f1_score(Yall[tei],p,average='macro')); cv[nm]['wF1'].append(f1_score(Yall[tei],p,average='weighted'))
cvs={nm:{k:(float(np.mean(v)),float(np.std(v,ddof=1))) for k,v in d.items()} for nm,d in cv.items()}
best=max([m for m in models if m!='Majority'],key=lambda m:cvs[m]['acc'][0]); print('best',best)
wil={nm:float(wilcoxon(cv['LR-char']['acc'],cv[nm]['acc']).pvalue) for nm in models if nm!='LR-char' and np.any(np.array(cv['LR-char']['acc'])!=np.array(cv[nm]['acc']))}
json.dump({'cv':cvs,'wilcoxon_vs_LRchar':wil,'best':best},open(OUT+'/cv.json','w'),indent=1)

# ---- OOF probabilities / rate-distortion / selective prediction / noise, using LR-char
C0=Cs('LR-char')
def centroid_groups(Xt,Yt,K):
    v=TfidfVectorizer(analyzer='char_wb',ngram_range=(2,5),sublinear_tf=True); M=v.fit_transform(Xt)
    labs=sorted(set(Yt)); cen=np.vstack([np.asarray(M[[i for i,y in enumerate(Yt) if y==l]].mean(0)).ravel() for l in labs])
    cen/=np.linalg.norm(cen,axis=1,keepdims=True)+1e-12
    if K>=len(labs): return {l:i for i,l in enumerate(labs)}
    a=AgglomerativeClustering(n_clusters=K,metric='cosine',linkage='average').fit(cen)
    return {l:int(c) for l,c in zip(labs,a.labels_)}
Ks=[2,3,4,6,8,10,12,14,16,18,20,22]
rd={K:[] for K in Ks}; topk={k:[] for k in [1,2,3,4,5]}
oof_conf=[];oof_pred=[];oof_true=[]
rs=list(RepeatedStratifiedKFold(n_splits=5,n_repeats=3,random_state=1).split(Xall,Yall))
for tri,tei in rs:
    Xt,Yt=list(Xall[tri]),Yall[tri]; Xe,Ye=list(Xall[tei]),Yall[tei]
    m=mk('LR-char',C0).fit(Xt,Yt); P=m.predict_proba(Xe); cl=list(m.classes_)
    pred=[cl[i] for i in P.argmax(1)]
    oof_conf+=P.max(1).tolist(); oof_pred+=pred; oof_true+=Ye.tolist()
    for K in Ks:
        g=centroid_groups(Xt,Yt,K)
        rd[K].append(np.mean([g.get(p,-1)==g.get(t,-2) for p,t in zip(pred,Ye)]))
    order=np.argsort(-P,axis=1)
    for k in topk: topk[k].append(np.mean([Ye[i] in [cl[j] for j in order[i,:k]] for i in range(len(Ye))]))
rdm={K:(float(np.mean(v)),float(np.std(v,ddof=1))) for K,v in rd.items()}
tkm={k:(float(np.mean(v)),float(np.std(v,ddof=1))) for k,v in topk.items()}
# entropy-coded rate (empirical label entropy)
cnt=np.array(list(collections.Counter(Yall.tolist()).values()),float); pr=cnt/cnt.sum(); H=float(-(pr*np.log2(pr)).sum())
conf=np.array(oof_conf); ok=np.array([p==t for p,t in zip(oof_pred,oof_true)])
o=np.argsort(-conf); cov=np.arange(1,len(o)+1)/len(o); risk=1-np.cumsum(ok[o])/np.arange(1,len(o)+1)
aurc=float(risk.mean())
def acc_at(c): n=int(round(c*len(o))); return float(ok[o][:n].mean())
bins=np.linspace(0,1,11); ece=0; rel=[]
for a,b in zip(bins[:-1],bins[1:]):
    s=(conf>a)&(conf<=b)
    if s.sum(): ece+=s.mean()*abs(ok[s].mean()-conf[s].mean()); rel.append((float(conf[s].mean()),float(ok[s].mean()),int(s.sum())))
sel_res={'aurc':aurc,'random_aurc':float(1-ok.mean()),'acc_cov':{c:acc_at(c) for c in [1,.9,.8,.7,.6,.5]},'ece':float(ece),'oof_acc':float(ok.mean()),
  'thr_for_90':None}
for t in np.linspace(0.05,0.9,171):
    s=conf>=t
    if s.sum()>20 and ok[s].mean()>=0.9: sel_res['thr_for_90']=(float(t),float(s.mean()),float(ok[s].mean())); break
json.dump({'rd':rdm,'topk':tkm,'H':H,'sel':sel_res,'rel':rel},open(OUT+'/rd_sel.json','w'),indent=1)
np.save(OUT+'/cov.npy',cov);np.save(OUT+'/risk.npy',risk)
# confusion (oof)
labs=sorted(set(Yall)); idx={l:i for i,l in enumerate(labs)}; Cm=np.zeros((len(labs),len(labs)))
for t,p in zip(oof_true,oof_pred): Cm[idx[t],idx[p]]+=1
np.save(OUT+'/conf.npy',Cm); json.dump(labs,open(OUT+'/labs.json','w'))
perclass={l:{'prec':float(Cm[idx[l],idx[l]]/max(Cm[:,idx[l]].sum(),1)),'rec':float(Cm[idx[l],idx[l]]/max(Cm[idx[l]].sum(),1)),'n':int(Cm[idx[l]].sum())} for l in labs}
json.dump(perclass,open(OUT+'/perclass.json','w'),indent=1)

# ---- SMS-style noise
rng=random.Random(0)
def noise(s,r,vowel=False):
    out=[]
    for ch in s:
        if vowel and ch in 'aeiou' and rng.random()<0.7: continue
        u=rng.random()
        if ch!=' ' and u<r/3: continue
        if ch!=' ' and u<2*r/3: out.append(rng.choice('abcdefghijklmnopqrstuvwxyz')); continue
        if ch!=' ' and u<r: out.append(ch+ch); continue
        out.append(ch)
    return ''.join(out)
levels=[0,.05,.1,.2,.3]
nz={'clean-train':{r:[] for r in levels},'aug-train':{r:[] for r in levels},'word-clean':{r:[] for r in levels},'vowel-drop':{'clean':[],'aug':[]}}
for tri,tei in rs[:10]:
    Xt,Yt=list(Xall[tri]),list(Yall[tri]); Xe,Ye=list(Xall[tei]),list(Yall[tei])
    Xa=Xt+[noise(x,r) for x in Xt for r in (.1,.2)]+[noise(x,0,True) for x in Xt]; Ya=Yt+[y for y in Yt for _ in (0,1)]+Yt
    mc=mk('LR-char',C0).fit(Xt,Yt); ma=mk('LR-char',C0).fit(Xa,Ya); mw=mk('LR-word',10).fit(Xt,Yt)
    for r in levels:
        Xn=[noise(x,r) for x in Xe]
        nz['clean-train'][r].append(accuracy_score(Ye,mc.predict(Xn))); nz['aug-train'][r].append(accuracy_score(Ye,ma.predict(Xn))); nz['word-clean'][r].append(accuracy_score(Ye,mw.predict(Xn)))
    Xv=[noise(x,0,True) for x in Xe]
    nz['vowel-drop']['clean'].append(accuracy_score(Ye,mc.predict(Xv))); nz['vowel-drop']['aug'].append(accuracy_score(Ye,ma.predict(Xv)))
nzm={k:{str(r):(float(np.mean(v)),float(np.std(v,ddof=1))) for r,v in d.items()} for k,d in nz.items()}
json.dump(nzm,open(OUT+'/noise.json','w'),indent=1)
print(json.dumps({'off':off,'cvs':cvs,'rd':rdm,'topk':tkm,'sel':sel_res,'noise':nzm},indent=1,default=str))


# ---- majority-group floor of each codebook (needed to normalise accuracy, see manuscript)
import collections
from sklearn.model_selection import RepeatedStratifiedKFold as _RSKF
rs=list(_RSKF(n_splits=5,n_repeats=3,random_state=1).split(Xall,Yall))
Ks=[2,3,4,6,8,10,12,14,16,18,20,22]; res={K:{'maj':[],'H':[],'maxsize':[]} for K in Ks}
for tri,tei in rs:
    Xt,Yt=list(Xall[tri]),Yall[tri]; Ye=Yall[tei]
    for K in Ks:
        g=centroid_groups(Xt,Yt,K); gt=np.array([g.get(y,-1) for y in Ye]); gtr=np.array([g[y] for y in Yt])
        mj=collections.Counter(gtr).most_common(1)[0][0]; res[K]['maj'].append(np.mean(gt==mj))
        p=np.array(list(collections.Counter(gtr).values()),float);p/=p.sum();res[K]['H'].append(-(p*np.log2(p)).sum())
        res[K]['maxsize'].append(max(collections.Counter(g.values()).values()))
out={K:{k:float(np.mean(v)) for k,v in d.items()} for K,d in res.items()}
json.dump(out,open(OUT+'/groupbase.json','w'),indent=1);print(json.dumps(out))
