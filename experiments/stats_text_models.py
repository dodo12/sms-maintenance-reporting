"""Paired Wilcoxon tests (logistic regression vs other models) on the 15 text folds, plus dataset/feature dimensions
and software versions. Requires results/curves/text_FMC__*_confok.npy from run_models.py. Output: results/extra.json
"""
import json, platform, sys, collections
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np, sklearn, scipy
from scipy.stats import wilcoxon
from sklearn.feature_extraction.text import TfidfVectorizer
from smsreport import datasets as ds
from smsreport.paths import RESULTS

X, y, outer, _ = ds.text()
cut = np.cumsum([len(te) for _, te in outer])[:-1]
acc = {}
for m in ["LR", "kNN", "RandomForest", "SVM-linear(Platt)", "MLP"]:
    ok = np.load(RESULTS / "curves" / f"text_FMC__{m}_confok.npy")[1]
    acc[m] = np.array([a.mean() for a in np.split(ok, cut)])
out = {"versions": {"sklearn": sklearn.__version__, "scipy": scipy.__version__, "numpy": np.__version__, "python": platform.python_version()},
       "wilcoxon": {m: {"diff": float((acc["LR"] - acc[m]).mean()), "p": float(wilcoxon(acc["LR"], acc[m]).pvalue),
                        "wins": int((acc["LR"] > acc[m]).sum()), "ties": int((acc["LR"] == acc[m]).sum()), "n": len(cut) + 1}
                    for m in acc if m != "LR"},
       "text_dim": int(TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True).fit(X).transform(X).shape[1])}
for key, loader in [("motor", lambda: ds.thermal("ex_motor")), ("transformer", lambda: ds.thermal("ex_trans"))]:
    Xi, yi, _, _ = loader()
    out[key + "_shape"] = list(Xi.shape)
    out[key + "_counts"] = {k: int(v) for k, v in collections.Counter(yi.tolist()).items()}
Xv, yv, _, _, ld = ds.vibration()
out["cwru_shape"] = list(Xv.shape)
out["cwru_counts"] = {k: int(v) for k, v in collections.Counter(yv.tolist()).items()}
out["cwru_load_counts"] = {int(k): int(v) for k, v in collections.Counter(ld.tolist()).items()}
json.dump(out, open(RESULTS / "extra.json", "w"), indent=1)
print(json.dumps(out["wilcoxon"], indent=1))
