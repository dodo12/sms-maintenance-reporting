"""Main experiment: reporting methodology x 6 model families x 4 benchmarks (identical protocols).

Usage:  python experiments/run_models.py [--only text_FMC|image_motor|image_transformer|vibration_CWRU]
Output: results/all_models.json  (or results/models_<only>.json) and results/curves/*.npy
Runtime: roughly 40-60 min on a laptop CPU (gradient boosting on image features is the slowest part).
"""
import argparse, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from smsreport.common import run, make_pipeline, LogisticRegression, StandardScaler
from smsreport import datasets as ds
from smsreport.paths import RESULTS


def zoo(kind):
    if kind == "text":
        tf = lambda: TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True)
        return {"LR": lambda: make_pipeline(tf(), LogisticRegression(C=100, max_iter=2000)),
                "kNN": lambda: make_pipeline(tf(), KNeighborsClassifier(5, metric="cosine", weights="distance")),
                "RandomForest": lambda: make_pipeline(tf(), RandomForestClassifier(200, random_state=0, n_jobs=-1)),
                "SVM-linear(Platt)": lambda: make_pipeline(tf(), SVC(kernel="linear", C=10, probability=True, random_state=0)),
                "MLP": lambda: make_pipeline(tf(), MLPClassifier((128,), max_iter=300, random_state=0))}
    sc = StandardScaler
    return {"LR": lambda: make_pipeline(sc(), LogisticRegression(C=1, max_iter=3000)),
            "kNN": lambda: make_pipeline(sc(), KNeighborsClassifier(5, weights="distance")),
            "RandomForest": lambda: RandomForestClassifier(300, random_state=0, n_jobs=-1),
            "SVM-RBF(Platt)": lambda: make_pipeline(sc(), SVC(kernel="rbf", C=10, probability=True, random_state=0)),
            "MLP": lambda: make_pipeline(sc(), MLPClassifier((128,), max_iter=500, random_state=0)),
            "HistGB": lambda: HistGradientBoostingClassifier(max_iter=100, random_state=0)}


SETS = [("text_FMC", "text", ds.text),
        ("image_motor", "x", lambda: ds.thermal("ex_motor")),
        ("image_transformer", "x", lambda: ds.thermal("ex_trans")),
        ("vibration_CWRU", "x", lambda: ds.vibration()[:4])]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    ap.add_argument("--max-folds", type=int, default=None, help="debug: limit number of outer folds")
    a = ap.parse_args()
    out = RESULTS / (f"models_{a.only}.json" if a.only else "all_models.json")
    RES = json.load(open(out)) if out.exists() else {}
    for nm, kind, fn in SETS:
        if a.only and nm != a.only:
            continue
        X, y, outer, inner = fn()
        if a.max_folds:
            outer = outer[:a.max_folds]
        RES.setdefault(nm, {})
        for mn, mk in zoo(kind).items():
            if mn in RES[nm] and not a.max_folds:
                continue                                          # resume support
            t0 = time.time()
            r = run(f"{nm}__{mn}", X, y, mk, outer, inner)
            r.pop("folds", None)
            RES[nm][mn] = r
            print(f"{nm:18s} {mn:18s} acc {r['acc_mean']:.3f} F1 {r['f1_mean']:.3f} AURC {r['aurc']:.4f} "
                  f"ECE {r['ece']:.3f} nested {r['nested_cov']:.3f}->{r['nested_sel_acc']} ({time.time()-t0:.0f}s)", flush=True)
            if not a.max_folds:
                json.dump(RES, open(out, "w"), indent=1, default=str)
