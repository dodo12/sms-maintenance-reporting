"""Negative controls for the CWRU vibration benchmark under leave-one-load-out.

Permuted labels and feature-group ablations (time-domain / spectrum / envelope spectrum). Output: results/controls_vibration.json
"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from sklearn.metrics import accuracy_score
from smsreport.common import make_pipeline, LogisticRegression, StandardScaler
from smsreport.datasets import vibration
from smsreport.paths import RESULTS

X, y, _, _, ld = vibration()


def lolo(F, labels):
    a = []
    for l in range(4):
        tr, te = ld != l, ld == l
        m = make_pipeline(StandardScaler(), LogisticRegression(C=1, max_iter=3000)).fit(F[tr], labels[tr])
        a.append(accuracy_score(labels[te], m.predict(F[te])))
    return float(np.mean(a))


rng = np.random.default_rng(0)
cnt = dict(zip(*np.unique(y, return_counts=True)))
out = {"windows": int(len(y)), "classes": sorted(set(y.tolist())), "windows_per_class": {k: int(v) for k, v in cnt.items()},
       "chance": 1 / len(set(y.tolist())), "majority": float(max(cnt.values()) / len(y)),
       "full_features": lolo(X, y), "permuted_labels_mean_of_3": float(np.mean([lolo(X, rng.permutation(y)) for _ in range(3)])),
       "time_domain_only": lolo(X[:, :10], y), "spectrum_only": lolo(X[:, 10:74], y), "envelope_only": lolo(X[:, 74:], y)}
print(json.dumps(out, indent=1))
json.dump(out, open(RESULTS / "controls_vibration.json", "w"), indent=1)
