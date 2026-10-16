"""Negative controls for the thermal-image benchmarks (same blocked protocol as the main experiment).

(i) permuted labels  (ii) four 40x30 corner patches only  (iii) five global colour statistics  (iv) 32x24 RGB thumbnail.
Output: results/controls_images.json
"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score
from smsreport.common import make_pipeline, LogisticRegression, StandardScaler
from smsreport.datasets import thermal_blocks
from smsreport.paths import THERMAL_DIR, RESULTS


def blocked_cv(F, y, blk):
    a = []
    for k in range(5):
        tr, te = blk != k, blk == k
        m = make_pipeline(StandardScaler(), LogisticRegression(C=1, max_iter=3000)).fit(F[tr], y[tr])
        a.append(accuracy_score(y[te], m.predict(F[te])))
    return float(np.mean(a)), float(np.std(a, ddof=1))


rng = np.random.default_rng(0)
out = {}
for name, sub in [("motor", "ex_motor"), ("transformer", "ex_trans")]:
    P, y, blk = thermal_blocks(str(THERMAL_DIR / sub))
    A = [np.asarray(Image.open(p).convert("RGB")) / 255.0 for p in P]
    thumb = np.vstack([np.asarray(Image.fromarray((a * 255).astype(np.uint8)).resize((32, 24))).ravel() / 255.0 for a in A])
    corners = np.vstack([np.concatenate([a[:30, :40].ravel(), a[:30, -40:].ravel(), a[-30:, :40].ravel(), a[-30:, -40:].ravel()])[::7] for a in A])
    glob_stats = np.array([[a.mean(), a.std(), a[..., 0].mean(), a[..., 1].mean(), a[..., 2].mean()] for a in A])
    out[name] = {"n": len(y), "classes": len(set(y.tolist())), "chance": 1 / len(set(y.tolist())),
                 "majority": float(max(np.unique(y, return_counts=True)[1]) / len(y)),
                 "permuted_labels_mean_of_5": float(np.mean([blocked_cv(thumb, rng.permutation(y), blk)[0] for _ in range(5)])),
                 "corners_only": blocked_cv(corners, y, blk), "global_colour_stats": blocked_cv(glob_stats, y, blk),
                 "thumbnail_32x24_rgb": blocked_cv(thumb, y, blk)}
    print(name, json.dumps(out[name]))
json.dump(out, open(RESULTS / "controls_images.json", "w"), indent=1)
