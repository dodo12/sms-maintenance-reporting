"""Fast end-to-end check (a few minutes): loads the 3 data types, verifies expected counts, runs logistic regression on 2 outer folds."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from smsreport import datasets as ds
from smsreport.common import run, make_pipeline, LogisticRegression, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer

exp = {"text": (626, 22), "motor": (369, 11), "trans": (255, 9)}
X, y, outer, inner = ds.text()
assert (len(y), len(set(y))) == exp["text"], (len(y), len(set(y)))
r = run("smoke_text", X, y, lambda: make_pipeline(TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True), LogisticRegression(C=100, max_iter=2000)), outer[:2], inner)
print("text   acc %.3f ECE %.3f" % (r["acc_mean"], r["ece"]))
for key, sub in [("motor", "ex_motor"), ("trans", "ex_trans")]:
    X, y, outer, inner = ds.thermal(sub)
    assert (len(y), len(set(y))) == exp[key], (key, len(y), len(set(y)))
    r = run(f"smoke_{key}", X, y, lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1, max_iter=3000)), outer[:2], inner)
    print("%-6s acc %.3f ECE %.3f" % (key, r["acc_mean"], r["ece"]))
X, y, outer, inner, ld = ds.vibration()
assert X.shape == (3897, 106), X.shape
r = run("smoke_cwru", X, y, lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1, max_iter=3000)), outer[:2], inner)
print("cwru   acc %.3f ECE %.3f" % (r["acc_mean"], r["ece"]))
print("SMOKE TEST OK")
