"""Evaluation framework: nested-threshold selective reporting, calibration, top-k coverage."""
import warnings
import numpy as np
from sklearn.linear_model import LogisticRegression          # noqa: F401  (re-exported for experiments)
from sklearn.pipeline import make_pipeline                   # noqa: F401
from sklearn.preprocessing import StandardScaler             # noqa: F401
from sklearn.metrics import f1_score
from .paths import RESULTS

warnings.filterwarnings("ignore")
TARGET = 0.90      # target accuracy alpha of reported diagnoses (fixed a priori)
MIN_N = 20         # minimum support n_min


def ece(conf, ok, M=10):
    """Expected calibration error with M equal-width confidence bins."""
    b = np.linspace(0, 1, M + 1)
    e = 0.0
    for a, c in zip(b[:-1], b[1:]):
        s = (conf > a) & (conf <= c)
        if s.sum():
            e += s.mean() * abs(ok[s].mean() - conf[s].mean())
    return float(e)


def choose_tau(conf, ok, target=TARGET, min_n=MIN_N):
    """Smallest confidence threshold whose inner-validation accuracy >= target with support >= min_n."""
    o = np.argsort(-conf)
    c = conf[o]
    a = np.cumsum(ok[o]) / np.arange(1, len(o) + 1)
    idx = [i for i in range(len(o)) if a[i] >= target and i + 1 >= min_n]
    return float(c[max(idx)]) if idx else np.inf


def run(name, X, y, make_model, outer, inner_fn, topk_max=3):
    """Outer evaluation with nested threshold selection.

    X: array or list indexable by integer arrays; y: labels;
    make_model: callable returning an unfitted estimator with predict_proba;
    outer: list of (train_idx, test_idx); inner_fn(train_idx) -> list of (train_in, val_in) index pairs.
    """
    y = np.asarray(y)
    classes = sorted(set(y.tolist()))
    C = len(classes)
    sel = lambda idx: ([X[i] for i in idx] if isinstance(X, list) else X[idx])
    confs, oks, folds = [], [], []
    for tr, te in outer:
        m = make_model().fit(sel(tr), y[tr])
        P = m.predict_proba(sel(te))
        cl = list(m.classes_)
        pred = np.array([cl[i] for i in P.argmax(1)])
        conf = P.max(1)
        ok = pred == y[te]
        ic, io = [], []
        for a, b in inner_fn(tr):                      # inner validation: threshold selection only
            mi = make_model().fit(sel(a), y[a])
            Pi = mi.predict_proba(sel(b))
            ci = list(mi.classes_)
            pi = np.array([ci[i] for i in Pi.argmax(1)])
            ic += Pi.max(1).tolist()
            io += (pi == y[b]).tolist()
        tau = choose_tau(np.array(ic), np.array(io))
        rep = conf >= tau
        order = np.argsort(-P, axis=1)
        tk = {k: float(np.mean([y[te][i] in [cl[j] for j in order[i, :k]] for i in range(len(te))]))
              for k in range(1, min(topk_max, C) + 1)}
        folds.append({"acc": float(ok.mean()), "macroF1": float(f1_score(y[te], pred, average="macro")),
                      "tau": tau if np.isfinite(tau) else None, "cov": float(rep.mean()),
                      "sel_acc": float(ok[rep].mean()) if rep.sum() else None,
                      "n_test": int(len(te)), "topk": tk})
        confs.append(conf)
        oks.append(ok)
    conf = np.concatenate(confs)
    ok = np.concatenate(oks)
    o = np.argsort(-conf)
    risk = 1 - np.cumsum(ok[o]) / np.arange(1, len(o) + 1)
    cov = np.arange(1, len(o) + 1) / len(o)
    acc_at = {str(c): float(ok[o][:int(round(c * len(o)))].mean()) for c in [1, .9, .8, .7, .6, .5]}
    nrep = sum(f["cov"] * f["n_test"] for f in folds)
    nsel = sum((f["sel_acc"] or 0) * f["cov"] * f["n_test"] for f in folds)
    res = {"name": name, "n": int(len(conf)), "C": C, "bits_per_label": int(np.ceil(np.log2(C))),
           "acc_mean": float(np.mean([f["acc"] for f in folds])), "acc_sd": float(np.std([f["acc"] for f in folds], ddof=1)),
           "f1_mean": float(np.mean([f["macroF1"] for f in folds])), "f1_sd": float(np.std([f["macroF1"] for f in folds], ddof=1)),
           "pooled_acc": float(ok.mean()), "aurc": float(risk.mean()), "ece": ece(conf, ok), "acc_at_cov": acc_at,
           "nested_cov": float(nrep / len(conf)), "nested_sel_acc": float(nsel / nrep) if nrep > 0 else None,
           "topk": {k: float(np.mean([f["topk"][k] for f in folds])) for k in folds[0]["topk"]}, "folds": folds}
    cdir = RESULTS / "curves"
    np.save(cdir / f"{name}_cov.npy", cov)
    np.save(cdir / f"{name}_risk.npy", risk)
    np.save(cdir / f"{name}_confok.npy", np.vstack([conf, ok]))
    return res
