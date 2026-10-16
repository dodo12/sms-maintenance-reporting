"""Loaders returning (X, y, outer_splits, inner_fn) for each benchmark, with leakage-aware protocols.

text      : repeated stratified 5-fold (x3); inner stratified 5-fold
thermal   : 5 contiguous blocks per class (neighbouring captures stay in the same fold); inner leave-one-block-out
vibration : leave-one-load-out; all windows of a recording share its load; inner leave-one-load-out
"""
import glob, os, re
import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from .paths import FMC_DIR, THERMAL_DIR, CWRU_DIR, CACHE


def text():
    def load(f):
        return [l.rstrip("\n").rsplit(",", 1) for l in open(FMC_DIR / f, encoding="utf8") if l.strip()]
    rows = load("train.txt") + load("dev.txt") + load("test.txt")
    X = [r[0].strip().lower() for r in rows]
    y = np.array([r[1].strip() for r in rows])
    outer = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1).split(np.zeros(len(y)), y))
    inner = lambda tr: [(tr[a], tr[b]) for a, b in
                        StratifiedKFold(5, shuffle=True, random_state=0).split(np.zeros(len(tr)), y[tr])]
    return X, y, outer, inner


def thermal_blocks(root):
    """Return (paths, labels, block ids). Images of a class are ordered by file index and cut into 5 contiguous blocks."""
    P, y, idx = [], [], []
    for c in sorted(os.listdir(root)):
        d = os.path.join(root, c)
        if os.path.isdir(d):
            for f in os.listdir(d):
                if f.lower().endswith(".bmp"):
                    P.append(os.path.join(d, f)); y.append(c); idx.append(int(re.sub(r"\D", "", f) or 0))
    y, idx = np.array(y), np.array(idx)
    blk = np.zeros(len(y), int)
    for c in set(y.tolist()):
        ii = np.where(y == c)[0]
        o = ii[np.argsort(idx[ii])]
        for r, i in enumerate(o):
            blk[i] = min(4, int(r * 5 / len(o)))
    return P, y, blk


def thermal(subdir):
    """subdir: 'ex_motor' or 'ex_trans' (extracted from the .rar archives of the thermal-image repository)."""
    root = THERMAL_DIR / subdir
    cache = CACHE / f"feat_{subdir}.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True); X, y, blk = z["X"], z["y"], z["blk"]
    else:
        from PIL import Image
        from skimage.feature import hog
        from skimage.color import rgb2hsv, rgb2gray

        def imfeat(p):
            a = np.asarray(Image.open(p).convert("RGB")) / 255.0
            hsv = rgb2hsv(a)
            h = np.histogramdd(hsv.reshape(-1, 3), bins=(8, 8, 8), range=((0, 1),) * 3)[0].ravel()
            h = np.sqrt(h / h.sum())
            g8 = (rgb2gray(a) * 255).astype(np.uint8)
            small = np.asarray(Image.fromarray(g8).resize((32, 24))) / 255.0
            hg = hog(np.asarray(Image.fromarray(g8).resize((160, 120))) / 255.0, orientations=9,
                     pixels_per_cell=(16, 16), cells_per_block=(2, 2))
            return np.concatenate([h, small.ravel(), hg])          # 512 + 768 + 1944 = 3224 features
        P, y, blk = thermal_blocks(str(root))
        X = np.vstack([imfeat(p) for p in P])
        np.savez(cache, X=X, y=y, blk=blk)
    outer = [(np.where(blk != k)[0], np.where(blk == k)[0]) for k in range(5)]
    inner = lambda tr: [(tr[blk[tr] != k], tr[blk[tr] == k]) for k in sorted(set(blk[tr].tolist()))]
    return X, y, outer, inner


def _vib_features(x):
    from scipy.signal import hilbert
    from scipy.stats import skew, kurtosis
    t = [x.mean(), x.std(), np.sqrt((x ** 2).mean()), np.abs(x).max(), np.ptp(x), skew(x), kurtosis(x)]
    rms = t[2]
    t += [np.abs(x).max() / rms, rms / np.abs(x).mean(), np.abs(x).max() / np.abs(x).mean()]
    S = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
    fr = np.fft.rfftfreq(len(x), 1 / 12000)
    be = np.linspace(0, 6000, 65)
    sp = [np.log10(S[(fr >= a) & (fr < b)].sum() + 1e-12) for a, b in zip(be[:-1], be[1:])]
    env = np.abs(hilbert(x)); env -= env.mean()
    E = np.abs(np.fft.rfft(env * np.hanning(len(env)))) ** 2
    fe = np.fft.rfftfreq(len(env), 1 / 12000)
    bb = np.linspace(0, 1000, 33)
    es = [np.log10(E[(fe >= a) & (fe < b)].sum() + 1e-12) for a, b in zip(bb[:-1], bb[1:])]
    return np.array(t + sp + es)                                   # 10 + 64 + 32 = 106 features


def vibration(window=2048):
    cache = CACHE / "feat_cwru.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True); X, y, ld = z["X"], z["y"], z["ld"]
    else:
        import scipy.io as sio
        R = str(CWRU_DIR) + os.sep
        files = [(f, "Normal") for f in glob.glob(R + "Normal/*.mat")]
        for t in ["IR", "B", "OR"]:
            pat = R + f"12k_Drive_End_Bearing_Fault_Data/{t}/"
            for f in glob.glob(pat + "*/*.mat") + glob.glob(pat + "*/*/*.mat"):
                s = f.split(f"/{t}/")[1].split("/")[0]
                if s in ("007", "014", "021"):                     # 0.028 inch faults excluded
                    files.append((f, f"{t}_{s}"))
        F, Y, L = [], [], []
        for f, c in sorted(files):
            m = sio.loadmat(f)
            k = [k for k in m if k.endswith("DE_time")][0]
            x = m[k].ravel().astype(float)
            load = int(re.findall(r"_(\d)\.mat$", f)[0])
            for s in range(0, len(x) - window + 1, window):
                F.append(_vib_features(x[s:s + window])); Y.append(c); L.append(load)
        X, y, ld = np.vstack(F), np.array(Y), np.array(L)
        np.savez(cache, X=X, y=y, ld=ld)
    outer = [(np.where(ld != l)[0], np.where(ld == l)[0]) for l in range(4)]
    inner = lambda tr: [(tr[ld[tr] != l], tr[ld[tr] == l]) for l in sorted(set(ld[tr].tolist()))]
    return X, y, outer, inner, ld
