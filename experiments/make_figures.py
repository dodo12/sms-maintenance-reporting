"""Regenerate the figures from results/*.json and results/curves/*.npy  ->  results/figures/*.png|pdf"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.patches as mp
from PIL import Image
from smsreport.paths import RESULTS, THERMAL_DIR

FIG = RESULTS / "figures"; FIG.mkdir(exist_ok=True)
plt.rcParams.update({"font.size": 9, "font.family": "serif", "axes.grid": True, "grid.alpha": .3, "figure.dpi": 200})
R = json.load(open(RESULTS / "all_models.json"))
DS = [("text_FMC", "Text", "C0"), ("image_motor", "Thermal image, motor", "C1"),
      ("image_transformer", "Thermal image, transformer", "C2"), ("vibration_CWRU", "Vibration (CWRU)", "C3")]


def save(fig, name):
    fig.tight_layout(); fig.savefig(FIG / f"{name}.png"); fig.savefig(FIG / f"{name}.pdf"); plt.close(fig)


def framework():
    fig, ax = plt.subplots(figsize=(7.4, 3.1)); ax.axis("off"); ax.set_xlim(0, 15); ax.set_ylim(0, 6.4)
    def box(x, y, w, h, t, c="#e8eef7"):
        ax.add_patch(mp.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05", fc=c, ec="k", lw=.8))
        ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=7)
    def arr(x1, y1, x2, y2, t=""):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", lw=.9))
        if t: ax.text((x1 + x2) / 2, (y1 + y2) / 2 + .28, t, ha="center", fontsize=6)
    box(.1, 3.9, 2.7, 1.9, "Observation $x_m$\n(m = text, thermal\nimage, vibration)", "#fdf0d5")
    box(3.5, 3.9, 2.7, 1.9, "Modality-specific\nfeatures $\\phi_m(x)$")
    box(6.9, 3.9, 2.7, 1.9, "Classifier $\\hat p(y|x)$\n(any model family)")
    box(10.3, 3.9, 2.1, 1.9, "Confidence\n$c=\\max_y \\hat p$")
    box(12.9, 3.9, 2.0, 1.9, "Threshold $\\hat\\tau$\n(nested validation)", "#f9dcdc")
    box(10.3, .5, 2.1, 1.9, "Report top-$k$ codes\n$k\\lceil\\log_2 C\\rceil$ bits", "#e2f3e0")
    box(12.9, .5, 2.0, 1.9, "Abstain /\nask for more input", "#f9dcdc")
    box(6.9, .5, 2.7, 1.9, "SMS payload\n$\\leq$1120 bits;\nlossy channel $p_\\ell$")
    box(3.5, .5, 2.7, 1.9, "Technician reads\ncode, confirms or\nescalates", "#fdf0d5")
    for a in [(2.8, 4.85, 3.5, 4.85), (6.2, 4.85, 6.9, 4.85), (9.6, 4.85, 10.3, 4.85), (12.4, 4.85, 12.9, 4.85),
              (10.3, 1.45, 9.6, 1.45), (6.9, 1.45, 6.2, 1.45)]: arr(*a)
    arr(11.35, 3.9, 11.35, 2.4, "$c\\geq\\hat\\tau$"); arr(13.9, 3.9, 13.9, 2.4, "$c<\\hat\\tau$")
    save(fig, "framework")


def models_fig():
    mods = ["LR", "kNN", "RandomForest", "SVM", "MLP", "HistGB"]; lab = ["LR", "k-NN", "RF", "SVM (Platt)", "MLP", "HistGB"]
    get = lambda d, m: next((v for k, v in d.items() if k == m or (m == "SVM" and k.startswith("SVM"))), None)
    fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.2)); w = .13
    for j, (m, l) in enumerate(zip(mods, lab)):
        for i, (k, _, _) in enumerate(DS):
            v = get(R[k], m)
            if v is None: continue
            ax[0].bar(i + (j - 2.5) * w, v["acc_mean"], w, yerr=v["acc_sd"], color=f"C{j}", capsize=1.5, error_kw={"lw": .6})
            ax[1].bar(i + (j - 2.5) * w, v["ece"], w, color=f"C{j}", label=l if i == 0 else None)
    for a, t, yl in [(ax[0], "(a) Accuracy (mean ± SD over folds)", "Accuracy"), (ax[1], "(b) Expected calibration error", "ECE")]:
        a.set_xticks(range(4)); a.set_xticklabels([n for _, n, _ in DS], fontsize=6.5); a.set_title(t, fontsize=9); a.set_ylabel(yl)
    ax[0].set_ylim(.55, 1.05); ax[1].legend(fontsize=6.5, ncol=2, loc="upper right")
    save(fig, "models")


def multi():
    fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.2))
    for k, l, c in DS:
        cv = RESULTS / "curves"
        cov, risk = np.load(cv / f"{k}__LR_cov.npy"), np.load(cv / f"{k}__LR_risk.npy")
        ax[0].plot(cov, risk + 1e-4, label=l, color=c)
        v = R[k]["LR"]; b = v["bits_per_label"]; ks = sorted(int(i) for i in v["topk"])
        ax[1].plot([i * b for i in ks], [v["topk"][str(i)] for i in ks], "o-", ms=3, label=l, color=c)
        cf, ok = np.load(cv / f"{k}__LR_confok.npy"); bins = np.linspace(0, 1, 11); xs, ys = [], []
        for a, bb in zip(bins[:-1], bins[1:]):
            s = (cf > a) & (cf <= bb)
            if s.sum() >= 5: xs.append(cf[s].mean()); ys.append(ok[s].mean())
        ax[2].plot(xs, ys, "o-", ms=3, color=c, label=l)
    ax[0].set_yscale("log"); ax[0].set_xlabel("Coverage"); ax[0].set_ylabel("Selective risk (+1e-4, log)"); ax[0].set_title("(a) Risk–coverage", fontsize=9); ax[0].legend(fontsize=6)
    ax[1].set_xlabel("Payload (bits)"); ax[1].set_ylabel("Top-$k$ coverage of true label"); ax[1].set_title("(b) Top-$k$ vs bits", fontsize=9)
    ax[2].plot([0, 1], [0, 1], "k--", lw=.8); ax[2].set_xlabel("Mean confidence"); ax[2].set_ylabel("Empirical accuracy"); ax[2].set_title("(c) Reliability", fontsize=9)
    save(fig, "multi_modality")


def samples():
    ps = [THERMAL_DIR / "ex_motor/Noload/007.bmp", THERMAL_DIR / "ex_motor/A50/262.bmp",
          THERMAL_DIR / "ex_motor/Fan/f090.bmp", THERMAL_DIR / "ex_trans/p9_600/p9089.bmp"]
    ims = [Image.open(p).convert("RGB") for p in ps]; w, h = ims[0].size
    m = Image.new("RGB", (w * 4, h)); [m.paste(i, (k * w, 0)) for k, i in enumerate(ims)]; m.save(FIG / "thermal_samples.png")


def text_figs():
    t = RESULTS / "text_study"
    if not (t / "noise.json").exists(): return
    nz = json.load(open(t / "noise.json")); lv = ["0", "0.05", "0.1", "0.2", "0.3"]; x = [float(l) for l in lv]
    fig, ax = plt.subplots(figsize=(3.6, 3))
    for k, l, mk in [("clean-train", "Char LR, clean training", "o-"), ("aug-train", "Char LR, noise-augmented", "s-"), ("word-clean", "Word LR, clean training", "^--")]:
        ax.errorbar(x, [nz[k][v][0] for v in lv], yerr=[nz[k][v][1] for v in lv], fmt=mk, ms=3, capsize=2, label=l)
    ax.set_xlabel("Character-noise rate $\\rho$"); ax.set_ylabel("Accuracy"); ax.legend(fontsize=6.5); save(fig, "text_noise")
    labs = json.load(open(t / "labs.json")); Cm = np.load(t / "conf.npy"); Cn = Cm / Cm.sum(1, keepdims=True).clip(1)
    fig, ax = plt.subplots(figsize=(6.4, 5.4)); im = ax.imshow(Cn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labs))); ax.set_yticks(range(len(labs))); ax.set_xticklabels(labs, rotation=90, fontsize=6); ax.set_yticklabels(labs, fontsize=6)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.grid(False); fig.colorbar(im, fraction=.04); save(fig, "text_confusion")


if __name__ == "__main__":
    framework(); models_fig(); multi(); samples(); text_figs(); print("figures written to", FIG)
