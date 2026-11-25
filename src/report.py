"""Tạo hình và số liệu cho README từ mô hình đã train.

    .venv/Scripts/python.exe src/report.py      (cần requirements-dev.txt)

Dùng mô hình chỉ học tập train (không phải bản refit), nên số trên test là trung thực.
Chỉ để báo cáo: không có lựa chọn nào của mô hình dựa vào kết quả ở đây.
"""

import sys

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

import train

DOCS = train.ROOT / "docs"
BLUE, INK, INK2, MUTED, GRID, SURFACE = "#2a78d6", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 10, "text.color": INK,
    "axes.labelcolor": INK2, "axes.edgecolor": "#c3c2b7", "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.titlelocation": "left",
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.axisbelow": True,
    "grid.color": GRID, "grid.linewidth": 0.8,
})


def heatmap(df):
    """Tỷ lệ churn theo hợp đồng và Internet, chỉ trên tập train."""
    g = df.assign(y=df["Churn"] == "Yes").pivot_table(
        index="InternetService", columns="Contract", values="y", aggfunc=["mean", "size"]
    ).loc[["Fiber optic", "DSL", "No"]]
    p, n = g["mean"], g["size"]
    cmap = LinearSegmentedColormap.from_list("b", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
    fig, ax = plt.subplots(figsize=(6.5, 3))
    ax.pcolormesh(p.to_numpy(dtype=float), cmap=cmap, vmin=0, vmax=0.6, edgecolors=SURFACE, linewidth=2)
    for (i, j), v in np.ndenumerate(p.to_numpy(dtype=float)):
        ax.text(j + 0.5, i + 0.5, f"{v:.0%}\nn={n.iloc[i, j]}", ha="center", va="center",
                color="white" if v > 0.25 else INK)
    ax.set(xticks=np.arange(3) + 0.5, yticks=np.arange(3) + 0.5)
    ax.set_xticklabels(p.columns)
    ax.set_yticklabels(["Fiber optic", "DSL", "Không Internet"])
    ax.invert_yaxis()
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("Tỷ lệ churn theo hợp đồng và loại Internet (tập train)")
    fig.tight_layout()
    fig.savefig(DOCS / "churn_heatmap.png", dpi=150)


def gains(y, proba):
    """Liên hệ k% khách rủi ro nhất thì gặp được bao nhiêu % khách sẽ rời đi?"""
    order = np.argsort(-proba, kind="stable")
    captured = np.r_[0, np.cumsum(y[order])] / y.sum() * 100
    contacted = np.arange(len(y) + 1) / len(y) * 100
    table = []
    for k in (10, 20, 30, 40):
        m = round(len(y) * k / 100)
        hit = int(y[order][:m].sum())
        table.append((k, m, hit, hit / m, hit / y.sum(), hit / m / y.mean()))

    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.plot([0, 100], [0, 100], color=MUTED, lw=1.5)
    ax.plot(contacted, captured, color=BLUE, lw=2)
    ax.text(72, 66, "Gọi ngẫu nhiên", color=INK2)
    ax.text(50, 91, "Mô hình", color=INK2, ha="right")
    k, _, _, _, cap, _ = table[1]
    ax.plot([k], [cap * 100], "o", ms=8, color=BLUE, mec=SURFACE, mew=2)
    ax.text(1.5, cap * 100 + 13, f"Gọi {k}% khách,\ngặp {cap:.0%} khách churn", color=INK)
    ax.set(xlabel="% khách được liên hệ (rủi ro cao trước)", ylabel="% khách churn gặp được",
           xlim=(0, 100), ylim=(0, 102))
    ax.grid(axis="y")
    ax.set_title("Gọi đúng người: mô hình so với gọi ngẫu nhiên (tập test)")
    fig.tight_layout()
    fig.savefig(DOCS / "gains.png", dpi=150)
    return table


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    DOCS.mkdir(exist_ok=True)
    bundle = joblib.load(train.MODEL_DIR / "model.pkl")
    spec = train.CANDIDATES[bundle["meta"]["candidate"]]
    cols = spec["numeric"] + spec["categorical"]

    df = train.load_data()
    y = (df[train.TARGET] == "Yes").astype(int)
    tr, te = train_test_split(df, test_size=0.2, random_state=train.RANDOM_STATE, stratify=y)
    pipe = train.build_pipeline(**spec).fit(tr[cols], y[tr.index])
    proba, yt = pipe.predict_proba(te[cols])[:, 1], y[te.index].to_numpy()
    auc = round(float(roc_auc_score(yt, proba)), 4)
    if auc != bundle["metrics"]["roc_auc"]:
        raise SystemExit(f"ROC-AUC {auc} không khớp model.pkl ({bundle['metrics']['roc_auc']}): hãy train lại")

    heatmap(tr)
    print("k%  số gọi  trúng  precision  recall  lift")
    for k, m, hit, prec, rec, lift in gains(yt, proba):
        print(f"{k:>2}  {m:>6}  {hit:>5}  {prec:>9.1%}  {rec:>6.1%}  {lift:>4.2f}")

    print("\nHiệu chuẩn: xác suất dự đoán và thực tế theo 5 nhóm bằng nhau")
    for part in np.array_split(np.argsort(proba, kind="stable"), 5):
        print(f"  dự đoán {proba[part].mean():.1%}  thực tế {yt[part].mean():.1%}  (n={len(part)})")

    rng, boot = np.random.default_rng(0), []
    for _ in range(2000):
        i = rng.integers(0, len(yt), len(yt))
        boot.append(roc_auc_score(yt[i], proba[i]))
    print(f"\nROC-AUC {auc}, KTC 95%: {np.percentile(boot, 2.5):.3f}–{np.percentile(boot, 97.5):.3f}")


if __name__ == "__main__":
    main()
