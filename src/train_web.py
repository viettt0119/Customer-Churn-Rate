import hashlib
import json
import platform
import sys
from datetime import datetime, timezone

import imblearn
import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from imblearn.over_sampling import SMOTENC
from imblearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
    train_test_split,
)
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

import train

MODEL_DIR = train.MODEL_DIR
SPEC = train.CANDIDATES["addons"]
NUM, CAT = SPEC["numeric"], SPEC["categorical"]
COLS = NUM + CAT
SEED = train.RANDOM_STATE


def preprocessor(scale):
    return ColumnTransformer([
        ("num", StandardScaler() if scale else "passthrough", NUM),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT),
    ])


def xgb(**params):
    return XGBClassifier(
        n_estimators=params.pop("n_estimators", 300),
        max_depth=params.pop("max_depth", 3),
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_lambda=2,
        random_state=SEED,
        n_jobs=1,
        eval_metric="logloss",
        **params,
    )


def smote():
    return SMOTENC(categorical_features=CAT + ["SeniorCitizen"], random_state=SEED)


def candidates(pos_weight):
    lr = LogisticRegression(max_iter=2000)
    return {
        "lr": Pipeline([("pre", preprocessor(True)), ("clf", lr)]),
        "lr_smote": Pipeline([("smote", smote()), ("pre", preprocessor(True)), ("clf", lr)]),
        "xgb_depth2": Pipeline([("pre", preprocessor(False)), ("clf", xgb(max_depth=2, n_estimators=500))]),
        "xgb_depth3": Pipeline([("pre", preprocessor(False)), ("clf", xgb())]),
        "xgb_depth4": Pipeline([("pre", preprocessor(False)), ("clf", xgb(max_depth=4, n_estimators=200))]),
        "xgb_weighted": Pipeline([("pre", preprocessor(False)), ("clf", xgb(scale_pos_weight=pos_weight))]),
        "xgb_smote": Pipeline([("smote", smote()), ("pre", preprocessor(False)), ("clf", xgb())]),
    }


def compare(models, x, y):
    k = 5
    cv = RepeatedStratifiedKFold(n_splits=k, n_repeats=3, random_state=SEED)
    scoring = ["average_precision", "roc_auc", "neg_brier_score"]
    res = {n: cross_validate(m, x, y, cv=cv, scoring=scoring) for n, m in models.items()}
    key = "test_average_precision"
    best = max(res, key=lambda n: res[n][key].mean())
    rows, chosen = [], None
    for name, r in res.items():
        gap = res[best][key] - r[key]
        se = float(np.sqrt((1 / len(gap) + 1 / (k - 1)) * gap.var(ddof=1)))
        ok = bool(gap.mean() <= se)
        if ok and chosen is None:
            chosen = name
        rows.append({
            "name": name,
            "pr_auc": round(float(r[key].mean()), 4),
            "roc_auc": round(float(r["test_roc_auc"].mean()), 4),
            "brier": round(float(-r["test_neg_brier_score"].mean()), 4),
            "within_1se": ok,
        })
    return chosen, rows


def test_metrics(y, proba, threshold):
    return {
        "roc_auc": round(float(roc_auc_score(y, proba)), 4),
        "pr_auc": round(float(average_precision_score(y, proba)), 4),
        "brier": round(float(brier_score_loss(y, proba)), 4),
        **train.evaluate(y, proba, threshold),
    }


def gains(y, proba):
    hits = np.cumsum(np.asarray(y)[np.argsort(-proba, kind="stable")])
    steps = np.arange(0, 101, 5)
    idx = np.round(steps / 100 * len(hits)).astype(int)
    return [round(float(hits[i - 1] / hits[-1]) * 100, 1) if i else 0.0 for i in idx]


def group_of(feature):
    return feature.split("__", 1)[1] if feature.startswith("num__") else feature.split("__", 1)[1].split("_", 1)[0]


def eda(x, y):
    d = x.assign(y=y)
    labels = ["0–6", "7–12", "13–24", "25–36", "37–48", "49–60", "61–72"]
    d["bin"] = pd.cut(d["tenure"], [-1, 6, 12, 24, 36, 48, 60, 72], labels=labels)
    t = d.groupby("bin", observed=False)["y"].agg(["mean", "size"])
    g = d.pivot_table(index="InternetService", columns="Contract", values="y", aggfunc=["mean", "size"])
    rows = ["No", "DSL", "Fiber optic"]
    cols = ["Month-to-month", "One year", "Two year"]
    return {
        "tenure": {"labels": labels, "rate": (t["mean"] * 100).round(1).tolist(), "n": t["size"].tolist()},
        "heatmap": {
            "rows": rows,
            "cols": cols,
            "rate": (g["mean"].loc[rows, cols] * 100).round(1).values.tolist(),
            "n": g["size"].loc[rows, cols].values.tolist(),
        },
    }


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    df = train.load_data()
    x, y = df[COLS], (df[train.TARGET] == "Yes").astype(int)
    x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.2, random_state=SEED, stratify=y)

    models = candidates(float((y_tr == 0).sum() / (y_tr == 1).sum()))
    chosen, table = compare(models, x_tr, y_tr)
    print(pd.DataFrame(table).to_string(index=False))
    print("chosen:", chosen)

    xgb_name = chosen if chosen.startswith("xgb") else max(
        (r for r in table if r["name"].startswith("xgb")), key=lambda r: r["pr_auc"])["name"]
    if "smote" in xgb_name:
        raise SystemExit("SMOTE pipeline was selected; only the encoder and booster are exported")
    pipe = models[xgb_name]

    lr_bundle = joblib.load(MODEL_DIR / "model.pkl")
    if lr_bundle["numeric"] + lr_bundle["categorical"] != COLS:
        raise SystemExit("model.pkl uses different features, run src/train.py first")

    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    oof = cross_val_predict(pipe, x_tr, y_tr, cv=cv, method="predict_proba")[:, 1]
    threshold = train.choose_threshold(y_tr, oof)

    honest_xgb = clone(pipe).fit(x_tr, y_tr)
    honest_lr = clone(lr_bundle["pipeline"]).fit(x_tr, y_tr)
    p_xgb = honest_xgb.predict_proba(x_te)[:, 1]
    p_lr = honest_lr.predict_proba(x_te)[:, 1]
    metrics = {
        "xgb": test_metrics(y_te, p_xgb, threshold),
        "lr": test_metrics(y_te, p_lr, lr_bundle["threshold"]),
    }
    print(pd.DataFrame(metrics).to_string())

    final = clone(pipe).fit(x, y)
    pre, booster = final.named_steps["pre"], final.named_steps["clf"].get_booster()
    names = pre.get_feature_names_out().tolist()

    contrib = booster.predict(xgboost.DMatrix(pre.transform(x)), pred_contribs=True)[:, :-1]
    by_group = {}
    for j, name in enumerate(names):
        by_group.setdefault(group_of(name), np.zeros(len(x)))
        by_group[group_of(name)] += contrib[:, j]
    importance = sorted(((g, round(float(np.abs(v).mean()), 4)) for g, v in by_group.items()), key=lambda i: -i[1])

    lr_pre = lr_bundle["pipeline"].named_steps["pre"]
    lr_matrix = lr_pre.transform(x)
    lr_mean = np.asarray(lr_matrix.mean(axis=0)).ravel()

    meta = {
        "champion": "xgb" if chosen.startswith("xgb") else "lr",
        "xgb_candidate": xgb_name,
        "threshold": threshold,
        "columns": COLS,
        "feature_names": names,
        "lr_feature_names": lr_pre.get_feature_names_out().tolist(),
        "lr_background": [float(v) for v in lr_mean],
        "metrics": metrics,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data_sha256": hashlib.sha256(train.DATA_PATH.read_bytes()).hexdigest(),
        "versions": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
            "imbalanced_learn": imblearn.__version__,
        },
    }
    insights = {
        "customers": len(df),
        "churn_rate": round(float(y.mean()) * 100, 1),
        "n_test": len(y_te),
        **eda(x_tr, y_tr),
        "gains": {"contacted": list(range(0, 101, 5)), "xgb": gains(y_te, p_xgb), "lr": gains(y_te, p_lr)},
        "importance": {"features": [g for g, _ in importance], "values": [v for _, v in importance]},
        "comparison": table,
        "chosen": chosen,
        "metrics": metrics,
    }

    joblib.dump(pre, MODEL_DIR / "xgb_preprocessor.pkl")
    booster.save_model(MODEL_DIR / "xgb_model.json")
    for name, obj in (("web_meta.json", meta), ("insights.json", insights)):
        (MODEL_DIR / name).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    print("saved to", MODEL_DIR)


if __name__ == "__main__":
    main()
