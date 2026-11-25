"""Huấn luyện mô hình churn trên data/Telco-Customer-Churn.csv.

Chạy từ thư mục gốc của repo:
    .venv/Scripts/python.exe src/train.py

Trình tự (tập test chỉ dùng đúng một lần, ở bước 4):
    1. Tách train/test.
    2. Chọn bộ feature bằng cross-validation trên train.
    3. Chọn ngưỡng bằng dự đoán out-of-fold trên train.
    4. Đo trên test.
    5. Refit trên toàn bộ dữ liệu và lưu vào model/.
"""

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "Telco-Customer-Churn.csv"
MODEL_DIR = ROOT / "model"

RANDOM_STATE = 42
TARGET = "Churn"

BASE_NUM = ["tenure", "MonthlyCharges", "SeniorCitizen"]
BASE_CAT = ["Contract", "InternetService", "PaymentMethod"]
ADDONS = ["OnlineSecurity", "TechSupport"]

# Các ứng viên, xếp từ đơn giản đến phức tạp.
CANDIDATES = {
    "no_total": dict(numeric=BASE_NUM, categorical=BASE_CAT),
    "lesson": dict(numeric=BASE_NUM + ["TotalCharges"], categorical=BASE_CAT),
    "interaction": dict(numeric=BASE_NUM, categorical=BASE_CAT, interaction=True),
    "addons": dict(numeric=BASE_NUM, categorical=BASE_CAT + ADDONS),
    "addons_interaction": dict(
        numeric=BASE_NUM, categorical=BASE_CAT + ADDONS, interaction=True
    ),
}
SELECTION_METRIC = "average_precision"  # PR-AUC: hợp với lớp hiếm


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Đọc và làm sạch theo quy tắc cố định, không học gì từ dữ liệu."""
    df = pd.read_csv(path)
    # 11 khách có tenure = 0 để TotalCharges là dấu cách.
    blank = df["TotalCharges"].astype(str).str.strip() == ""
    if not (df.loc[blank, "tenure"] == 0).all():
        raise ValueError("TotalCharges trống ở khách có tenure > 0: cần xem lại dữ liệu")
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].mask(blank, "0"))
    if df.isna().any().any() or not set(df[TARGET]) <= {"Yes", "No"}:
        raise ValueError("Dữ liệu có giá trị thiếu hoặc nhãn lạ")
    return df


def build_pipeline(numeric, categorical, interaction=False) -> Pipeline:
    """Chỉ dùng lớp có sẵn của sklearn để model.pkl nạp được ở mọi tiến trình."""
    transformers = [
        ("num", StandardScaler(), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
    ]
    if interaction:
        # InternetService × MonthlyCharges: hệ số giá riêng cho từng loại Internet.
        transformers.append((
            "price_by_net",
            Pipeline([
                ("enc", ColumnTransformer([
                    ("num", StandardScaler(), ["MonthlyCharges"]),
                    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                     ["InternetService"]),
                ])),
                ("cross", PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)),
            ]),
            ["MonthlyCharges", "InternetService"],
        ))
    return Pipeline([
        ("pre", ColumnTransformer(transformers)),
        ("clf", LogisticRegression(max_iter=2000)),
    ])


def select_candidate(x, y):
    """Chọn ứng viên đơn giản nhất không thua rõ rệt ứng viên tốt nhất.

    Sai số chuẩn của chênh lệch theo từng fold được hiệu chỉnh (Nadeau-Bengio)
    vì các fold dùng chung dữ liệu train.
    """
    k, repeats = 5, 4
    cv = RepeatedStratifiedKFold(n_splits=k, n_repeats=repeats, random_state=RANDOM_STATE)
    scores = {}
    for name, spec in CANDIDATES.items():
        scores[name] = cross_validate(
            build_pipeline(**spec), x, y, cv=cv, n_jobs=1,
            scoring=["roc_auc", "average_precision"],
        )
    key = f"test_{SELECTION_METRIC}"
    best = max(scores, key=lambda n: scores[n][key].mean())
    rows, chosen = [], None
    for name in CANDIDATES:
        gap = scores[best][key] - scores[name][key]
        se = float(np.sqrt((1 / len(gap) + 1 / (k - 1)) * gap.var(ddof=1)))
        ok = bool(gap.mean() <= se)
        if ok and chosen is None:
            chosen = name
        rows.append({
            "candidate": name,
            "roc_auc": scores[name]["test_roc_auc"].mean(),
            "pr_auc": scores[name]["test_average_precision"].mean(),
            "gap_to_best": float(gap.mean()),
            "gap_se": se,
            "within_1se": ok,
        })
    return chosen, pd.DataFrame(rows)


def choose_threshold(y, proba) -> float:
    """Ngưỡng cho F1 cao nhất (chưa có chi phí thật của FN/FP)."""
    precision, recall, thresholds = precision_recall_curve(y, proba)
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    return round(float(thresholds[f1.argmax()]), 4)


def evaluate(y, proba, threshold) -> dict:
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    return {
        "threshold": threshold,
        "accuracy": round(float(accuracy_score(y, pred)), 4),
        "precision": round(float(precision_score(y, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y, pred)), 4),
        "f1": round(float(f1_score(y, pred)), 4),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    df = load_data()
    features = sorted({c for s in CANDIDATES.values() for c in s["numeric"] + s["categorical"]})
    x, y = df[features], (df[TARGET] == "Yes").astype(int)

    # 1. Tách trước, mọi lựa chọn bên dưới chỉ dùng train.
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    # 2. Chọn bộ feature.
    chosen, table = select_candidate(x_train, y_train)
    spec = CANDIDATES[chosen]
    print("== Cross-validation trên train ==")
    print(table.round(4).to_string(index=False))
    print(f"Chọn: {chosen}\n")

    # 3. Chọn ngưỡng.
    cols = spec["numeric"] + spec["categorical"]
    pipe = build_pipeline(**spec)
    cv = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)
    oof = cross_val_predict(pipe, x_train[cols], y_train, cv=cv, method="predict_proba")[:, 1]
    threshold = choose_threshold(y_train, oof)

    # 4. Đo trên test, một lần duy nhất.
    pipe.fit(x_train[cols], y_train)
    proba = pipe.predict_proba(x_test[cols])[:, 1]
    metrics = {
        "note": "Đo trên test với mô hình fit trên train, TRƯỚC khi refit toàn bộ dữ liệu",
        "n_train": len(x_train),
        "n_test": len(x_test),
        "roc_auc": round(float(roc_auc_score(y_test, proba)), 4),
        "pr_auc": round(float(average_precision_score(y_test, proba)), 4),
        "at_chosen_threshold": evaluate(y_test, proba, threshold),
        "at_0.5": evaluate(y_test, proba, 0.5),
        "baseline_all_no": evaluate(y_test, np.zeros(len(y_test)), 0.5),
        "baseline_all_yes": evaluate(y_test, np.ones(len(y_test)), 0.5),
        "baseline_pr_auc": round(float(y_test.mean()), 4),
        "cv_on_train": table.round(4).to_dict("records"),
    }
    print("== Test ==")
    print(f"ROC-AUC {metrics['roc_auc']}  |  PR-AUC {metrics['pr_auc']} (baseline {metrics['baseline_pr_auc']})")
    names = ["at_chosen_threshold", "at_0.5", "baseline_all_no", "baseline_all_yes"]
    print(pd.DataFrame([metrics[n] for n in names], index=names).to_string())

    # 5. Refit toàn bộ để xuất xưởng.
    final = clone(pipe).fit(x[cols], y)
    bundle = {
        "pipeline": final,
        "numeric": spec["numeric"],
        "categorical": spec["categorical"],
        "categories": {c: sorted(df[c].unique().tolist()) for c in spec["categorical"]},
        "numeric_ranges": {c: [float(df[c].min()), float(df[c].max())] for c in spec["numeric"]},
        "threshold": threshold,
        "metrics": metrics,
        "meta": {
            "candidate": chosen,
            "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "data_file": DATA_PATH.name,
            "data_sha256": hashlib.sha256(DATA_PATH.read_bytes()).hexdigest(),
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
        },
    }
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(bundle, MODEL_DIR / "model.pkl")
    sidecar = {k: bundle[k] for k in ("meta", "numeric", "categorical", "threshold", "metrics")}
    (MODEL_DIR / "metrics.json").write_text(
        json.dumps(sidecar, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nĐã lưu {MODEL_DIR / 'model.pkl'} và metrics.json")


if __name__ == "__main__":
    main()
