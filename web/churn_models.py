import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost

from schema import load_bundle

LABELS = {"lr": "Logistic Regression", "xgb": "XGBoost"}


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def group_of(feature):
    kind, name = feature.split("__", 1)
    return name if kind == "num" else name.split("_", 1)[0]


def group_contributions(names, values):
    out = {}
    for name, value in zip(names, values):
        g = group_of(name)
        out[g] = out.get(g, 0.0) + float(value)
    return dict(sorted(out.items(), key=lambda i: -abs(i[1])))


class LinearModel:
    def __init__(self, bundle, meta):
        self.pipeline = bundle["pipeline"]
        self.columns = bundle["numeric"] + bundle["categorical"]
        self.threshold = bundle["threshold"]
        self.trained_at = bundle["meta"]["trained_at"]
        self.pre = self.pipeline.named_steps["pre"]
        clf = self.pipeline.named_steps["clf"]
        self.names = self.pre.get_feature_names_out().tolist()
        if self.names != meta["lr_feature_names"]:
            raise RuntimeError("web_meta.json không khớp model.pkl. Hãy chạy lại src/train_web.py.")
        self.coef = clf.coef_[0]
        self.background = np.array(meta["lr_background"])
        self.base = float(clf.intercept_[0] + self.coef @ self.background)

    def predict(self, frame):
        return self.pipeline.predict_proba(frame[self.columns])[:, 1]

    def explain(self, frame):
        x = np.asarray(self.pre.transform(frame[self.columns]))[0]
        return self.base, group_contributions(self.names, self.coef * (x - self.background))


class XgbModel:
    def __init__(self, model_dir, meta):
        if meta["versions"]["scikit_learn"] != sklearn.__version__:
            raise RuntimeError(
                f"xgb_preprocessor.pkl được tạo bằng scikit-learn {meta['versions']['scikit_learn']}, "
                f"môi trường đang là {sklearn.__version__}"
            )
        self.pre = joblib.load(model_dir / "xgb_preprocessor.pkl")
        self.booster = xgboost.Booster()
        self.booster.load_model(model_dir / "xgb_model.json")
        self.columns = meta["columns"]
        self.names = meta["feature_names"]
        if self.pre.get_feature_names_out().tolist() != self.names or self.booster.num_features() != len(self.names):
            raise RuntimeError("Các file XGBoost không khớp nhau. Hãy chạy lại src/train_web.py.")
        self.threshold = meta["threshold"]
        self.trained_at = meta["trained_at"]

    def matrix(self, frame):
        return xgboost.DMatrix(self.pre.transform(frame[self.columns]))

    def predict(self, frame):
        return self.booster.predict(self.matrix(frame)).astype(float)

    def explain(self, frame):
        row = self.booster.predict(self.matrix(frame), pred_contribs=True)[0]
        return float(row[-1]), group_contributions(self.names, row[:-1])


class Registry:
    def __init__(self, model_dir: Path):
        needed = ["model.pkl", "xgb_preprocessor.pkl", "xgb_model.json", "web_meta.json", "insights.json"]
        missing = [n for n in needed if not (model_dir / n).exists()]
        if missing:
            raise RuntimeError(
                f"Thiếu {', '.join(missing)} trong {model_dir}. Hãy chạy src/train.py rồi src/train_web.py."
            )
        self.meta = json.loads((model_dir / "web_meta.json").read_text(encoding="utf-8"))
        self.insights = json.loads((model_dir / "insights.json").read_text(encoding="utf-8"))
        bundle = load_bundle(model_dir / "model.pkl")
        if bundle["meta"]["data_sha256"] != self.meta["data_sha256"]:
            raise RuntimeError("model.pkl và XGBoost được train trên dữ liệu khác nhau. Hãy train lại cả hai.")
        self.categories = bundle["categories"]
        self.ranges = bundle["numeric_ranges"]
        self.models = {"xgb": XgbModel(model_dir, self.meta), "lr": LinearModel(bundle, self.meta)}
        self.champion = self.meta["champion"]

    def warnings(self, data):
        return [
            f"{c} = {data[c]:g} nằm ngoài khoảng đã học [{lo:g}, {hi:g}], dự đoán kém tin cậy"
            for c, (lo, hi) in self.ranges.items()
            if not lo <= data[c] <= hi
        ]

    def predict(self, name, data):
        model = self.models[name]
        frame = pd.DataFrame([data])
        proba = float(model.predict(frame)[0])
        base, contributions = model.explain(frame)
        return {
            "model": name,
            "model_label": LABELS[name],
            "churn_probability": round(proba, 4),
            "will_churn": bool(proba >= model.threshold),
            "threshold": model.threshold,
            "trained_at": model.trained_at,
            "warnings": self.warnings(data),
            "explanation": {
                "base_probability": round(float(sigmoid(base)), 4),
                "contributions": [
                    {"feature": k, "value": data[k], "log_odds": round(v, 4)} for k, v in contributions.items()
                ],
            },
        }

    def tenure_curve(self, name, data):
        months = list(range(0, int(self.ranges["tenure"][1]) + 1, 3))
        frame = pd.DataFrame([{**data, "tenure": m} for m in months])
        return {
            "model": name,
            "tenure": months,
            "probability": [round(float(p), 4) for p in self.models[name].predict(frame)],
            "threshold": self.models[name].threshold,
        }

    def describe(self):
        return {
            "champion": self.champion,
            "models": [
                {
                    "name": n,
                    "label": LABELS[n],
                    "threshold": m.threshold,
                    "trained_at": m.trained_at,
                    "test_metrics": self.meta["metrics"][n],
                }
                for n, m in self.models.items()
            ],
        }
