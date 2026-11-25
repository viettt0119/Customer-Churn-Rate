"""Làm sạch dữ liệu và các bước huấn luyện chống leakage."""

import json

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

import train


@pytest.fixture
def csv(tmp_path):
    """Ghi một CSV nhỏ lấy từ dữ liệu thật, có thể sửa trước khi ghi."""

    def make(edit=lambda d: d):
        d = pd.read_csv(train.DATA_PATH, nrows=20)
        path = tmp_path / "mini.csv"
        edit(d).to_csv(path, index=False)
        return path

    return make


def test_load_real_data(raw_data):
    assert len(raw_data) == 7043
    assert pd.api.types.is_numeric_dtype(raw_data["TotalCharges"])
    assert not raw_data.isna().any().any()
    assert (raw_data.loc[raw_data["tenure"] == 0, "TotalCharges"] == 0).all()


def test_blank_total_for_new_customer_becomes_zero(csv):
    df = train.load_data(csv(lambda d: d.assign(
        tenure=[0] + list(d["tenure"][1:]), TotalCharges=[" "] + list(d["TotalCharges"][1:]))))
    assert df.loc[0, "TotalCharges"] == 0


def test_blank_total_for_old_customer_is_an_error(csv):
    path = csv(lambda d: d.assign(tenure=5, TotalCharges=[" "] + list(d["TotalCharges"][1:])))
    with pytest.raises(ValueError, match="TotalCharges"):
        train.load_data(path)


def test_unknown_label_is_an_error(csv):
    path = csv(lambda d: d.assign(Churn=["Maybe"] + list(d["Churn"][1:])))
    with pytest.raises(ValueError):
        train.load_data(path)


@pytest.mark.parametrize("name", list(train.CANDIDATES))
def test_no_candidate_uses_target_or_id(name):
    spec = train.CANDIDATES[name]
    assert not {train.TARGET, "customerID"} & set(spec["numeric"] + spec["categorical"])


def test_scaler_learns_from_fit_rows_only(raw_data):
    """Chống train-test leakage: predict trên dữ liệu khác không được đổi tham số scaler."""
    spec = train.CANDIDATES["addons"]
    cols = spec["numeric"] + spec["categorical"]
    part, other = raw_data.iloc[:3000], raw_data.iloc[3000:]
    pipe = train.build_pipeline(**spec).fit(part[cols], (part["Churn"] == "Yes").astype(int))
    scaler = pipe.named_steps["pre"].named_transformers_["num"]
    expected = part[spec["numeric"]].mean().to_numpy()
    np.testing.assert_allclose(scaler.mean_, expected)
    pipe.predict_proba(other[cols])
    np.testing.assert_allclose(scaler.mean_, expected)


@pytest.mark.parametrize("name", list(train.CANDIDATES))
def test_every_candidate_fits_and_is_deterministic(raw_data, name):
    spec = train.CANDIDATES[name]
    x = raw_data[spec["numeric"] + spec["categorical"]].iloc[:1000]
    y = (raw_data["Churn"].iloc[:1000] == "Yes").astype(int)
    p1, p2 = (train.build_pipeline(**spec).fit(x, y).predict_proba(x)[:, 1] for _ in range(2))
    assert np.isfinite(p1).all()
    np.testing.assert_array_equal(p1, p2)


def test_threshold_maximises_f1():
    y = np.array([0, 0, 0, 1, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    t = train.choose_threshold(y, p)
    best = max(train.evaluate(y, p, c)["f1"] for c in p)
    assert train.evaluate(y, p, t)["f1"] == best


def test_evaluate_counts_positive_as_churn():
    m = train.evaluate(np.array([1, 1, 1, 0, 0]), np.array([0.9, 0.8, 0.1, 0.7, 0.2]), 0.5)
    assert (m["tp"], m["fn"], m["fp"], m["tn"]) == (2, 1, 1, 1)
    assert m["recall"] == pytest.approx(2 / 3, abs=1e-4)
    assert m["precision"] == pytest.approx(2 / 3, abs=1e-4)


def test_main_keeps_test_rows_out_then_refits_on_all(monkeypatch, tmp_path):
    """Chạy main() thật trên tập con, ghi lại những dòng nào được dùng ở từng bước."""
    # Chỉ số dòng khác mặc định để nhận diện từng khách.
    small = train.load_data().sample(1500, random_state=0)
    monkeypatch.setattr(train, "load_data", lambda: small)
    monkeypatch.setattr(train, "MODEL_DIR", tmp_path)

    seen = {}
    real_select, real_threshold = train.select_candidate, train.choose_threshold

    def spy_select(x, y):
        seen["select"] = set(x.index) | set(y.index)
        return real_select(x, y)

    def spy_threshold(y, p):
        seen["threshold"] = set(y.index)
        return real_threshold(y, p)

    monkeypatch.setattr(train, "select_candidate", spy_select)
    monkeypatch.setattr(train, "choose_threshold", spy_threshold)
    train.main()

    y = (small["Churn"] == "Yes").astype(int)
    _, test = train_test_split(y, test_size=0.2, random_state=train.RANDOM_STATE, stratify=y)
    assert len(seen["select"]) == len(seen["threshold"]) == len(small) - len(test)
    assert not seen["select"] & set(test.index)
    assert not seen["threshold"] & set(test.index)

    saved = joblib.load(tmp_path / "model.pkl")
    cols = saved["numeric"] + saved["categorical"]
    spec = train.CANDIDATES[saved["meta"]["candidate"]]
    train_idx = y.index.difference(test.index, sort=False)

    # Điểm báo cáo phải đến từ mô hình chỉ học train, không phải bản refit.
    honest = train.build_pipeline(**spec).fit(small.loc[train_idx, cols], y[train_idx])
    auc = roc_auc_score(test, honest.predict_proba(small.loc[test.index, cols])[:, 1])
    assert saved["metrics"]["roc_auc"] == pytest.approx(auc, abs=2e-4)
    refit_auc = roc_auc_score(test, saved["pipeline"].predict_proba(small.loc[test.index, cols])[:, 1])
    assert abs(refit_auc - auc) > 4e-4, "hai mô hình quá giống nhau, test này không phân biệt được"

    # Mô hình xuất xưởng phải học toàn bộ dữ liệu.
    shipped = train.build_pipeline(**spec).fit(small[cols], y)
    np.testing.assert_allclose(
        saved["pipeline"].predict_proba(small[cols]), shipped.predict_proba(small[cols]), atol=1e-4
    )
    assert json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))["metrics"] == saved["metrics"]
