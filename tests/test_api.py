"""Hợp đồng request/response của API.

Không so xác suất với một con số cố định: train lại sẽ đổi số, còn hợp đồng thì không.
"""

import json

import numpy as np
import pytest


def post(client, body):
    return client.post("/predict", json=body)


# --- Response ---------------------------------------------------------------


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_root_redirects_to_docs(client):
    r = client.get("/", follow_redirects=False)
    assert r.status_code == 307
    assert r.headers["location"] == "/docs"
    assert client.get("/docs").status_code == 200


def test_info_describes_running_model(client, bundle):
    r = client.get("/info")
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"meta", "numeric", "categorical", "threshold", "metrics"}
    assert body["threshold"] == bundle["threshold"]
    assert body["meta"]["data_sha256"] == bundle["meta"]["data_sha256"]


def test_predict_response_contract(client, high_risk, bundle):
    r = post(client, high_risk)
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"churn_probability", "will_churn", "threshold", "trained_at", "warnings"}
    assert type(body["churn_probability"]) is float
    assert 0 <= body["churn_probability"] <= 1
    assert type(body["will_churn"]) is bool
    assert body["threshold"] == bundle["threshold"]
    assert body["trained_at"] == bundle["meta"]["trained_at"]
    assert body["warnings"] == []


def test_openapi_contract(client):
    """Đổi tên hay bỏ trường là phá vỡ client, nên phải sửa test này một cách có chủ ý."""
    spec = client.get("/openapi.json").json()
    paths = {(p, m) for p, ops in spec["paths"].items() for m in ops}
    assert paths == {("/health", "get"), ("/info", "get"), ("/predict", "post")}

    customer = spec["components"]["schemas"]["Customer"]
    assert set(customer["required"]) == set(customer["properties"]) == {
        "tenure", "MonthlyCharges", "SeniorCitizen", "Contract",
        "InternetService", "PaymentMethod", "OnlineSecurity", "TechSupport",
    }
    prediction = spec["components"]["schemas"]["Prediction"]["properties"]
    assert {k: v["type"] for k, v in prediction.items()} == {
        "churn_probability": "number", "will_churn": "boolean",
        "threshold": "number", "trained_at": "string", "warnings": "array",
    }


def test_documented_example_works(client):
    """Ví dụ trên trang /docs phải chạy được."""
    spec = client.get("/openapi.json").json()
    example = spec["components"]["schemas"]["Customer"]["examples"][0]
    assert post(client, example).status_code == 200


# --- Hành vi dự đoán --------------------------------------------------------


def test_risk_ordering(client, high_risk, low_risk):
    high, low = post(client, high_risk).json(), post(client, low_risk).json()
    assert high["churn_probability"] > low["churn_probability"]
    assert high["will_churn"] is True
    assert low["will_churn"] is False


def test_longer_tenure_lowers_risk(client, high_risk):
    probs = [post(client, {**high_risk, "tenure": t}).json()["churn_probability"] for t in (0, 24, 72)]
    assert probs == sorted(probs, reverse=True)
    assert len(set(probs)) == 3


def test_new_customer_accepted(client, high_risk):
    r = post(client, {**high_risk, "tenure": 0})
    assert r.status_code == 200
    assert r.json()["warnings"] == []


def test_field_order_does_not_matter(client, high_risk):
    shuffled = dict(reversed(list(high_risk.items())))
    assert post(client, shuffled).json() == post(client, high_risk).json()


def test_matches_pipeline_on_real_rows(client, bundle, raw_data):
    cols = bundle["numeric"] + bundle["categorical"]
    rows = raw_data.sample(100, random_state=1)[cols]
    expected = bundle["pipeline"].predict_proba(rows)[:, 1]
    for body, p in zip(rows.to_dict("records"), expected):
        r = post(client, body)
        assert r.status_code == 200, r.text
        assert r.json()["churn_probability"] == pytest.approx(p, abs=1e-4)
        assert r.json()["will_churn"] == (p >= bundle["threshold"])


class FixedProba:
    """Pipeline giả trả về đúng một xác suất, để kiểm tra riêng logic ngưỡng."""

    def __init__(self, p):
        self.p = p

    def predict_proba(self, rows):
        return np.array([[1 - self.p, self.p]] * len(rows))


@pytest.fixture
def swap_bundle(client):
    """Thay tạm bundle đang chạy rồi trả lại như cũ."""
    original = client.app.state.bundle

    def swap(**changes):
        client.app.state.bundle = {**original, **changes}

    yield swap
    client.app.state.bundle = original


@pytest.mark.parametrize(
    "threshold, p, expected",
    [
        (0.3, 0.3, True),  # đúng bằng ngưỡng thì tính là churn
        (0.3, 0.2999, False),
        (0.3, 0.4, True),  # dưới 0.5 vẫn là churn: ngưỡng không bị viết cứng
        (0.7, 0.6, False),
        (0.3, 0.299999, False),  # so trước khi làm tròn (0.299999 làm tròn thành 0.3)
    ],
)
def test_threshold_comes_from_bundle(client, high_risk, swap_bundle, threshold, p, expected):
    swap_bundle(pipeline=FixedProba(p), threshold=threshold)
    body = post(client, high_risk).json()
    assert body["will_churn"] is expected
    assert body["threshold"] == threshold


# --- Cảnh báo ngoài khoảng đã học -------------------------------------------


def test_out_of_range_is_predicted_with_warning(client, high_risk):
    r = post(client, {**high_risk, "tenure": 90, "MonthlyCharges": 500000.0})
    assert r.status_code == 200
    text = " ".join(r.json()["warnings"])
    assert len(r.json()["warnings"]) == 2
    assert "tenure" in text and "MonthlyCharges" in text


def test_range_edges_have_no_warning(client, high_risk, bundle):
    for i in (0, 1):
        edge = {
            "tenure": int(bundle["numeric_ranges"]["tenure"][i]),
            "MonthlyCharges": bundle["numeric_ranges"]["MonthlyCharges"][i],
        }
        assert post(client, {**high_risk, **edge}).json()["warnings"] == []


# --- Request sai phải bị từ chối --------------------------------------------


@pytest.mark.parametrize(
    "field, value",
    [
        ("tenure", -1),
        ("tenure", "2"),
        ("tenure", 2.5),
        ("tenure", None),
        ("MonthlyCharges", 0),
        ("MonthlyCharges", -5.0),
        ("MonthlyCharges", "95"),
        ("MonthlyCharges", None),
        ("SeniorCitizen", 2),
        ("SeniorCitizen", -1),
        ("SeniorCitizen", True),
        ("Contract", "Three year"),
        ("Contract", "month-to-month"),
        ("InternetService", "5G"),
        ("PaymentMethod", "Momo"),
        ("OnlineSecurity", "Maybe"),
        ("TechSupport", ""),
    ],
)
def test_bad_value_rejected(client, high_risk, field, value):
    r = post(client, {**high_risk, field: value})
    assert r.status_code == 422
    assert field in [e["loc"][-1] for e in r.json()["detail"]]


@pytest.mark.parametrize("field", [
    "tenure", "MonthlyCharges", "SeniorCitizen", "Contract",
    "InternetService", "PaymentMethod", "OnlineSecurity", "TechSupport",
])
def test_missing_field_rejected(client, high_risk, field):
    del high_risk[field]
    r = post(client, high_risk)
    assert r.status_code == 422
    assert r.json()["detail"][0]["loc"] == ["body", field]
    assert r.json()["detail"][0]["type"] == "missing"


def test_extra_field_rejected(client, high_risk):
    r = post(client, {**high_risk, "Churn": "Yes"})
    assert r.status_code == 422
    assert r.json()["detail"][0]["type"] == "extra_forbidden"


@pytest.mark.parametrize("raw", ['{"tenure": ', "", "[]", "null"])
def test_malformed_body_rejected(client, raw):
    r = client.post("/predict", content=raw, headers={"content-type": "application/json"})
    assert r.status_code == 422


@pytest.mark.parametrize("number", ["NaN", "Infinity"])
def test_non_finite_number_rejected(client, high_risk, number):
    raw = json.dumps(high_risk).replace("95.0", number)
    r = client.post("/predict", content=raw, headers={"content-type": "application/json"})
    assert r.status_code == 422


def test_error_does_not_echo_customer_data(client, high_risk):
    r = post(client, {**high_risk, "Contract": "du_lieu_rieng_tu"})
    assert r.status_code == 422
    assert "du_lieu_rieng_tu" not in r.text
    assert set(r.json()["detail"][0]) == {"type", "loc", "msg"}


def test_get_predict_not_allowed(client):
    assert client.get("/predict").status_code == 405


# --- Quan hệ giữa InternetService và dịch vụ kèm ----------------------------

NO_NET = "No internet service"


def test_customer_without_internet_accepted(client, high_risk):
    body = {**high_risk, "InternetService": "No", "OnlineSecurity": NO_NET,
            "TechSupport": NO_NET, "MonthlyCharges": 20.0}
    assert post(client, body).status_code == 200


@pytest.mark.parametrize(
    "change",
    [
        {"InternetService": "No"},
        {"InternetService": "No", "OnlineSecurity": NO_NET},
        {"InternetService": "No", "TechSupport": NO_NET},
        {"OnlineSecurity": NO_NET},
        {"TechSupport": NO_NET},
        {"InternetService": "DSL", "TechSupport": NO_NET},
    ],
)
def test_inconsistent_internet_rejected(client, high_risk, change):
    assert post(client, {**high_risk, **change}).status_code == 422
