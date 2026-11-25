import json
import re
import shutil

import numpy as np
import pandas as pd
import pytest

import churn_models
import server
import train
import train_web

MODELS = ["xgb", "lr"]
FILES = ["model.pkl", "xgb_preprocessor.pkl", "xgb_model.json", "web_meta.json", "insights.json"]


@pytest.fixture(scope="session")
def web_dir(model_path, tmp_path_factory):
    if all((train.MODEL_DIR / f).exists() for f in FILES):
        return train.MODEL_DIR
    tmp = tmp_path_factory.mktemp("web_model")
    shutil.copy(model_path, tmp / "model.pkl")
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(train_web, "MODEL_DIR", tmp)
        train_web.main()
    return tmp


@pytest.fixture(scope="session")
def flask_app(web_dir):
    return server.create_app(web_dir)


@pytest.fixture(scope="session")
def registry(flask_app):
    return flask_app.extensions["registry"]


@pytest.fixture
def web(flask_app):
    return flask_app.test_client()


def predict(web, customer, **extra):
    return web.post("/api/predict", json={"customer": customer, **extra})


def test_health(web):
    assert web.get("/api/health").get_json() == {"status": "ok"}


def test_models(web, registry):
    body = web.get("/api/models").get_json()
    assert body["champion"] == registry.champion
    assert {m["name"] for m in body["models"]} == set(MODELS)


def test_options_match_schema(web):
    from typing import get_args

    body = web.get("/api/options").get_json()
    for col, values in body["categories"].items():
        assert set(values) == set(get_args(server.Customer.model_fields[col].annotation))


def test_insights_shape(web):
    d = web.get("/api/insights").get_json()
    assert d["customers"] == 7043
    assert len(d["gains"]["contacted"]) == len(d["gains"]["xgb"]) == len(d["gains"]["lr"]) == 21
    assert d["gains"]["xgb"][0] == 0 and d["gains"]["xgb"][-1] == 100
    assert len(d["tenure"]["labels"]) == len(d["tenure"]["rate"])
    assert [r["name"] for r in d["comparison"]].count(d["chosen"]) == 1


@pytest.mark.parametrize("name", MODELS)
def test_predict_contract(web, high_risk, name):
    r = predict(web, high_risk, model=name)
    assert r.status_code == 200
    b = r.get_json()
    assert set(b) == {"model", "model_label", "churn_probability", "will_churn", "threshold", "trained_at", "warnings", "explanation"}
    assert b["model"] == name
    assert 0 <= b["churn_probability"] <= 1
    assert type(b["will_churn"]) is bool
    assert {c["feature"] for c in b["explanation"]["contributions"]} == set(high_risk)


def test_default_model_is_champion(web, high_risk, registry):
    assert predict(web, high_risk).get_json()["model"] == registry.champion


@pytest.mark.parametrize("name", MODELS)
def test_risk_ordering(web, high_risk, low_risk, name):
    hi, lo = (predict(web, c, model=name).get_json() for c in (high_risk, low_risk))
    assert hi["will_churn"] and not lo["will_churn"]
    assert hi["churn_probability"] > lo["churn_probability"]


@pytest.mark.parametrize("name", MODELS)
def test_explanation_adds_up_to_prediction(registry, raw_data, name):
    model = registry.models[name]
    rows = raw_data.sample(40, random_state=2)[model.columns]
    for i in range(len(rows)):
        row = rows.iloc[[i]]
        base, parts = model.explain(row)
        assert churn_models.sigmoid(base + sum(parts.values())) == pytest.approx(model.predict(row)[0], abs=1e-4)


def test_linear_model_matches_fastapi(web, client, high_risk, low_risk):
    for c in (high_risk, low_risk):
        a = predict(web, c, model="lr").get_json()
        b = client.post("/predict", json=c).json()
        assert a["churn_probability"] == b["churn_probability"]
        assert a["will_churn"] == b["will_churn"]


def test_threshold_edge(registry, high_risk, monkeypatch):
    model = registry.models["xgb"]
    monkeypatch.setattr(model, "threshold", 0.3)
    for p, expected in ((0.3, True), (0.299999, False), (0.4, True)):
        monkeypatch.setattr(model, "predict", lambda frame, p=p: np.array([p]))
        assert registry.predict("xgb", high_risk)["will_churn"] is expected


@pytest.mark.parametrize("name", MODELS)
def test_tenure_curve(web, high_risk, name):
    r = web.post("/api/tenure-curve", json={"customer": high_risk, "model": name})
    b = r.get_json()
    assert r.status_code == 200
    assert b["tenure"][0] == 0 and b["tenure"][-1] == 72
    assert len(b["tenure"]) == len(b["probability"])
    assert b["probability"][0] > b["probability"][-1]


def test_curve_agrees_with_predict(web, high_risk):
    b = web.post("/api/tenure-curve", json={"customer": high_risk}).get_json()
    p = predict(web, {**high_risk, "tenure": 30}).get_json()["churn_probability"]
    assert b["probability"][b["tenure"].index(30)] == p


def test_out_of_range_warns(web, high_risk):
    b = predict(web, {**high_risk, "tenure": 90}).get_json()
    assert len(b["warnings"]) == 1 and "tenure" in b["warnings"][0]


@pytest.mark.parametrize("field, value", [
    ("tenure", -1), ("tenure", "2"), ("tenure", 2.5), ("tenure", None), ("MonthlyCharges", 0),
    ("MonthlyCharges", None), ("SeniorCitizen", True), ("SeniorCitizen", 2), ("Contract", "Three year"),
    ("PaymentMethod", "Momo"), ("TechSupport", "No internet service"), ("InternetService", "No"),
])
@pytest.mark.parametrize("path", ["/api/predict", "/api/tenure-curve"])
def test_bad_customer_rejected(web, high_risk, path, field, value):
    r = web.post(path, json={"customer": {**high_risk, field: value}})
    assert r.status_code == 422
    assert all(set(e) == {"type", "loc", "msg"} for e in r.get_json()["detail"])


@pytest.mark.parametrize("number", ["NaN", "Infinity", "1e999"])
def test_non_finite_rejected(web, high_risk, number):
    raw = json.dumps({"customer": high_risk}).replace("95.0", number)
    r = web.post("/api/predict", data=raw, content_type="application/json")
    assert r.status_code == 422


@pytest.mark.parametrize("body", [[], None, 1, {}, {"customer": None}, {"customer": []}])
def test_wrong_shape_rejected(web, body):
    r = web.post("/api/predict", data=json.dumps(body), content_type="application/json")
    assert r.status_code == 422


@pytest.mark.parametrize("model", ["svm", 1, None, ["xgb"]])
def test_unknown_model_rejected(web, high_risk, model):
    assert predict(web, high_risk, model=model).status_code == 422


def test_extra_fields_rejected(web, high_risk):
    assert predict(web, high_risk, other=1).status_code == 422
    assert predict(web, {**high_risk, "Churn": "Yes"}).status_code == 422


def test_error_does_not_echo_input(web, high_risk):
    assert "du_lieu_rieng" not in predict(web, {**high_risk, "Contract": "du_lieu_rieng"}).get_data(as_text=True)


@pytest.mark.parametrize("kwargs, status", [
    ({"data": "{", "content_type": "application/json"}, 400),
    ({"data": "{}", "content_type": "text/plain"}, 415),
    ({"data": b"x" * 20000, "content_type": "application/json"}, 413),
])
def test_http_errors_are_json(web, kwargs, status):
    r = web.post("/api/predict", **kwargs)
    assert r.status_code == status
    assert set(r.get_json()) == {"detail"}


def test_unknown_api_path_and_method(web):
    assert web.get("/api/nope").status_code == 404
    assert web.get("/api/nope").is_json
    assert web.get("/api/predict").status_code == 405


def test_page_has_no_inline_script_or_remote_asset(web):
    r = web.get("/")
    html = r.get_data(as_text=True)
    assert r.status_code == 200
    assert not re.search(r"<script(?![^>]*\bsrc=)", html)
    assert not re.search(r'(?:src|href)="(?:https?:)?//', html)
    assert "script-src 'self'" in r.headers["Content-Security-Policy"]


def test_every_referenced_asset_exists(web):
    html = web.get("/").get_data(as_text=True)
    sw = web.get("/sw.js").get_data(as_text=True)
    shell = re.findall(r'"(/(?:static|manifest)[^"?]*[^"?/])"', sw) + re.findall(r'"(/api/[^"]+)"', sw)
    urls = re.findall(r'(?:src|href)="(/[^"]+)"', html) + shell
    assert len(urls) > 15
    for url in urls:
        assert web.get(url).status_code == 200, url


def test_service_worker(web):
    r = web.get("/sw.js")
    assert r.mimetype == "text/javascript"
    assert r.headers["Cache-Control"] == "no-cache"
    assert "{{" not in r.get_data(as_text=True)
    assert "/api/predict" not in r.get_data(as_text=True)


def test_manifest(web):
    from io import BytesIO

    from PIL import Image

    r = web.get("/manifest.webmanifest")
    m = json.loads(r.data)
    assert r.mimetype == "application/manifest+json"
    assert m["display"] == "standalone" and m["start_url"] == "/"
    assert {i["purpose"] for i in m["icons"]} == {"any", "maskable"}
    for i in m["icons"]:
        image = Image.open(BytesIO(web.get(i["src"]).data))
        assert f"{image.width}x{image.height}" == i["sizes"]


def test_api_responses_are_not_cached(web):
    assert web.get("/api/health").headers["Cache-Control"] == "no-store"


@pytest.fixture
def copied(web_dir, tmp_path):
    for f in FILES:
        shutil.copy(web_dir / f, tmp_path / f)
    return tmp_path


@pytest.mark.parametrize("name", FILES)
def test_missing_file_refused(copied, name):
    (copied / name).unlink()
    with pytest.raises(RuntimeError, match=re.escape(name)):
        server.create_app(copied)


@pytest.mark.parametrize("edit, message", [
    (lambda m: m["versions"].update(scikit_learn="0.0.0"), "0.0.0"),
    (lambda m: m["feature_names"].pop(), "XGBoost"),
    (lambda m: m["lr_feature_names"].pop(), "model.pkl"),
    (lambda m: m.update(data_sha256="x"), "dữ liệu khác nhau"),
], ids=["sklearn", "xgb_features", "lr_features", "data"])
def test_inconsistent_meta_refused(copied, edit, message):
    path = copied / "web_meta.json"
    meta = json.loads(path.read_text(encoding="utf-8"))
    edit(meta)
    path.write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(RuntimeError, match=message):
        server.create_app(copied)


def test_quality_and_calibration(registry):
    xgb, lr = registry.meta["metrics"]["xgb"], registry.meta["metrics"]["lr"]
    assert xgb["pr_auc"] >= 0.60
    assert xgb["roc_auc"] >= 0.82
    assert xgb["brier"] <= 0.15
    assert lr["roc_auc"] >= 0.81


def test_smote_only_touches_training_rows():
    pipe = train_web.candidates(2.7)["lr_smote"]
    rng = np.random.default_rng(0)
    df = train.load_data().iloc[rng.permutation(7043)[:800]]
    x, y = df[train_web.COLS], (df["Churn"] == "Yes").astype(int)
    pipe.fit(x, y)
    assert len(pipe.predict_proba(x.iloc[:50])) == 50
    resampled, target = train_web.smote().fit_resample(x, y)
    assert len(resampled) > len(x) and target.mean() == pytest.approx(0.5)


def test_main_never_selects_on_test_rows(monkeypatch, tmp_path, model_path):
    from sklearn.model_selection import train_test_split

    shutil.copy(model_path, tmp_path / "model.pkl")
    small = train.load_data().sample(1200, random_state=0)
    monkeypatch.setattr(train, "load_data", lambda: small)
    monkeypatch.setattr(train_web, "MODEL_DIR", tmp_path)
    seen = {}
    real_compare, real_threshold, real_eda = train_web.compare, train.choose_threshold, train_web.eda

    def spy_compare(models, x, y):
        seen["compare"] = set(x.index)
        keep = {k: models[k] for k in ("lr", "xgb_depth2")}
        return real_compare(keep, x, y)

    monkeypatch.setattr(train_web, "compare", spy_compare)
    monkeypatch.setattr(train, "choose_threshold", lambda y, p: seen.update(threshold=set(y.index)) or real_threshold(y, p))
    monkeypatch.setattr(train_web, "eda", lambda x, y: seen.update(eda=set(x.index)) or real_eda(x, y))
    train_web.main()

    y = (small["Churn"] == "Yes").astype(int)
    tr, te = train_test_split(y, test_size=0.2, random_state=train.RANDOM_STATE, stratify=y)
    assert seen["compare"] == seen["threshold"] == seen["eda"] == set(tr.index)

    meta = json.loads((tmp_path / "web_meta.json").read_text(encoding="utf-8"))
    pipe = train_web.candidates(1.0)["xgb_depth2"]
    honest = pipe.fit(small.loc[tr.index, train_web.COLS], tr).predict_proba(small.loc[te.index, train_web.COLS])[:, 1]
    assert meta["metrics"]["xgb"]["roc_auc"] == pytest.approx(train_web.roc_auc_score(te, honest), abs=2e-4)

    import xgboost

    shipped = churn_models.XgbModel(tmp_path, meta)
    full = train_web.candidates(1.0)["xgb_depth2"].fit(small[train_web.COLS], y)
    np.testing.assert_allclose(shipped.predict(small[train_web.COLS]), full.predict_proba(small[train_web.COLS])[:, 1], atol=1e-5)
    assert isinstance(shipped.booster, xgboost.Booster)
    assert abs(train_web.roc_auc_score(te, shipped.predict(small.loc[te.index, train_web.COLS])) - meta["metrics"]["xgb"]["roc_auc"]) > 1e-3


def test_group_contributions_covers_every_feature(registry):
    groups = {churn_models.group_of(n) for n in registry.meta["feature_names"]}
    assert groups == set(train_web.COLS)
    assert pd.Series(registry.insights["importance"]["features"]).is_unique
