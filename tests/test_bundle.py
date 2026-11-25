"""model.pkl: nội dung đúng, và API từ chối khởi động với file không phù hợp."""

import copy
import hashlib
import subprocess
import sys

import joblib
import pytest
from fastapi.testclient import TestClient

import app as app_module
import train


@pytest.fixture
def broken(bundle, tmp_path):
    """Lưu một bản sao đã sửa của bundle và trả về đường dẫn."""

    def make(edit):
        b = copy.deepcopy(bundle)
        edit(b)
        path = tmp_path / "model.pkl"
        joblib.dump(b, path)
        return path

    return make


def test_valid_bundle_loads(model_path):
    assert app_module.load_bundle(model_path)["threshold"] > 0


def test_missing_file_refused(tmp_path):
    with pytest.raises(RuntimeError, match="train.py"):
        app_module.load_bundle(tmp_path / "khong_co.pkl")


def test_other_sklearn_version_refused(broken):
    path = broken(lambda b: b["meta"].update(scikit_learn="0.0.0"))
    with pytest.raises(RuntimeError, match="0.0.0"):
        app_module.load_bundle(path)


@pytest.mark.parametrize(
    "edit",
    [
        lambda b: b["numeric"].append("TotalCharges"),
        lambda b: b["numeric"].remove("tenure"),
    ],
    ids=["thua_feature", "thieu_feature"],
)
def test_feature_mismatch_refused(broken, edit):
    with pytest.raises(RuntimeError, match="feature"):
        app_module.load_bundle(broken(edit))


@pytest.mark.parametrize(
    "edit",
    [
        lambda b: b["categories"]["Contract"].append("Three year"),
        lambda b: b["categories"]["Contract"].remove("One year"),
    ],
    ids=["them_gia_tri", "bot_gia_tri"],
)
def test_category_mismatch_refused(broken, edit):
    with pytest.raises(RuntimeError, match="Contract"):
        app_module.load_bundle(broken(edit))


def test_server_does_not_start_without_model(monkeypatch, tmp_path):
    monkeypatch.setattr(app_module, "MODEL_PATH", tmp_path / "khong_co.pkl")
    with pytest.raises(RuntimeError, match="train.py"):
        with TestClient(app_module.app):
            pass


def test_bundle_contents(bundle):
    assert set(bundle) == {
        "pipeline", "numeric", "categorical", "categories",
        "numeric_ranges", "threshold", "metrics", "meta",
    }
    assert 0 < bundle["threshold"] < 1
    cols = bundle["numeric"] + bundle["categorical"]
    assert train.TARGET not in cols and "customerID" not in cols


def test_bundle_was_trained_on_current_data(bundle):
    """Đổi CSV mà quên train lại thì test này báo."""
    assert bundle["meta"]["data_sha256"] == hashlib.sha256(train.DATA_PATH.read_bytes()).hexdigest()


def test_quality_floor(bundle):
    """Chặn tụt chất lượng: cận dưới KTC 95% trên test (làm tròn xuống), và phải hơn baseline."""
    m = bundle["metrics"]
    assert m["roc_auc"] >= 0.81
    assert m["pr_auc"] >= 0.57
    assert m["at_chosen_threshold"]["f1"] > m["baseline_all_yes"]["f1"]
    assert m["at_chosen_threshold"]["accuracy"] > m["baseline_all_no"]["accuracy"]


def test_loads_without_training_code(model_path):
    """model.pkl phải nạp được ở tiến trình không import train.py."""
    code = (
        "import sys, joblib; b = joblib.load(sys.argv[1]); "
        "assert 'train' not in sys.modules; assert hasattr(b['pipeline'], 'predict_proba')"
    )
    r = subprocess.run([sys.executable, "-I", "-c", code, str(model_path)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
