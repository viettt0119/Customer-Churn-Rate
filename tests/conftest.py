"""Fixture dùng chung. Test dùng model/model.pkl; nếu chưa có thì train một bản tạm."""

import joblib
import pytest
from fastapi.testclient import TestClient

import app as app_module
import train


@pytest.fixture(scope="session")
def model_path(tmp_path_factory):
    path = train.MODEL_DIR / "model.pkl"
    if path.exists():
        return path
    # Bản clone mới chưa có mô hình: train vào thư mục tạm, không ghi vào repo.
    tmp = tmp_path_factory.mktemp("model")
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(train, "MODEL_DIR", tmp)
        train.main()
    return tmp / "model.pkl"


@pytest.fixture(scope="session")
def bundle(model_path):
    return joblib.load(model_path)


@pytest.fixture(scope="session")
def raw_data():
    return train.load_data()


@pytest.fixture(scope="session")
def client(model_path):
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(app_module, "MODEL_PATH", model_path)
        # `with` để chạy lifespan: mô hình được nạp như khi bật server thật.
        with TestClient(app_module.app) as c:
            yield c


@pytest.fixture
def high_risk():
    return {
        "tenure": 2,
        "MonthlyCharges": 95.0,
        "SeniorCitizen": 0,
        "Contract": "Month-to-month",
        "InternetService": "Fiber optic",
        "PaymentMethod": "Electronic check",
        "OnlineSecurity": "No",
        "TechSupport": "No",
    }


@pytest.fixture
def low_risk():
    return {
        "tenure": 60,
        "MonthlyCharges": 50.0,
        "SeniorCitizen": 0,
        "Contract": "Two year",
        "InternetService": "DSL",
        "PaymentMethod": "Credit card (automatic)",
        "OnlineSecurity": "Yes",
        "TechSupport": "Yes",
    }
