

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal, get_args

import joblib
import pandas as pd
import sklearn
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, model_validator

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = Path(os.environ.get("CHURN_MODEL_PATH", ROOT / "model" / "model.pkl"))

NO_NET = "No internet service"
Addon = Literal["No", "No internet service", "Yes"]


class Customer(BaseModel):
    """Thông tin một khách hàng. Sai kiểu, sai giá trị hoặc thừa trường đều bị từ chối (422)."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [{
                "tenure": 2,
                "MonthlyCharges": 95.0,
                "SeniorCitizen": 0,
                "Contract": "Month-to-month",
                "InternetService": "Fiber optic",
                "PaymentMethod": "Electronic check",
                "OnlineSecurity": "No",
                "TechSupport": "No",
            }]
        },
    )

    tenure: int = Field(ge=0, strict=True, description="Số tháng đã gắn bó")
    MonthlyCharges: float = Field(gt=0, strict=True, allow_inf_nan=False, description="Tiền mỗi tháng")
    SeniorCitizen: int = Field(ge=0, le=1, strict=True, description="1 nếu là người cao tuổi")
    Contract: Literal["Month-to-month", "One year", "Two year"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    PaymentMethod: Literal[
        "Bank transfer (automatic)", "Credit card (automatic)", "Electronic check", "Mailed check"
    ]
    OnlineSecurity: Addon
    TechSupport: Addon

    @model_validator(mode="after")
    def check_internet_addons(self):
        """Không có Internet thì dịch vụ kèm phải là 'No internet service', và ngược lại."""
        for name in ("OnlineSecurity", "TechSupport"):
            if (self.InternetService == "No") != (getattr(self, name) == NO_NET):
                raise ValueError(
                    f"{name} phải là '{NO_NET}' khi và chỉ khi InternetService là 'No'"
                )
        return self


class Prediction(BaseModel):
    churn_probability: float
    will_churn: bool
    threshold: float
    trained_at: str
    warnings: list[str]


def load_bundle(path: Path) -> dict:
  
    if not path.exists():
        raise RuntimeError(f"Không thấy {path}. Hãy chạy src/train.py trước.")
    bundle = joblib.load(path)

    trained_with = bundle["meta"]["scikit_learn"]
    if trained_with != sklearn.__version__:
        raise RuntimeError(
            f"Mô hình train bằng scikit-learn {trained_with}, môi trường đang là {sklearn.__version__}"
        )
    fields = Customer.model_fields
    if set(fields) != set(bundle["numeric"] + bundle["categorical"]):
        raise RuntimeError("Các trường của API không khớp feature của mô hình. Hãy cập nhật Customer.")
    for col, values in bundle["categories"].items():
        if set(get_args(fields[col].annotation)) != set(values):
            raise RuntimeError(f"Các giá trị của {col} trong API không khớp mô hình: {values}")
    return bundle


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.bundle = load_bundle(MODEL_PATH)
    yield


app = FastAPI(title="Telco Churn API", version="1.0.0", lifespan=lifespan)


@app.exception_handler(RequestValidationError)
def invalid_request(request: Request, exc: RequestValidationError) -> JSONResponse:
   
    detail = [{k: e[k] for k in ("type", "loc", "msg")} for e in exc.errors()]
    return JSONResponse(status_code=422, content={"detail": detail})


@app.get("/health")
def health() -> dict:
    """Server chỉ khởi động được khi đã nạp xong mô hình."""
    return {"status": "ok"}


@app.get("/info")
def info(request: Request) -> dict:
    """Mô hình đang chạy: feature, ngưỡng, và điểm trên test (đo trước khi refit)."""
    b = request.app.state.bundle
    return {k: b[k] for k in ("meta", "numeric", "categorical", "threshold", "metrics")}


@app.post("/predict")
def predict(customer: Customer, request: Request) -> Prediction:
    b = request.app.state.bundle
    data = customer.model_dump()
    row = pd.DataFrame([data])[b["numeric"] + b["categorical"]]
    proba = float(b["pipeline"].predict_proba(row)[0, 1])

    warnings = [
        f"{col} = {data[col]} nằm ngoài khoảng đã học [{lo:g}, {hi:g}], dự đoán kém tin cậy"
        for col, (lo, hi) in b["numeric_ranges"].items()
        if not lo <= data[col] <= hi
    ]
    return Prediction(
        churn_probability=round(proba, 4),
        will_churn=proba >= b["threshold"],
        threshold=b["threshold"],
        trained_at=b["meta"]["trained_at"],
        warnings=warnings,
    )
