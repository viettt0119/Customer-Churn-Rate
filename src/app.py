

import os
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel

from schema import Customer, load_bundle

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = Path(os.environ.get("CHURN_MODEL_PATH", ROOT / "model" / "model.pkl"))


class Prediction(BaseModel):
    churn_probability: float
    will_churn: bool
    threshold: float
    trained_at: str
    warnings: list[str]


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.bundle = load_bundle(MODEL_PATH)
    yield


app = FastAPI(title="Telco Churn API", version="1.0.0", lifespan=lifespan)


@app.exception_handler(RequestValidationError)
def invalid_request(request: Request, exc: RequestValidationError) -> JSONResponse:
   
    detail = [{k: e[k] for k in ("type", "loc", "msg")} for e in exc.errors()]
    return JSONResponse(status_code=422, content={"detail": detail})


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse("/docs")


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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8080)
