from pathlib import Path
from typing import Literal, get_args

import joblib
import sklearn
from pydantic import BaseModel, ConfigDict, Field, model_validator

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
