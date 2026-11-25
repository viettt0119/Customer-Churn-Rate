# Cùng bản Python với môi trường train trên máy (3.12); numpy đang ghim không hỗ trợ 3.11.
FROM python:3.12-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app
# Cài thư viện trước khi copy code để tận dụng layer cache.
COPY requirements.txt .
RUN pip install -r requirements.txt

# Train lúc build: mô hình luôn khớp phiên bản thư viện trong image.
FROM base AS builder
COPY data/ data/
COPY src/train.py src/
RUN python src/train.py

# Chạy test ngay trong môi trường Linux của image: docker build --target test .
FROM builder AS test
COPY requirements-dev.txt .
RUN pip install -r requirements-dev.txt
COPY src/ src/
COPY tests/ tests/
COPY pytest.ini .
RUN python -m pytest -q -p no:cacheprovider

# Image cuối: chỉ có thư viện, API và mô hình; không có CSV khách hàng hay code train.
FROM base AS runtime
RUN useradd --create-home --uid 1000 app
COPY src/app.py src/
COPY --from=builder /app/model/ model/
USER app
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import urllib.request as u; u.urlopen('http://127.0.0.1:8080/health', timeout=3)"]
CMD ["python", "-m", "uvicorn", "app:app", "--app-dir", "src", "--host", "0.0.0.0", "--port", "8080"]
