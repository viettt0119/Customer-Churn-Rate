import hashlib
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web"
for path in (ROOT / "src", WEB):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from flask import Flask, jsonify, make_response, render_template, request, send_from_directory
from pydantic import ValidationError
from werkzeug.exceptions import HTTPException

from churn_models import LABELS, Registry
from schema import Customer

CSP = (
    "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
    "img-src 'self' data: blob:; font-src 'self'; connect-src 'self'; "
    "manifest-src 'self'; worker-src 'self'; object-src 'none'; base-uri 'self'; form-action 'self'"
)


def asset_version():
    digest = hashlib.sha256()
    for path in sorted(p for folder in ("static", "templates") for p in (WEB / folder).rglob("*") if p.is_file()):
        digest.update(path.relative_to(WEB).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()[:12]


def error(status, items):
    return jsonify({"detail": items}), status


def create_app(model_dir=None):
    model_dir = Path(model_dir or os.environ.get("CHURN_MODEL_DIR", ROOT / "model"))
    app = Flask(__name__, template_folder=str(WEB / "templates"), static_folder=str(WEB / "static"))
    app.json.sort_keys = False
    app.json.ensure_ascii = False
    app.config["MAX_CONTENT_LENGTH"] = 16 * 1024
    registry = Registry(model_dir)
    app.extensions["registry"] = registry
    version = asset_version()

    def read_request():
        body = request.get_json()
        if not isinstance(body, dict) or not isinstance(body.get("customer"), dict):
            return None, None, error(422, [{"type": "missing", "loc": ["customer"], "msg": "Thiếu thông tin khách hàng"}])
        unknown = set(body) - {"customer", "model"}
        if unknown:
            return None, None, error(422, [{"type": "extra_forbidden", "loc": [k], "msg": "Trường không hợp lệ"} for k in sorted(unknown)])
        name = body.get("model", registry.champion)
        if not isinstance(name, str) or name not in LABELS:
            return None, None, error(422, [{"type": "literal_error", "loc": ["model"], "msg": f"model phải là một trong {list(LABELS)}"}])
        try:
            customer = Customer.model_validate(body["customer"])
        except ValidationError as exc:
            items = [
                {"type": e["type"], "loc": ["customer", *e["loc"]], "msg": e["msg"]}
                for e in exc.errors(include_url=False, include_context=False, include_input=False)
            ]
            return None, None, error(422, items)
        return name, customer.model_dump(), None

    @app.after_request
    def headers(response):
        response.headers["Content-Security-Policy"] = CSP
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        if request.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.errorhandler(HTTPException)
    def http_error(exc):
        if request.path.startswith("/api/"):
            return error(exc.code, [{"type": "http_error", "loc": [], "msg": exc.name}])
        return exc

    @app.get("/")
    def index():
        return render_template("index.html", version=version)

    @app.get("/sw.js")
    def service_worker():
        response = make_response(render_template("sw.js", version=version))
        response.headers["Content-Type"] = "text/javascript; charset=utf-8"
        response.headers["Cache-Control"] = "no-cache"
        return response

    @app.get("/manifest.webmanifest")
    def manifest():
        return send_from_directory(app.static_folder, "manifest.webmanifest", mimetype="application/manifest+json")

    @app.get("/api/health")
    def health():
        return {"status": "ok"}

    @app.get("/api/models")
    def models():
        return registry.describe()

    @app.get("/api/options")
    def options():
        return {"categories": registry.categories, "ranges": registry.ranges}

    @app.get("/api/insights")
    def insights():
        return registry.insights

    @app.post("/api/predict")
    def predict():
        name, data, failure = read_request()
        return failure or registry.predict(name, data)

    @app.post("/api/tenure-curve")
    def tenure_curve():
        name, data, failure = read_request()
        return failure or registry.tenure_curve(name, data)

    return app


if __name__ == "__main__":
    from waitress import serve

    port = int(os.environ.get("PORT", 5000))
    application = create_app()
    print(f"http://127.0.0.1:{port}", flush=True)
    serve(application, host="127.0.0.1", port=port)
