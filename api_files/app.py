# FreightRates API - one Flask backend for the rate forecast and the RAG assistant, with Swagger docs
import io
import json
import logging
import os
import threading
import time

import joblib
import pandas as pd
from flask import Flask, Response, jsonify, request

import rag_core as rc
from freight_features import FEATURES, engineer_features
from openapi_spec import build_spec

# ---------- configuration (environment variables override the defaults) ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))     # paths work whatever the working directory is
FORECAST_DIR = os.getenv("FORECAST_DIR", os.path.join(BASE_DIR, "forecast"))   # model + market index (forecast notebook)
INDEX_DIR = os.getenv("INDEX_DIR", os.path.join(BASE_DIR, "rag_index"))        # documents + vectors (RAG notebook)
ENABLE_RAG = os.getenv("ENABLE_RAG", "1") == "1"          # set to 0 to run the forecast only (no LLM, ~200 MB RAM)
N_CTX = int(os.getenv("N_CTX", "4096"))
N_THREADS = int(os.getenv("N_THREADS", "0")) or None      # 0 = let llama.cpp decide
MAX_QUESTION_CHARS = 500
MAX_BATCH_QUESTIONS = 50

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("freight_api")

freight_api = Flask("FreightRates API")
freight_api.json.sort_keys = False   # keep batch results in row order

# ======================================================================================
# Forecast model (small, loads in a second)
# ======================================================================================
model = joblib.load(os.path.join(FORECAST_DIR, "freight_rate_model.joblib"))   # pipeline predicting rate_correction
market = pd.read_csv(os.path.join(FORECAST_DIR, "market_moves.csv"), parse_dates=["rate_date"])
market["rate_date"] = market["rate_date"].astype("datetime64[ns]")
with open(os.path.join(FORECAST_DIR, "model_info.json")) as f:
    model_info = json.load(f)
MARKET_THROUGH = market["rate_date"].max()
KNOWN_TRADES = set(market["trade"])
logger.info("Forecast model loaded (%s); market data through %s", model_info.get("model"), MARKET_THROUGH.date())


def forecast(raw: pd.DataFrame) -> pd.DataFrame:
    """Raw inputs -> one row per input with the forecast price and the move vs the 4-week average."""
    if raw.empty:
        raise ValueError("No rows to forecast")
    features = engineer_features(raw, market)
    unknown = sorted(set(features["trade"]) - KNOWN_TRADES)
    if unknown:
        raise ValueError(f"No market data for trade {unknown} (origin-destination country); known: {sorted(KNOWN_TRADES)}")
    correction = model.predict(features[FEATURES])
    price = correction * features["market_anchor"]          # the model adjusts the market-adjusted rule
    return pd.DataFrame({
        "Predicted_Price": price.round(2),
        "Predicted_Move_Pct": ((price / features["price_avg_4w"] - 1) * 100).round(2),
        "Price_Avg_4w": features["price_avg_4w"].round(2),
        "Market_Data_Stale": (features["rate_date"] - pd.Timedelta(weeks=1)) > MARKET_THROUGH,
    })


# ======================================================================================
# RAG assistant (embedder + LLM take ~1 minute: loaded in the background so the forecast
# and the docs are available immediately)
# ======================================================================================
with open(os.path.join(INDEX_DIR, "rag_config.json")) as f:
    rag_config = json.load(f)                  # tuned settings + model names saved by the RAG notebook
EMBED_MODEL = os.getenv("EMBED_MODEL", rag_config["embed_model"])
LLM_REPO = os.getenv("LLM_REPO", rag_config["llm_repo"])
LLM_FILE = os.getenv("LLM_FILE", rag_config["llm_file"])
if EMBED_MODEL != rag_config["embed_model"]:
    # Query and document vectors must come from the same model, or retrieval silently breaks
    raise RuntimeError(f"EMBED_MODEL={EMBED_MODEL} but the index was built with {rag_config['embed_model']}; rebuild the index.")

rag = {"status": "disabled" if not ENABLE_RAG else "loading", "error": None,
       "index": None, "embed_query": None, "chat": None}
llm_lock = threading.Lock()                    # llama.cpp handles one generation at a time


def _load_rag():
    t0 = time.time()
    try:
        rag["index"] = rc.RagIndex.load(INDEX_DIR)
        _, rag["embed_query"] = rc.load_embedder(EMBED_MODEL, device="cpu")
        rag["chat"] = rc.load_llm(LLM_REPO, LLM_FILE, n_ctx=N_CTX, n_gpu_layers=0, n_threads=N_THREADS)
        rag["status"] = "ready"
        logger.info("Assistant ready in %.0fs: %d documents; embedder %s; LLM %s/%s",
                    time.time() - t0, len(rag["index"].docs), EMBED_MODEL, LLM_REPO, LLM_FILE)
    except Exception as e:  # keep the forecast running even if the assistant cannot load
        rag["status"], rag["error"] = "failed", f"{type(e).__name__}: {e}"
        logger.exception("Assistant failed to load")


if ENABLE_RAG:
    threading.Thread(target=_load_rag, name="load-rag", daemon=True).start()


class NotReady(Exception):
    pass


def _require_rag():
    if rag["status"] == "ready":
        return
    messages = {"loading": "The assistant is still loading its models (about a minute after start). Try again shortly.",
                "disabled": "The assistant is disabled on this server (ENABLE_RAG=0).",
                "failed": f"The assistant failed to load: {rag['error']}"}
    raise NotReady(messages[rag["status"]])


# ======================================================================================
# Request helpers
# ======================================================================================
def error_message(e: Exception) -> str:
    """Readable message for a bad request (str() of a KeyError adds extra quotes)."""
    return str(e.args[0]) if isinstance(e, KeyError) and e.args else str(e)


def _int(value, default, lo, hi, name):
    if value is None:
        return default
    if isinstance(value, bool):
        raise ValueError(f"'{name}' must be an integer")
    try:
        v = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"'{name}' must be an integer")
    if not lo <= v <= hi:
        raise ValueError(f"'{name}' must be between {lo} and {hi}")
    return v


def _question(value) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("'question' is required and must be a non-empty string")
    q = value.strip()
    if len(q) > MAX_QUESTION_CHARS:
        raise ValueError(f"'question' is limited to {MAX_QUESTION_CHARS} characters")
    return q


def _uploaded_csv() -> pd.DataFrame:
    if "file" not in request.files:
        raise ValueError("No file uploaded under the key 'file'")
    try:
        return pd.read_csv(io.BytesIO(request.files["file"].read()))
    except Exception as e:
        raise ValueError(f"Could not read the uploaded file as CSV: {e}")


def ask(question: str, k: int, max_tokens: int, include_context: bool = False) -> dict:
    t0 = time.time()
    with llm_lock:
        result = rc.answer(question, rag["index"], rag["embed_query"], rag["chat"], k=k,
                           temperature=rag_config["temperature"], max_tokens=max_tokens)
    out = {"question": question, "answer": result["answer"], "sources": result["sources"],
           "understood": result["understood"], "seconds": round(time.time() - t0, 2)}
    if include_context:
        out["context"] = result["context"]
    return out


# ======================================================================================
# General routes: overview, health, Swagger
# ======================================================================================
@freight_api.get("/")
def home():
    """Overview of both services."""
    return jsonify({
        "message": "Welcome to the FreightRates API! Interactive docs: /docs",
        "docs": "/docs",
        "forecast": {"status": "ready", "endpoints": ["/v1/predict", "/v1/predictbatch"], **model_info},
        "assistant": {"status": rag["status"], "endpoints": ["/v1/ask", "/v1/askbatch", "/v1/search"],
                      "documents": len(rag["index"].docs) if rag["index"] else rag_config.get("documents"),
                      "embed_model": EMBED_MODEL, "llm": f"{LLM_REPO}/{LLM_FILE}",
                      "k": rag_config["k"], "temperature": rag_config["temperature"],
                      "data_through": rag_config.get("data_through"), "data_source": rag_config.get("data_source"),
                      **({"error": rag["error"]} if rag["error"] else {})},
    })


@freight_api.get("/health")
def health():
    """Liveness: 200 while the forecast works; the assistant's state is reported alongside."""
    return jsonify({"forecast": "ready", "assistant": rag["status"]})


@freight_api.get("/openapi.json")
def openapi_json():
    return jsonify(build_spec(model_info, rag_config))


SWAGGER_HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>FreightRates API - Swagger</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5.17.14/swagger-ui.css">
</head><body><div id="swagger-ui"></div>
<script src="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5.17.14/swagger-ui-bundle.js"></script>
<script>window.ui = SwaggerUIBundle({url: "openapi.json", dom_id: "#swagger-ui", deepLinking: true,
  displayRequestDuration: true, tryItOutEnabled: true});</script>
</body></html>"""


@freight_api.get("/docs")
def swagger_ui():
    return Response(SWAGGER_HTML, mimetype="text/html")


# ======================================================================================
# Forecast routes (same request / response shapes as the original forecast API)
# ======================================================================================
@freight_api.post("/v1/predict")
def predict_rate():
    """Online forecast: JSON for one lane and week -> forecast price."""
    try:
        payload = request.get_json(force=True, silent=True)
        if not isinstance(payload, dict):
            raise ValueError("The body must be a JSON object")
        row = forecast(pd.DataFrame([payload])).iloc[0]
        return jsonify({
            "Predicted_Price": round(float(row["Predicted_Price"]), 2),
            "Predicted_Move_Pct": round(float(row["Predicted_Move_Pct"]), 2),
            "Price_Avg_4w": round(float(row["Price_Avg_4w"]), 2),
            "Price_Unit": model_info["price_unit"],
            "Market_Data_Through": str(MARKET_THROUGH.date()),
            "Market_Data_Stale": bool(row["Market_Data_Stale"]),
        })
    except (KeyError, ValueError, TypeError) as e:
        return jsonify({"error": error_message(e)}), 400
    except Exception as e:
        logger.exception("Prediction failed")
        return jsonify({"error": f"Prediction failed: {e}"}), 500


@freight_api.post("/v1/predictbatch")
def predict_rate_batch():
    """Batch forecast: CSV file (form field 'file') -> {row_index: forecast price}."""
    try:
        result = forecast(_uploaded_csv())
        return jsonify({str(i): round(float(p), 2) for i, p in enumerate(result["Predicted_Price"])})
    except (KeyError, ValueError, TypeError) as e:
        return jsonify({"error": error_message(e)}), 400
    except Exception as e:
        logger.exception("Batch prediction failed")
        return jsonify({"error": f"Batch prediction failed: {e}"}), 500


# ======================================================================================
# Assistant routes (same request / response shapes as the original RAG API)
# ======================================================================================
@freight_api.post("/v1/ask")
def ask_one():
    """Online question: {"question": "...", "k": optional, "max_tokens": optional, "include_context": optional}."""
    try:
        body = request.get_json(force=True, silent=True) or {}
        q = _question(body.get("question"))
        k = _int(body.get("k"), rag_config["k"], 1, 8, "k")
        mt = _int(body.get("max_tokens"), rag_config["max_tokens"], 32, 800, "max_tokens")
        _require_rag()
        return jsonify(ask(q, k, mt, bool(body.get("include_context", False))))
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except NotReady as e:
        return jsonify({"error": str(e), "assistant": rag["status"]}), 503
    except Exception as e:
        logger.exception("Answer failed")
        return jsonify({"error": f"Answer failed: {e}"}), 500


@freight_api.post("/v1/askbatch")
def ask_batch():
    """Batch questions: CSV file (form field 'file') with a 'question' column -> {row_index: {answer, sources}}."""
    try:
        batch = _uploaded_csv()
        if "question" not in batch.columns:
            raise ValueError("The CSV needs a 'question' column")
        if len(batch) > MAX_BATCH_QUESTIONS:
            raise ValueError(f"Batches are limited to {MAX_BATCH_QUESTIONS} questions")
        _require_rag()
        results = {}
        for i, q in enumerate(batch["question"]):
            try:
                r = ask(_question(q), rag_config["k"], rag_config["max_tokens"])
                results[str(i)] = {"answer": r["answer"], "sources": [s["id"] for s in r["sources"]]}
            except ValueError as e:
                results[str(i)] = {"error": str(e)}
        return jsonify(results)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except NotReady as e:
        return jsonify({"error": str(e), "assistant": rag["status"]}), 503
    except Exception as e:
        logger.exception("Batch failed")
        return jsonify({"error": f"Batch failed: {e}"}), 500


@freight_api.post("/v1/search")
def search():
    """Retrieval only (no LLM, fast): {"question": "...", "k": optional} -> documents with scores and text."""
    try:
        body = request.get_json(force=True, silent=True) or {}
        q = _question(body.get("question"))
        k = _int(body.get("k"), rag_config["k"], 1, 20, "k")
        _require_rag()
        hits, info = rag["index"].search(q, rag["embed_query"](q), k=k)
        return jsonify({"question": q, "understood": info.to_dict(), "results": hits})
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except NotReady as e:
        return jsonify({"error": str(e), "assistant": rag["status"]}), 503


if __name__ == "__main__":
    # Local run (inside Docker, gunicorn is used instead - see Dockerfile)
    freight_api.run(host="0.0.0.0", port=7860, debug=False)
