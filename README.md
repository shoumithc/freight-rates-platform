# FreightRates Platform

One API and one app for the FreightRates project (ocean lanes between the US and India / China):

* **Rate forecast** - next week's FCL rate per lane (Tuned XGBoost, test MAPE 2.1%)
* **Rate assistant** - questions about ports, rate history and quotes (RAG: Qwen/Qwen3-Embedding-0.6B + qwen2.5-1.5b-instruct-q4_k_m.gguf)

Data: **SYNTHETIC** test data through 2026-11-02, not real market prices.

## Run in a Codespace
```bash
bash run_containers.sh
```
* API: port **7860**, interactive docs at **/docs** (Swagger), spec at /openapi.json
* App: port **8501**

| Route | Purpose |
|---|---|
| `POST /v1/predict`, `/v1/predictbatch` | Rate forecast (JSON / CSV upload) |
| `POST /v1/ask`, `/v1/askbatch` | Assistant answers with sources (JSON / CSV upload) |
| `POST /v1/search` | Matching documents only (fast, no LLM) |
| `GET /health` | `forecast` and `assistant` status |

Built from `shoumithc/freight-rate-forecast` and `shoumithc/freight-rag-assistant` by `FreightRates_Platform_Deployment_Notebook.ipynb`.
