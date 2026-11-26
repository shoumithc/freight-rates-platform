# OpenAPI 3.0 description of the FreightRates API (served at /openapi.json, rendered by Swagger UI at /docs)

def _error(description):
    return {"description": description,
            "content": {"application/json": {"schema": {"$ref": "#/components/schemas/Error"}}}}


def _json_body(ref, example=None):
    media = {"schema": {"$ref": f"#/components/schemas/{ref}"}}
    if example is not None:
        media["example"] = example
    return {"required": True, "content": {"application/json": media}}


def _csv_upload(description):
    return {"required": True, "content": {"multipart/form-data": {"schema": {
        "type": "object", "required": ["file"],
        "properties": {"file": {"type": "string", "format": "binary", "description": description}}}}}}


def _ok(ref, description="OK"):
    return {"description": description, "content": {"application/json": {"schema": {"$ref": f"#/components/schemas/{ref}"}}}}


def build_spec(model_info: dict, rag_config: dict) -> dict:
    not_ready = _error("The assistant is still loading its models (about a minute after start), disabled or failed to load")
    return {
        "openapi": "3.0.3",
        "info": {
            "title": "FreightRates API",
            "version": "1.0.0",
            "description": (
                "One backend for the FreightRates project, covering ocean lanes between the US and India / China in both directions.\n\n"
                f"* **Forecast**: next week's FCL rate per lane and container ({model_info.get('model')}, "
                f"test MAPE {model_info.get('test_mape', 0):.1%}, market data through {model_info.get('market_data_through')}).\n"
                f"* **Assistant**: answers questions about ports, rate history and carrier quotes with RAG "
                f"({rag_config['embed_model']} retrieval + {rag_config['llm_file']}), data through {rag_config.get('data_through')}.\n\n"
                f"Data source: {', '.join(sorted(set(model_info.get('data_source', [])) | set(rag_config.get('data_source', []))))}. "
                "SYNTHETIC means generated test data, not real market prices."),
        },
        "servers": [{"url": "/"}],
        "tags": [
            {"name": "Forecast", "description": "One-week-ahead rate forecast"},
            {"name": "Assistant", "description": "Questions about ports, rate history and quotes (RAG)"},
            {"name": "Service", "description": "Overview and health"},
        ],
        "paths": {
            "/": {"get": {"tags": ["Service"], "summary": "Overview of both services", "operationId": "overview",
                          "responses": {"200": {"description": "Model details and assistant status",
                                                "content": {"application/json": {"schema": {"type": "object"}}}}}}},
            "/health": {"get": {"tags": ["Service"], "summary": "Health check", "operationId": "health",
                                "responses": {"200": _ok("Health")}}},
            "/v1/predict": {"post": {
                "tags": ["Forecast"], "summary": "Forecast one lane and week", "operationId": "predict",
                "description": "Give the lane, container, the Monday of the week to forecast and the lane's weekly prices for the 4 weeks before it.",
                "requestBody": _json_body("ForecastRequest", {
                    "origin_locode": "CNSHA", "dest_locode": "USLAX", "container_type": "ST40", "rate_date": "2026-11-09",
                    "price_week_1": 2575, "price_week_2": 2610, "price_week_3": 2585, "price_week_4": 2555}),
                "responses": {"200": _ok("ForecastResponse"), "400": _error("Missing or invalid fields"),
                              "500": _error("Unexpected failure")}}},
            "/v1/predictbatch": {"post": {
                "tags": ["Forecast"], "summary": "Forecast many rows from a CSV", "operationId": "predictBatch",
                "requestBody": _csv_upload("CSV with columns origin_locode, dest_locode, container_type, rate_date, "
                                           "price_week_1, price_week_2, price_week_3, price_week_4"),
                "responses": {"200": _ok("ForecastBatchResponse", "Forecast price per row index"),
                              "400": _error("No file, unreadable CSV or missing columns"), "500": _error("Unexpected failure")}}},
            "/v1/ask": {"post": {
                "tags": ["Assistant"], "summary": "Ask a question", "operationId": "ask",
                "description": "Retrieves the matching lane / quote / port / market documents and answers only from them, with sources. "
                               "Takes roughly 10-40 seconds on a CPU.",
                "requestBody": _json_body("AskRequest", {
                    "question": "Which carrier offers the cheapest 40' quote from Nhava Sheva to New York?"}),
                "responses": {"200": _ok("AskResponse"), "400": _error("Invalid question or parameters"),
                              "503": not_ready, "500": _error("Unexpected failure")}}},
            "/v1/askbatch": {"post": {
                "tags": ["Assistant"], "summary": "Answer up to 50 questions from a CSV", "operationId": "askBatch",
                "requestBody": _csv_upload("CSV with a 'question' column (up to 50 rows)"),
                "responses": {"200": _ok("AskBatchResponse", "Answer (or error) per row index"),
                              "400": _error("No file, no 'question' column or too many rows"), "503": not_ready,
                              "500": _error("Unexpected failure")}}},
            "/v1/search": {"post": {
                "tags": ["Assistant"], "summary": "Find the matching documents (no LLM, fast)", "operationId": "search",
                "requestBody": _json_body("SearchRequest", {"question": "Is Tughlakabad an inland depot?", "k": 3}),
                "responses": {"200": _ok("SearchResponse"), "400": _error("Invalid question or parameters"), "503": not_ready}}},
        },
        "components": {"schemas": {
            "Error": {"type": "object", "properties": {"error": {"type": "string"}, "assistant": {"type": "string"}}},
            "Health": {"type": "object", "properties": {
                "forecast": {"type": "string", "example": "ready"},
                "assistant": {"type": "string", "enum": ["loading", "ready", "failed", "disabled"]}}},
            "ForecastRequest": {"type": "object", "required": [
                "origin_locode", "dest_locode", "container_type", "rate_date",
                "price_week_1", "price_week_2", "price_week_3", "price_week_4"], "properties": {
                "origin_locode": {"type": "string", "description": "UN/LOCODE, e.g. CNSHA, INNSA, USLAX"},
                "dest_locode": {"type": "string", "description": "UN/LOCODE"},
                "container_type": {"type": "string", "enum": ["ST20", "ST40", "HC40"]},
                "rate_date": {"type": "string", "format": "date", "description": "Monday of the week to forecast"},
                "price_week_1": {"type": "number", "minimum": 0, "exclusiveMinimum": True, "description": "Price 1 week before (USD)"},
                "price_week_2": {"type": "number", "minimum": 0, "exclusiveMinimum": True},
                "price_week_3": {"type": "number", "minimum": 0, "exclusiveMinimum": True},
                "price_week_4": {"type": "number", "minimum": 0, "exclusiveMinimum": True}}},
            "ForecastResponse": {"type": "object", "properties": {
                "Predicted_Price": {"type": "number"}, "Predicted_Move_Pct": {"type": "number"},
                "Price_Avg_4w": {"type": "number"}, "Price_Unit": {"type": "string"},
                "Market_Data_Through": {"type": "string", "format": "date"},
                "Market_Data_Stale": {"type": "boolean", "description": "True when the forecast week is beyond the market index data"}}},
            "ForecastBatchResponse": {"type": "object", "additionalProperties": {"type": "number"},
                                      "example": {"0": 2581.4, "1": 1312.75}},
            "AskRequest": {"type": "object", "required": ["question"], "properties": {
                "question": {"type": "string", "maxLength": 500},
                "k": {"type": "integer", "minimum": 1, "maximum": 8, "description": f"Documents to retrieve (default {rag_config['k']})"},
                "max_tokens": {"type": "integer", "minimum": 32, "maximum": 800, "description": f"Answer length limit (default {rag_config['max_tokens']})"},
                "include_context": {"type": "boolean", "default": False, "description": "Also return the retrieved text"}}},
            "Source": {"type": "object", "properties": {
                "id": {"type": "string", "example": "quote:INNSA-USNYC:ST40"},
                "type": {"type": "string", "enum": ["lane", "quote", "port", "trade"]},
                "title": {"type": "string"}, "score": {"type": "number"}}},
            "Understood": {"type": "object", "description": "What the question parser recognised", "properties": {
                "ports": {"type": "array", "items": {"type": "string"}},
                "countries": {"type": "array", "items": {"type": "string"}},
                "container": {"type": "string", "nullable": True},
                "type_hints": {"type": "array", "items": {"type": "string"}},
                "direction": {"type": "string", "nullable": True, "enum": ["import", "export", None]}}},
            "AskResponse": {"type": "object", "properties": {
                "question": {"type": "string"}, "answer": {"type": "string"},
                "sources": {"type": "array", "items": {"$ref": "#/components/schemas/Source"}},
                "understood": {"$ref": "#/components/schemas/Understood"},
                "seconds": {"type": "number"}, "context": {"type": "string"}}},
            "AskBatchResponse": {"type": "object", "additionalProperties": {"type": "object", "properties": {
                "answer": {"type": "string"}, "sources": {"type": "array", "items": {"type": "string"}},
                "error": {"type": "string"}}}},
            "SearchRequest": {"type": "object", "required": ["question"], "properties": {
                "question": {"type": "string", "maxLength": 500},
                "k": {"type": "integer", "minimum": 1, "maximum": 20}}},
            "SearchResponse": {"type": "object", "properties": {
                "question": {"type": "string"}, "understood": {"$ref": "#/components/schemas/Understood"},
                "results": {"type": "array", "items": {"type": "object", "properties": {
                    "id": {"type": "string"}, "type": {"type": "string"}, "title": {"type": "string"},
                    "score": {"type": "number"}, "text": {"type": "string"}}}}}},
        }},
    }
