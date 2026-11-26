# ============================================================
# FreightRates - one Streamlit frontend: Rate Forecast + Rate Assistant
# ============================================================
import datetime as dt
import os

import pandas as pd
import requests
import streamlit as st

# Backend address: container name on the shared Docker network (overridable via env var)
API_URL = os.getenv("API_URL", "http://freight-api:7860").rstrip("/")
# Optional public address of the API (e.g. the Codespaces URL for port 7860), used for the Swagger link
API_PUBLIC_URL = os.getenv("API_PUBLIC_URL", "").rstrip("/")

US_PORTS = {"USLAX": "Los Angeles", "USLGB": "Long Beach", "USOAK": "Oakland", "USSEA": "Seattle",
            "USTIW": "Tacoma", "USNYC": "New York / New Jersey", "USSAV": "Savannah",
            "USHOU": "Houston", "USCHS": "Charleston"}
ASIA_PORTS = {"INNSA": "Nhava Sheva (JNPT)", "INMUN": "Mundra", "INMAA": "Chennai", "INPAV": "Pipavav",
              "INCCU": "Kolkata", "INTKD": "Tughlakabad ICD (Delhi)", "CNSHA": "Shanghai", "CNNGB": "Ningbo",
              "CNYTN": "Yantian (Shenzhen)", "CNTAO": "Qingdao", "CNTXG": "Tianjin Xingang", "CNXMN": "Xiamen"}
CONTAINERS = {"ST20": "20' standard", "ST40": "40' standard", "HC40": "40' high cube"}
EXAMPLES = [
    "What is the latest 40' rate from Shanghai to Los Angeles?",
    "Which carrier offers the cheapest 40' quote from Nhava Sheva to New York?",
    "How has the 20' rate from Qingdao to Long Beach changed over the last 13 weeks?",
    "Which US coast is cheapest for importing from China?",
    "Is Tughlakabad a seaport or an inland depot?",
]

st.set_page_config(page_title="FreightRates", page_icon="🚢", layout="centered")


def api_status() -> dict | None:
    try:
        return requests.get(f"{API_URL}/health", timeout=5).json()
    except Exception:
        return None


def show_error(resp: requests.Response):
    try:
        msg = resp.json().get("error", resp.text)
    except ValueError:
        msg = resp.text
    if resp.status_code == 503:
        st.info(f"⏳ {msg}")
    else:
        st.error(f"API error {resp.status_code}: {msg}")


# ---------------- shared sidebar: API status + docs link ----------------
status = api_status()
with st.sidebar:
    st.header("API")
    if status is None:
        st.error(f"Not reachable at {API_URL}")
    else:
        st.success("Forecast: ready")
        a = status.get("assistant")
        {"ready": st.success, "loading": st.info}.get(a, st.warning)(
            f"Assistant: {a}" + (" (about a minute after start)" if a == "loading" else ""))
    if API_PUBLIC_URL:
        st.link_button("Open API docs (Swagger)", f"{API_PUBLIC_URL}/docs", use_container_width=True)
    else:
        st.caption("API docs: open port 7860 and add **/docs** to the address.")


# =====================================================================
# Page 1: Rate Forecast
# =====================================================================
def forecast_page():
    st.title("📈 Rate Forecast")
    st.caption("Forecast next week's ocean rate between the US and India / China, in either direction.")
    tab_single, tab_batch = st.tabs(["Single prediction", "Batch prediction"])

    with tab_single:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Lane")
            direction = st.radio("Direction", ["Import: Asia → US", "Export: US → Asia"], horizontal=True)
            origins, dests = (ASIA_PORTS, US_PORTS) if direction.startswith("Import") else (US_PORTS, ASIA_PORTS)
            origin = st.selectbox("Origin port", list(origins), format_func=lambda c: f"{c} - {origins[c]}")
            dest = st.selectbox("Destination port", list(dests), format_func=lambda c: f"{c} - {dests[c]}")
            container = st.selectbox("Container", list(CONTAINERS), index=1, format_func=lambda c: f"{c} - {CONTAINERS[c]}")
            today = dt.date.today()
            week = st.date_input("Week to forecast (Monday)", value=today + dt.timedelta(days=7 - today.weekday()))
        with col2:
            st.subheader("Recent weekly prices (USD)")
            p1 = st.number_input("1 week before", min_value=1.0, value=2500.0, step=5.0)
            p2 = st.number_input("2 weeks before", min_value=1.0, value=2480.0, step=5.0)
            p3 = st.number_input("3 weeks before", min_value=1.0, value=2520.0, step=5.0)
            p4 = st.number_input("4 weeks before", min_value=1.0, value=2460.0, step=5.0)

        if st.button("Forecast rate", type="primary"):
            payload = {"origin_locode": origin, "dest_locode": dest, "container_type": container,
                       "rate_date": str(week), "price_week_1": p1, "price_week_2": p2,
                       "price_week_3": p3, "price_week_4": p4}
            try:
                resp = requests.post(f"{API_URL}/v1/predict", json=payload, timeout=30)
                if resp.status_code == 200:
                    out = resp.json()
                    st.success(f"Forecast: **USD {out['Predicted_Price']:,.2f}** ({out['Price_Unit']})")
                    st.write(f"That is **{out['Predicted_Move_Pct']:+.2f}%** vs the 4-week average of USD {out['Price_Avg_4w']:,.2f}.")
                    if out.get("Market_Data_Stale"):
                        st.warning(f"Market index data ends {out['Market_Data_Through']}; the latest week was used.")
                else:
                    show_error(resp)
            except requests.exceptions.RequestException as e:
                st.error(f"Could not reach the API at {API_URL}: {e}")

    with tab_batch:
        st.write("Upload a CSV with columns: origin_locode, dest_locode, container_type, rate_date, "
                 "price_week_1, price_week_2, price_week_3, price_week_4.")
        uploaded = st.file_uploader("Choose a CSV file", type="csv", key="forecast_csv")
        if uploaded is not None:
            batch_df = pd.read_csv(uploaded)
            st.dataframe(batch_df.head())
            if st.button("Forecast all rows", type="primary"):
                try:
                    resp = requests.post(f"{API_URL}/v1/predictbatch", files={"file": uploaded.getvalue()}, timeout=120)
                    if resp.status_code == 200:
                        preds = resp.json()
                        batch_df["Predicted_Price"] = [preds[str(i)] for i in range(len(batch_df))]
                        st.success(f"Forecast {len(batch_df)} rows.")
                        st.dataframe(batch_df)
                        st.download_button("Download forecasts", batch_df.to_csv(index=False),
                                           file_name="freight_rate_forecasts.csv", mime="text/csv")
                    else:
                        show_error(resp)
                except requests.exceptions.RequestException as e:
                    st.error(f"Could not reach the API at {API_URL}: {e}")


# =====================================================================
# Page 2: Rate Assistant (RAG)
# =====================================================================
def assistant_page():
    st.title("💬 Rate Assistant")
    st.caption("Ask about ports, rate history and carrier quotes between the US and India / China. "
               "Answers come only from the FreightRates data (synthetic test data).")

    with st.sidebar:
        st.header("Assistant")
        k = st.slider("Documents to retrieve (k)", 1, 8, 6)
        show_sources = st.toggle("Show sources", value=True)
        st.caption("Try:")
        for ex in EXAMPLES:
            if st.button(ex, use_container_width=True):
                st.session_state["pending"] = ex

    tab_chat, tab_batch = st.tabs(["Chat", "Batch questions"])

    with tab_chat:
        st.session_state.setdefault("history", [])
        for turn in st.session_state["history"]:
            with st.chat_message(turn["role"]):
                st.markdown(turn["content"])
                if turn.get("sources") and show_sources:
                    with st.expander("Sources"):
                        for s in turn["sources"]:
                            st.write(f"• {s['title']}  (score {s['score']})")

        question = st.chat_input("Ask about a lane, port, rate trend or quote…") or st.session_state.pop("pending", None)
        if question:
            st.session_state["history"].append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)
            with st.chat_message("assistant"):
                with st.spinner("Searching the rate data…"):
                    try:
                        resp = requests.post(f"{API_URL}/v1/ask", json={"question": question, "k": k}, timeout=300)
                        if resp.status_code == 200:
                            out = resp.json()
                            st.markdown(out["answer"])
                            st.caption(f"{out['seconds']} s")
                            if show_sources:
                                with st.expander("Sources"):
                                    for s in out["sources"]:
                                        st.write(f"• {s['title']}  (score {s['score']})")
                            st.session_state["history"].append({"role": "assistant", "content": out["answer"],
                                                                "sources": out["sources"]})
                        else:
                            show_error(resp)
                    except requests.exceptions.RequestException as e:
                        st.error(f"Could not reach the API at {API_URL}: {e}")

    with tab_batch:
        st.write("Upload a CSV with a **question** column (up to 50 rows).")
        uploaded = st.file_uploader("Choose a CSV file", type="csv", key="questions_csv")
        if uploaded is not None:
            batch_df = pd.read_csv(uploaded)
            st.dataframe(batch_df.head())
            if st.button("Answer all questions", type="primary"):
                with st.spinner("Answering…"):
                    try:
                        resp = requests.post(f"{API_URL}/v1/askbatch", files={"file": uploaded.getvalue()}, timeout=1800)
                        if resp.status_code == 200:
                            res = resp.json()
                            batch_df["answer"] = [res[str(i)].get("answer", res[str(i)].get("error")) for i in range(len(batch_df))]
                            batch_df["sources"] = ["; ".join(res[str(i)].get("sources", [])) for i in range(len(batch_df))]
                            st.dataframe(batch_df)
                            st.download_button("Download answers", batch_df.to_csv(index=False),
                                               file_name="freight_rag_answers.csv", mime="text/csv")
                        else:
                            show_error(resp)
                    except requests.exceptions.RequestException as e:
                        st.error(f"Could not reach the API at {API_URL}: {e}")


# ---------------- navigation ----------------
page = st.navigation([
    st.Page(forecast_page, title="Rate Forecast", icon="📈", url_path="forecast", default=True),
    st.Page(assistant_page, title="Rate Assistant", icon="💬", url_path="assistant"),
])
page.run()
