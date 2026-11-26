"""Freight RAG core - shared by the notebook and the Flask API.

Turns the FreightRates exports (ports, weekly rate history, carrier quotes) into
text documents, embeds them, retrieves the right ones for a question (port / lane /
container filters + embedding similarity) and builds the prompt for the LLM.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- reference data
TRADE_NAMES = {"US-IN": "US → India", "US-CN": "US → China", "CN-US": "China → US", "IN-US": "India → US"}
COUNTRY_NAMES = {"US": "United States", "IN": "India", "CN": "China"}
US_COAST = {"USLAX": "West Coast", "USLGB": "West Coast", "USOAK": "West Coast", "USSEA": "West Coast",
            "USTIW": "West Coast", "USNYC": "East Coast", "USSAV": "East Coast", "USCHS": "East Coast",
            "USHOU": "Gulf Coast"}
CONTAINER_NAMES = {"ST20": "20' standard", "ST40": "40' standard", "HC40": "40' high cube", "LCL": "LCL (per cbm)"}
CHARGE_NAMES = {"OTHC": "Terminal Handling Charge - Origin", "OF": "Ocean Freight", "BAF": "Bunker Adjustment Factor",
                "DTHC": "Terminal Handling Charge - Destination", "DOC": "Documentation Fee"}

# Extra names people use for ports (lower case). Port names from the data are added automatically.
PORT_ALIASES = {
    "la": "USLAX", "l.a.": "USLAX", "los angeles": "USLAX", "long beach": "USLGB", "oakland": "USOAK",
    "seattle": "USSEA", "tacoma": "USTIW", "new york": "USNYC", "nyc": "USNYC", "new jersey": "USNYC",
    "ny/nj": "USNYC", "newark": "USNYC", "savannah": "USSAV", "houston": "USHOU", "charleston": "USCHS",
    "nhava sheva": "INNSA", "jnpt": "INNSA", "jawaharlal nehru": "INNSA", "mumbai": "INNSA", "navi mumbai": "INNSA",
    "mundra": "INMUN", "chennai": "INMAA", "madras": "INMAA", "pipavav": "INPAV", "kolkata": "INCCU",
    "calcutta": "INCCU", "tughlakabad": "INTKD", "delhi": "INTKD", "new delhi": "INTKD", "icd delhi": "INTKD",
    "shanghai": "CNSHA", "ningbo": "CNNGB", "yantian": "CNYTN", "shenzhen": "CNYTN", "qingdao": "CNTAO",
    "tianjin": "CNTXG", "xingang": "CNTXG", "xiamen": "CNXMN",
}
COUNTRY_ALIASES = {"china": "CN", "chinese": "CN", "india": "IN", "indian": "IN",
                   "united states": "US", "usa": "US", "u.s.": "US", "america": "US", "american": "US"}
CONTAINER_PATTERNS = [   # checked in order: high cube before plain 40'; a size needs a unit (so "20 days" is not a 20')
    ("HC40", r"(?<![a-z0-9])(40\s*(?:'|ft|foot|feet|-foot)?\s*(hc|hq)|hc\s*-?40|high[\s-]*cube)(?![a-z0-9])"),
    ("ST20", r"(?<![a-z0-9])(20\s*(?:'|ft|foot|feet|-foot|gp|dv|st)|st\s*-?20|twenty[\s-]*foot)"),
    ("ST40", r"(?<![a-z0-9])(40\s*(?:'|ft|foot|feet|-foot|gp|dv|st)|st\s*-?40|forty[\s-]*foot)"),
    ("LCL", r"(?<![a-z])(lcl|less than container|per cbm|cubic met)"),
]
DOC_TYPE_HINTS = {
    "quote": r"\b(quote|quotes|quoted|carrier|carriers|offer|offers|book|booking|cheapest carrier|validity|valid until|surcharge|baf|thc|charge|charges|breakdown)\b",
    "lane": r"\b(history|historical|trend|trending|last year|weeks? ago|52[\s-]*week|average|averaged|highest|lowest|peak|changed|change|latest rate|current rate|rate history|over time|quarter)\b",
    "port": r"\b(port|ports|terminal|inland depot|icd|seaport|which ports|located)\b",
    "trade": r"\b(market|markets|overall|seasonality|season|peak season|index|import market|export market|in general|coast|coasts|importing|exporting)\b",
}


def _fmt_money(x: float) -> str:
    return f"USD {x:,.2f}" if abs(x) < 1000 and x != round(x) else f"USD {x:,.0f}"


def _pct(a: float, b: float) -> str:
    return "n/a" if not b else f"{(a / b - 1) * 100:+.1f}%"


def _bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin(["t", "true", "1", "yes"])


# ---------------------------------------------------------------- loading
def load_sources(port_csv: str, history_csv: str, quote_csv: str | None = None):
    """Read the three pgAdmin exports (comma- or tab-separated)."""
    ports = pd.read_csv(port_csv, sep=None, engine="python")
    ports["locode"] = ports["locode"].str.strip()
    ports["is_inland_depot"] = _bool(ports["is_inland_depot"])

    hist = pd.read_csv(history_csv, sep=None, engine="python")
    hist["rate_date"] = pd.to_datetime(hist["rate_date"])
    hist["container_type"] = hist["container_type"].fillna("LCL")

    quotes = None
    if quote_csv:
        quotes = pd.read_csv(quote_csv, sep=None, engine="python")
        quotes["valid_from"] = pd.to_datetime(quotes["valid_from"])
        quotes["valid_to"] = pd.to_datetime(quotes["valid_to"])
    return ports, hist, quotes


@dataclass
class Doc:
    id: str
    type: str                     # port | lane | trade | quote
    title: str
    text: str
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"id": self.id, "type": self.type, "title": self.title, "text": self.text, "meta": self.meta}


def _port_label(code: str, port_names: dict) -> str:
    name = port_names.get(code, code)
    if code.startswith("US"):
        return f"{code} ({name}, US {US_COAST.get(code, '')})".replace(" )", ")")
    return f"{code} ({name}, {COUNTRY_NAMES.get(code[:2], code[:2])})"


def _trade(origin: str, dest: str) -> str:
    return f"{origin[:2]}-{dest[:2]}"


# ---------------------------------------------------------------- documents
def build_lane_docs(hist: pd.DataFrame, port_names: dict) -> list[Doc]:
    """One document per lane and load type: latest rate, recent changes, 52-week and 2-year range, quarterly averages."""
    docs = []
    last_week = hist["rate_date"].max()
    for (o, d, c), g in hist.groupby(["origin_locode", "dest_locode", "container_type"]):
        g = g.sort_values("rate_date")
        p = g.set_index("rate_date")["price"]
        latest_date, latest = p.index[-1], p.iloc[-1]
        unit = "per cbm" if c == "LCL" else "per container"
        prev4 = p.iloc[-5:-1].mean() if len(p) >= 5 else np.nan
        ago = lambda w: p.iloc[-1 - w] if len(p) > w else np.nan
        y = p[p.index > latest_date - pd.Timedelta(weeks=52)]
        q = p.groupby(p.index.to_period("Q")).mean()
        trade = _trade(o, d)
        lines = [
            f"Rate history: {_port_label(o, port_names)} to {_port_label(d, port_names)}, {CONTAINER_NAMES[c]}.",
            f"Trade: {TRADE_NAMES.get(trade, trade)} ({'import to the US' if d.startswith('US') else 'export from the US'}). "
            f"Typical port-to-port transit: {int(g['transit_days'].iloc[-1])} days. Prices in USD {unit}.",
            f"Latest weekly rate (week of {latest_date:%Y-%m-%d}): {_fmt_money(latest)}.",
            f"Average of the previous 4 weeks: {_fmt_money(prev4)}; change vs 4 weeks ago {_pct(latest, ago(4))}, "
            f"vs 13 weeks ago {_pct(latest, ago(13))}, vs 52 weeks ago {_pct(latest, ago(52))}.",
            f"52-week low {_fmt_money(y.min())} (week of {y.idxmin():%Y-%m-%d}); 52-week high {_fmt_money(y.max())} (week of {y.idxmax():%Y-%m-%d}); 52-week average {_fmt_money(y.mean())}.",
            f"Full history {p.index[0]:%Y-%m-%d} to {latest_date:%Y-%m-%d} ({len(p)} weeks): low {_fmt_money(p.min())} (week of {p.idxmin():%Y-%m-%d}), "
            f"high {_fmt_money(p.max())} (week of {p.idxmax():%Y-%m-%d}).",
            "Quarterly averages: " + "; ".join(f"{str(k)} {_fmt_money(v)}" for k, v in q.items()) + ".",
            f"Data source: {', '.join(sorted(g['source'].astype(str).unique()))}.",
        ]
        docs.append(Doc(
            id=f"lane:{o}-{d}:{c}", type="lane",
            title=f"Rate history {o} → {d} {c}",
            text="\n".join(lines),
            meta={"origin": o, "dest": d, "container": c, "trade": trade, "ports": [o, d],
                  "latest_price": float(latest), "latest_week": f"{latest_date:%Y-%m-%d}",
                  "transit_days": int(g["transit_days"].iloc[-1]), "data_through": f"{last_week:%Y-%m-%d}"}))
    return docs


def build_port_docs(ports: pd.DataFrame, hist: pd.DataFrame) -> list[Doc]:
    """One document per port: location, role, partners, transit range and latest 40' prices in and out."""
    names = dict(zip(ports["locode"], ports["name"]))
    latest = hist[hist["rate_date"] == hist["rate_date"].max()]
    st40 = latest[latest["container_type"] == "ST40"]
    docs = []
    for _, r in ports.iterrows():
        code = r["locode"]
        out = st40[st40["origin_locode"] == code].sort_values("price")
        inn = st40[st40["dest_locode"] == code].sort_values("price")
        lanes = hist[(hist["origin_locode"] == code) | (hist["dest_locode"] == code)]
        if code.startswith("US"):
            where = f"in the United States ({US_COAST.get(code, 'coast unknown')})"
        else:
            where = f"in {COUNTRY_NAMES.get(r['country_code'], r['country_code'])}"
        kind = "an inland container depot (ICD), reached by rail/road from the coast" if r["is_inland_depot"] else "a seaport"
        lines = [f"Port profile: {code} - {r['name']}, {kind} {where}. Coordinates {r['latitude']:.3f}, {r['longitude']:.3f}."]
        if len(lanes):
            partners = sorted((set(lanes["origin_locode"]) | set(lanes["dest_locode"])) - {code})
            lines.append(f"Lanes in the rate data: {lanes[['origin_locode', 'dest_locode']].drop_duplicates().shape[0]} "
                         f"(to and from {len(partners)} ports: {', '.join(partners)}). "
                         f"Transit times {int(lanes['transit_days'].min())}-{int(lanes['transit_days'].max())} days.")
        week = f"{hist['rate_date'].max():%Y-%m-%d}"
        if len(out):
            lines.append(f"Outbound 40' standard rates, week of {week}: average {_fmt_money(out['price'].mean())}; "
                         f"cheapest to {_port_label(out.iloc[0]['dest_locode'], names)} at {_fmt_money(out.iloc[0]['price'])}; "
                         f"dearest to {_port_label(out.iloc[-1]['dest_locode'], names)} at {_fmt_money(out.iloc[-1]['price'])}.")
        if len(inn):
            lines.append(f"Inbound 40' standard rates, week of {week}: average {_fmt_money(inn['price'].mean())}; "
                         f"cheapest from {_port_label(inn.iloc[0]['origin_locode'], names)} at {_fmt_money(inn.iloc[0]['price'])}; "
                         f"dearest from {_port_label(inn.iloc[-1]['origin_locode'], names)} at {_fmt_money(inn.iloc[-1]['price'])}.")
        docs.append(Doc(id=f"port:{code}", type="port", title=f"Port {code} {r['name']}", text="\n".join(lines),
                        meta={"ports": [code], "country": r["country_code"]}))
    return docs


def build_trade_docs(hist: pd.DataFrame, port_names: dict) -> list[Doc]:
    """One document per trade: market level, 52-week change, peak/trough, seasonality, cheapest/dearest lanes, coasts."""
    docs = []
    h = hist[hist["container_type"] == "ST40"].copy()
    h["trade"] = h["origin_locode"].str[:2] + "-" + h["dest_locode"].str[:2]
    h["us_port"] = np.where(h["origin_locode"].str.startswith("US"), h["origin_locode"], h["dest_locode"])
    h["coast"] = h["us_port"].map(US_COAST)
    last = h["rate_date"].max()
    for trade, g in h.groupby("trade"):
        weekly = g.groupby("rate_date")["price"].mean()
        idx = weekly / weekly.mean()
        rel = g["price"] / g.groupby(["origin_locode", "dest_locode"])["price"].transform("mean")
        month = rel.groupby(g["rate_date"].dt.month).mean()
        lanes = g[g["rate_date"] == last].sort_values("price")
        coast = g[g["rate_date"] == last].groupby("coast")["price"].mean().sort_values()
        lines = [
            f"Market summary: {TRADE_NAMES.get(trade, trade)} trade ({'imports to' if trade.endswith('US') else 'exports from'} the US), 40' standard containers.",
            f"Average 40' rate across all {g[['origin_locode', 'dest_locode']].drop_duplicates().shape[0]} lanes, week of {last:%Y-%m-%d}: {_fmt_money(weekly.iloc[-1])} "
            f"({_pct(weekly.iloc[-1], weekly.iloc[-14] if len(weekly) > 13 else np.nan)} vs 13 weeks earlier, "
            f"{_pct(weekly.iloc[-1], weekly.iloc[-53] if len(weekly) > 52 else np.nan)} vs 52 weeks earlier).",
            f"Market index (1.00 = average {weekly.index[0]:%Y-%m-%d} to {last:%Y-%m-%d}): now {idx.iloc[-1]:.2f}; peak {idx.max():.2f} in the week of {idx.idxmax():%Y-%m-%d}; low {idx.min():.2f} in the week of {idx.idxmin():%Y-%m-%d}.",
            "Seasonality (average price vs each lane's own average, by month): " +
            "; ".join(f"{pd.Timestamp(2000, m, 1):%b} {v:.2f}" for m, v in month.items()) + ".",
            f"Cheapest lane now: {_port_label(lanes.iloc[0]['origin_locode'], port_names)} to {_port_label(lanes.iloc[0]['dest_locode'], port_names)} at {_fmt_money(lanes.iloc[0]['price'])}. "
            f"Dearest: {_port_label(lanes.iloc[-1]['origin_locode'], port_names)} to {_port_label(lanes.iloc[-1]['dest_locode'], port_names)} at {_fmt_money(lanes.iloc[-1]['price'])}.",
            "Average 40' rate by US coast now: " + "; ".join(f"{k} {_fmt_money(v)}" for k, v in coast.items()) + f" (cheapest: {coast.index[0]}).",
        ]
        docs.append(Doc(id=f"trade:{trade}", type="trade", title=f"Market {TRADE_NAMES.get(trade, trade)}", text="\n".join(lines),
                        meta={"trade": trade, "countries": sorted({trade[:2], trade[3:]})}))
    return docs


def build_quote_docs(quotes: pd.DataFrame, hist: pd.DataFrame, port_names: dict) -> list[Doc]:
    """One document per lane and container listing every carrier quote, cheapest first, with charge breakdowns."""
    if quotes is None or quotes.empty:
        return []
    latest = (hist.sort_values("rate_date").groupby(["origin_locode", "dest_locode", "container_type"])
                  .tail(1).set_index(["origin_locode", "dest_locode", "container_type"])["price"])
    docs = []
    for (o, d, c), g in quotes.groupby(["origin_locode", "dest_locode", "container_type"]):
        g = g.sort_values("price")
        trade = _trade(o, d)
        market = latest.get((o, d, c), np.nan)
        lines = [f"Carrier quotes: {_port_label(o, port_names)} to {_port_label(d, port_names)}, {CONTAINER_NAMES.get(c, c)}. "
                 f"{len(g)} quotes; all-in prices in USD per container."]
        for i, (_, q) in enumerate(g.iterrows(), 1):
            parts = []
            for item in str(q.get("charges", "") or "").split(";"):
                if "=" in item:
                    code, amt = item.split("=", 1)
                    parts.append(f"{CHARGE_NAMES.get(code.strip(), code.strip())} {_fmt_money(float(amt))}")
            lines.append(f"{i}. {q['carrier']}: {_fmt_money(q['price'])} all-in, transit {int(q['transit_days'])} days, "
                         f"valid {q['valid_from']:%Y-%m-%d} to {q['valid_to']:%Y-%m-%d}, reference {q['provider_ref']}"
                         + (f". Breakdown: {', '.join(parts)}." if parts else "."))
        fastest = g.sort_values(["transit_days", "price"]).iloc[0]
        lines.append(f"Cheapest: {g.iloc[0]['carrier']} ({_fmt_money(g.iloc[0]['price'])}). Fastest: {fastest['carrier']} ({int(fastest['transit_days'])} days). "
                     f"Spread between cheapest and dearest: {_fmt_money(g['price'].max() - g['price'].min())}.")
        if not np.isnan(market):
            lines.append(f"Latest weekly market rate on this lane: {_fmt_money(market)}; cheapest quote is {_pct(g.iloc[0]['price'], market)} vs market.")
        lines.append(f"Quote source: {', '.join(sorted(g['provider'].astype(str).unique()))}.")
        docs.append(Doc(id=f"quote:{o}-{d}:{c}", type="quote", title=f"Quotes {o} → {d} {c}", text="\n".join(lines),
                        meta={"origin": o, "dest": d, "container": c, "trade": trade, "ports": [o, d],
                              "cheapest_carrier": g.iloc[0]["carrier"], "cheapest_price": float(g.iloc[0]["price"]),
                              "fastest_carrier": fastest["carrier"]}))
    return docs


def build_documents(ports: pd.DataFrame, hist: pd.DataFrame, quotes: pd.DataFrame | None) -> list[Doc]:
    names = dict(zip(ports["locode"], ports["name"]))
    return (build_port_docs(ports, hist) + build_trade_docs(hist, names)
            + build_lane_docs(hist, names) + build_quote_docs(quotes, hist, names))


# ---------------------------------------------------------------- understanding the question
@dataclass
class QueryInfo:
    ports: list            # LOCODEs in the order they appear
    countries: list        # country codes mentioned without a specific port
    container: str | None
    type_hints: list
    direction: str | None = None    # "import" (into the US) / "export" (out of the US) when the question says so

    def to_dict(self) -> dict:
        return {"ports": self.ports, "countries": self.countries, "container": self.container,
                "type_hints": self.type_hints, "direction": self.direction}


def build_alias_map(ports: pd.DataFrame) -> dict:
    aliases = dict(PORT_ALIASES)
    for code, name in zip(ports["locode"], ports["name"]):
        aliases[code.lower()] = code
        base = re.sub(r"\(.*?\)", "", name).strip().lower()
        aliases.setdefault(base, code)
    return aliases


def parse_question(question: str, aliases: dict) -> QueryInfo:
    q = " " + question.lower() + " "
    found = []   # (position, code)
    taken = []
    for alias in sorted(aliases, key=len, reverse=True):      # longest first: "new york" before "york"
        for m in re.finditer(r"(?<![a-z0-9])" + re.escape(alias) + r"(?![a-z0-9])", q):
            span = range(m.start(), m.end())
            if any(s in t for t in taken for s in (m.start(), m.end() - 1)):
                continue
            taken.append(span)
            found.append((m.start(), aliases[alias]))
    ports = []
    for _, code in sorted(found):
        if code not in ports:
            ports.append(code)
    countries = []
    for word, cc in COUNTRY_ALIASES.items():
        if re.search(r"(?<![a-z])" + re.escape(word) + r"(?![a-z])", q) and cc not in countries:
            countries.append(cc)
    if re.search(r"(?<![a-z])(us|u\.s)(?![a-z])", q) and "US" not in countries:
        countries.append("US")
    container = None
    for code, pattern in CONTAINER_PATTERNS:
        if re.search(pattern, q):
            container = code
            break
    hints = [t for t, pattern in DOC_TYPE_HINTS.items() if re.search(pattern, q)]
    direction = None
    if re.search(r"\b(import|imports|importing|imported|importer|inbound to the us|into the us)\b", q):
        direction = "import"
    elif re.search(r"\b(export|exports|exporting|exported|exporter|out of the us|from the us)\b", q):
        direction = "export"
    return QueryInfo(ports, countries, container, hints, direction)


# ---------------------------------------------------------------- index + retrieval
class RagIndex:
    """Documents + their normalized embedding vectors, searched with cosine similarity."""

    def __init__(self, docs: list[dict], vectors: np.ndarray, aliases: dict):
        self.docs = docs
        v = np.asarray(vectors, dtype=np.float32)
        self.vectors = v / np.clip(np.linalg.norm(v, axis=1, keepdims=True), 1e-12, None)
        self.aliases = aliases
        self.coverage = coverage_text(docs)

    # -- persistence
    def save(self, folder: str):
        import os
        os.makedirs(folder, exist_ok=True)
        np.save(os.path.join(folder, "embeddings.npy"), self.vectors.astype(np.float16))
        with open(os.path.join(folder, "documents.jsonl"), "w", encoding="utf-8") as f:
            for d in self.docs:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
        with open(os.path.join(folder, "aliases.json"), "w", encoding="utf-8") as f:
            json.dump(self.aliases, f, ensure_ascii=False, indent=0)

    @classmethod
    def load(cls, folder: str) -> "RagIndex":
        import os
        with open(os.path.join(folder, "documents.jsonl"), encoding="utf-8") as f:
            docs = [json.loads(line) for line in f if line.strip()]
        with open(os.path.join(folder, "aliases.json"), encoding="utf-8") as f:
            aliases = json.load(f)
        return cls(docs, np.load(os.path.join(folder, "embeddings.npy")).astype(np.float32), aliases)

    # -- candidate selection from what the question names
    def _candidates(self, info: QueryInfo, use_filters: bool) -> np.ndarray:
        n = len(self.docs)
        if not use_filters:
            return np.arange(n)
        keep = np.ones(n, dtype=bool)
        ports = info.ports[:2]
        for i, d in enumerate(self.docs):
            m, t = d["meta"], d["type"]
            if ports:
                if t in ("lane", "quote"):
                    keep[i] = all(p in m["ports"] for p in ports)
                elif t == "port":
                    keep[i] = m["ports"][0] in ports
                elif t == "trade":
                    # the trade(s) these ports belong to
                    keep[i] = any(p[:2] in m["countries"] for p in ports) and (len(ports) < 2 or
                              {ports[0][:2], ports[1][:2]} == set(m["countries"]))
            elif info.countries:
                cs = set(info.countries) - {"US"} if len(info.countries) > 1 else set(info.countries)
                if t in ("lane", "quote"):
                    keep[i] = any(m["trade"].startswith(c) or m["trade"].endswith(c) for c in cs)
                elif t == "port":
                    keep[i] = m.get("country") in cs
                elif t == "trade":
                    keep[i] = bool(cs & set(m["countries"]))
            if keep[i] and info.container and t in ("lane", "quote"):
                keep[i] = m["container"] == info.container
        idx = np.flatnonzero(keep)
        return idx if len(idx) else np.arange(n)

    def search(self, question: str, query_vector: np.ndarray, k: int = 4, use_filters: bool = True):
        info = parse_question(question, self.aliases)
        cand = self._candidates(info, use_filters)
        q = np.asarray(query_vector, dtype=np.float32).ravel()
        q = q / max(np.linalg.norm(q), 1e-12)
        scores = self.vectors[cand] @ q
        if use_filters:
            boost = np.zeros(len(cand), dtype=np.float32)
            for j, i in enumerate(cand):
                d = self.docs[i]
                if d["type"] in info.type_hints:
                    boost[j] += 0.05
                m = d["meta"]
                if len(info.ports) >= 2 and m.get("origin") == info.ports[0] and m.get("dest") == info.ports[1]:
                    boost[j] += 0.08          # the direction the question asked for ("from X to Y")
                if not info.container and m.get("container") == "ST40":
                    boost[j] += 0.02          # default to the 40' when no container is named
                trade = m.get("trade", "")
                if info.direction and trade:
                    if (info.direction == "import") == trade.endswith("US"):
                        boost[j] += 0.06      # the direction the question asked about
            scores = scores + boost
        order = np.argsort(-scores)[:k]
        hits = [{"id": self.docs[cand[j]]["id"], "type": self.docs[cand[j]]["type"],
                 "title": self.docs[cand[j]]["title"], "score": round(float(scores[j]), 4),
                 "text": self.docs[cand[j]]["text"]} for j in order]
        return hits, info


# ---------------------------------------------------------------- prompting
SYSTEM_PROMPT = """You are FreightRates Assistant, an expert on ocean freight between the United States and India / China.
Answer the question using ONLY the facts in the CONTEXT documents.
- Quote numbers exactly as they appear (USD amounts, dates, days); do not round or estimate.
- Name the lane, container type and week or validity dates your numbers refer to.
- If the context does not contain the answer, say "I don't have that in the FreightRates data." and stop.
- Be concise: at most 5 sentences or a short list.
- The data is SYNTHETIC test data, not real market prices; mention this only if asked whether figures are real.
{coverage}"""

USER_TEMPLATE = """CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""


def format_context(hits: list[dict]) -> str:
    return "\n\n".join(f"[{i}] {h['title']}\n{h['text']}" for i, h in enumerate(hits, 1))


def coverage_text(docs: list[dict]) -> str:
    """Which ports the data covers, so the model can refuse questions about anywhere else."""
    ports = [d for d in docs if d["type"] == "port"]
    by = {}
    for d in ports:
        by.setdefault(d["meta"]["country"], []).append(d["title"].split(" ", 2)[1] + " " + d["title"].split(" ", 2)[2])
    parts = [f"{COUNTRY_NAMES.get(c, c)}: {', '.join(sorted(v))}" for c, v in sorted(by.items())]
    return ("- The data ONLY covers lanes between these ports (any other port or country is not covered): "
            + "; ".join(parts) + ".") if parts else ""


def build_messages(question: str, hits: list[dict], coverage: str = "") -> list[dict]:
    return [{"role": "system", "content": SYSTEM_PROMPT.format(coverage=coverage)},
            {"role": "user", "content": USER_TEMPLATE.format(context=format_context(hits), question=question)}]


# ---------------------------------------------------------------- models (loaded lazily; heavy imports inside)
def load_embedder(model_name: str = "Qwen/Qwen3-Embedding-0.6B", device: str | None = None):
    """Returns (embed_documents, embed_query) callables producing normalized numpy vectors."""
    from sentence_transformers import SentenceTransformer
    # On CPU, load in float32: the model's default bfloat16 is much slower on most CPUs
    kwargs = {"model_kwargs": {"dtype": "float32"}} if device == "cpu" else {}
    model = SentenceTransformer(model_name, device=device, **kwargs)

    def embed_documents(texts, batch_size: int = 16, show_progress_bar: bool = False):
        return model.encode_document(list(texts), batch_size=batch_size, normalize_embeddings=True,
                                     convert_to_numpy=True, show_progress_bar=show_progress_bar)

    def embed_query(text):
        # Qwen3-Embedding uses an instruction prompt for queries ("query" prompt in the model config)
        return model.encode_query([text], normalize_embeddings=True, convert_to_numpy=True)[0]

    return embed_documents, embed_query


def load_llm(repo_id: str, filename: str, n_ctx: int = 4096, n_gpu_layers: int = 0, n_threads: int | None = None):
    """Downloads (once) and loads a GGUF chat model with llama-cpp-python. Returns chat(messages, ...) -> str."""
    from huggingface_hub import hf_hub_download
    from llama_cpp import Llama
    # hf_hub_download uses the local cache (works with HF_HUB_OFFLINE=1); Llama.from_pretrained always lists the repo online
    model_path = hf_hub_download(repo_id=repo_id, filename=filename)
    llm = Llama(model_path=model_path, n_ctx=n_ctx, n_gpu_layers=n_gpu_layers, n_threads=n_threads, verbose=False)

    def chat(messages, max_tokens: int = 300, temperature: float = 0.0, top_p: float = 0.95):
        out = llm.create_chat_completion(messages=messages, max_tokens=max_tokens,
                                         temperature=temperature, top_p=top_p)
        return out["choices"][0]["message"]["content"].strip()

    return chat


def answer(question: str, index: RagIndex, embed_query, chat, k: int = 4,
           temperature: float = 0.0, max_tokens: int = 300, use_filters: bool = True) -> dict:
    """Full RAG round trip: retrieve -> prompt -> generate."""
    hits, info = index.search(question, embed_query(question), k=k, use_filters=use_filters)
    reply = chat(build_messages(question, hits, index.coverage), max_tokens=max_tokens, temperature=temperature)
    return {"answer": reply, "sources": [{k_: h[k_] for k_ in ("id", "type", "title", "score")} for h in hits],
            "understood": info.to_dict(), "context": format_context(hits)}


# ---------------------------------------------------------------- evaluation set with known answers
def build_eval_set(docs: list[dict], port_names: dict, n_per_kind: int = 6, seed: int = 1) -> list[dict]:
    """Questions whose correct answer (and correct source document) is known from the data.
    expected: text that must appear in a correct answer; expected_docs: any of these ids in the top-k is a retrieval hit."""
    import random
    rng = random.Random(seed)
    lane = [d for d in docs if d["type"] == "lane" and d["meta"]["container"] != "LCL"]
    quote = [d for d in docs if d["type"] == "quote"]
    nm = lambda c: re.sub(r"\(.*?\)", "", port_names.get(c, c)).strip()
    items = []
    for d in rng.sample(lane, n_per_kind):
        m = d["meta"]
        items.append({"kind": "latest_rate", "question": f"What is the latest {CONTAINER_NAMES[m['container']]} rate from {nm(m['origin'])} to {nm(m['dest'])}?",
                      "expected": f"{m['latest_price']:,.0f}", "expected_docs": [d["id"]]})
    for d in rng.sample(quote, n_per_kind):
        m = d["meta"]
        items.append({"kind": "cheapest_carrier", "question": f"Which carrier offers the cheapest {CONTAINER_NAMES[m['container']]} quote from {nm(m['origin'])} to {nm(m['dest'])}?",
                      "expected": m["cheapest_carrier"], "expected_docs": [d["id"]]})
    for d in rng.sample(lane, n_per_kind):
        m = d["meta"]
        pair = [x["id"] for x in lane if x["meta"]["origin"] == m["origin"] and x["meta"]["dest"] == m["dest"]]
        items.append({"kind": "transit", "question": f"How many days does it take to ship from {nm(m['origin'])} to {nm(m['dest'])}?",
                      "expected": f"{m['transit_days']} days", "expected_docs": pair})
    for d in rng.sample(lane, n_per_kind):
        m = d["meta"]
        hi = re.search(r"52-week high USD ([\d,\.]+)", d["text"]).group(1)
        items.append({"kind": "52w_high", "question": f"What was the 52-week high for a {CONTAINER_NAMES[m['container']]} from {nm(m['origin'])} to {nm(m['dest'])}?",
                      "expected": hi, "expected_docs": [d["id"]]})
    items += [
        {"kind": "port", "question": "Is Tughlakabad a seaport or an inland container depot?", "expected": "inland", "expected_docs": ["port:INTKD"]},
        {"kind": "trade", "question": "Which US coast is cheapest for importing 40' containers from China?", "expected": "West Coast", "expected_docs": ["trade:CN-US"]},
        {"kind": "trade", "question": "In which week did the India to US market peak?", "expected": re.search(r"peak [\d\.]+ in the week of ([\d-]+)",
            next(d["text"] for d in docs if d["id"] == "trade:IN-US")).group(1), "expected_docs": ["trade:IN-US"]},
        {"kind": "refusal", "question": "What is the 40' rate from Tokyo to Los Angeles?", "expected": "don't have", "expected_docs": []},
        {"kind": "refusal", "question": "Who won the last football World Cup?", "expected": "don't have", "expected_docs": []},
    ]
    return items


def answer_is_correct(answer_text: str, expected: str) -> bool:
    norm = lambda s: re.sub(r"[\s,]", "", s.lower()).replace("’", "'")
    a, e = norm(answer_text), norm(expected)
    return e in a or (e.endswith(".00") and e[:-3] in a) or (e + ".00") in a
