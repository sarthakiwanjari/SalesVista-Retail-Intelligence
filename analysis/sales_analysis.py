"""Reusable, defensive analytics for the SalesVista Superstore dashboard."""
from pathlib import Path
import re
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "superstore.csv"
ALIASES = {
    "order_id": ["order id", "order_id", "orderid"], "order_date": ["order date", "order_date", "orderdate"],
    "ship_date": ["ship date", "ship_date"], "customer_id": ["customer id", "customer_id"],
    "customer_name": ["customer name", "customer_name"], "segment": ["segment"], "country": ["country"],
    "city": ["city"], "state": ["state"], "region": ["region"], "product_id": ["product id", "product_id"],
    "category": ["category"], "sub_category": ["sub-category", "sub category", "sub_category"],
    "product_name": ["product name", "product_name"], "sales": ["sales", "revenue"],
    "quantity": ["quantity", "qty"], "discount": ["discount"], "profit": ["profit"],
    "market": ["market"], "ship_mode": ["ship mode", "ship_mode"], "order_priority": ["order priority", "order_priority"],
}
NUMERIC = ["sales", "profit", "quantity", "discount", "shipping_cost"]

def normalize_name(name):
    return re.sub(r"[^a-z0-9]+", " ", str(name).strip().lower()).strip()

def load_data(path=DATA_PATH):
    """Load CSV and map common column-name variants to stable internal names."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}. Download the real Kaggle Superstore CSV and save it as data/superstore.csv.")
    last_error = None
    df = None
    for encoding in ("utf-8-sig", "latin-1", "cp1252"):
        try:
            df = pd.read_csv(path, encoding=encoding, low_memory=False)
            break
        except Exception as exc:
            last_error = exc
    if df is None:
        raise ValueError(f"Could not read CSV: {last_error}")
    lookup = {normalize_name(c): c for c in df.columns}
    rename = {}
    for target, options in ALIASES.items():
        for option in options:
            if normalize_name(option) in lookup:
                rename[lookup[normalize_name(option)]] = target
                break
    df = df.rename(columns=rename).copy()
    for col in NUMERIC:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(r"[$,]", "", regex=True), errors="coerce")
    for col in ("order_date", "ship_date"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    if "sales" not in df.columns:
        raise ValueError("The CSV must contain a Sales column. Please check that you downloaded the Superstore transaction dataset.")
    for col in ("profit", "quantity", "discount"):
        if col not in df.columns:
            df[col] = np.nan
    for col in ("region", "category", "segment", "product_name", "sub_category", "order_id", "customer_name"):
        if col in df.columns:
            df[col] = df[col].fillna("Unknown").astype(str)
    return df

def _num(s):
    return pd.to_numeric(s, errors="coerce").fillna(0)

def filter_data(df, params):
    out = df.copy()
    if "order_date" in out.columns:
        start, end = params.get("start"), params.get("end")
        if start:
            out = out[out.order_date >= pd.to_datetime(start, errors="coerce")]
        if end:
            out = out[out.order_date < (pd.to_datetime(end, errors="coerce") + pd.Timedelta(days=1))]
    for key in ("region", "category", "segment"):
        value = params.get(key)
        if value and value != "All" and key in out.columns:
            out = out[out[key].astype(str) == str(value)]
    return out

def _records(df):
    clean = df.copy()
    for c in clean.columns:
        if pd.api.types.is_datetime64_any_dtype(clean[c]): clean[c] = clean[c].dt.strftime("%Y-%m-%d").fillna("")
    clean = clean.replace({np.nan: None, np.inf: None, -np.inf: None})
    return clean.to_dict(orient="records")

def _group(df, field, metrics):
    if field not in df.columns or df.empty: return []
    use = [m for m in metrics if m in df.columns]
    if not use: return []
    g = df.groupby(field, dropna=False)[use].sum(numeric_only=True).reset_index()
    return _records(g.sort_values(use[0], ascending=False))

def calculate_kpis(df):
    sales = _num(df.get("sales", pd.Series(dtype=float))).sum()
    profit = _num(df.get("profit", pd.Series(dtype=float))).sum()
    quantity = _num(df.get("quantity", pd.Series(dtype=float))).sum()
    orders = int(df["order_id"].nunique()) if "order_id" in df.columns else int(len(df))
    return {"sales": float(sales), "profit": float(profit), "orders": orders, "quantity": float(quantity),
            "aov": float(sales / orders) if orders else 0, "margin": float(profit / sales * 100) if sales else 0,
            "discount": float(_num(df["discount"]).mean()) if "discount" in df.columns and len(df) else 0,
            "rows": int(len(df))}

def monthly_performance(df):
    if "order_date" not in df.columns or df.empty: return []
    d = df[df.order_date.notna()].copy()
    if d.empty: return []
    d["month"] = d.order_date.dt.to_period("M").astype(str)
    cols = [c for c in ("sales", "profit") if c in d.columns]
    return _records(d.groupby("month", as_index=False)[cols].sum().sort_values("month"))

def category_performance(df): return _group(df, "category", ["sales", "profit", "quantity"])
def regional_performance(df):
    rows = _group(df, "region", ["sales", "profit", "quantity"])
    for row in rows:
        key = row.get("region")
        part = df[df.region.astype(str) == str(key)] if "region" in df.columns else df.iloc[0:0]
        row["orders"] = int(part.order_id.nunique()) if "order_id" in part.columns else len(part)
        row["margin"] = float(row.get("profit", 0) / row.get("sales", 0) * 100) if row.get("sales", 0) else 0
    return rows

def product_performance(df, limit=None):
    if "product_name" not in df.columns or df.empty: return []
    group = ["product_name"] + [c for c in ("category", "sub_category") if c in df.columns]
    agg = df.groupby(group, dropna=False).agg(sales=("sales", "sum"), profit=("profit", "sum"), quantity=("quantity", "sum")).reset_index()
    agg["margin"] = np.where(agg.sales != 0, agg.profit / agg.sales * 100, 0)
    agg = agg.sort_values("sales", ascending=False)
    if limit: agg = agg.head(limit)
    return _records(agg)

def generate_insights(df):
    insights=[]
    if df.empty: return ["No records match these filters. Try widening the date range or resetting filters."]
    if "category" in df.columns:
        g=df.groupby("category").sales.sum().sort_values(ascending=False)
        if len(g): insights.append(f"{g.index[0]} is the highest-sales category at ₹{g.iloc[0]:,.0f}.")
    if "region" in df.columns and "profit" in df.columns:
        g=df.groupby("region").profit.sum().sort_values(ascending=False)
        if len(g): insights.append(f"{g.index[0]} leads regional profit with ₹{g.iloc[0]:,.0f}.")
    if "product_name" in df.columns:
        g=df.groupby("product_name").sales.sum().sort_values(ascending=False)
        if len(g): insights.append(f"{g.index[0]} has the highest sales among products at ₹{g.iloc[0]:,.0f}.")
    if "order_date" in df.columns:
        d=df[df.order_date.notna()].copy()
        if len(d):
            d["month"]=d.order_date.dt.to_period("M").astype(str); g=d.groupby("month").sales.sum()
            if len(g): insights.append(f"{g.idxmax()} is the strongest month by sales (₹{g.max():,.0f}).")
    k=calculate_kpis(df); insights.append(f"Overall profit margin for this selection is {k['margin']:.1f}%.")
    if "category" in df.columns and "profit" in df.columns:
        loss=df.groupby("category").profit.sum(); loss=loss[loss<0]
        if len(loss): insights.append("Loss-making categories in this selection: " + ", ".join(f"{n} (₹{v:,.0f})" for n,v in loss.items()) + ".")
    if "discount" in df.columns and "profit" in df.columns and df.discount.nunique()>1:
        corr=pd.DataFrame({"discount":_num(df.discount),"profit":_num(df.profit)}).corr().iloc[0,1]
        if pd.notna(corr):
            description="a negative" if corr < -0.15 else "a positive" if corr > 0.15 else "a weak or near-zero"
            insights.append(f"Discount and profit show {description} correlation ({corr:.2f}); this is an association, not proof of causation.")
    return insights

def dashboard_payload(df, params=None):
    filtered=filter_data(df, params or {})
    fields=list(filtered.columns)
    def options(col): return sorted(filtered[col].dropna().astype(str).unique().tolist()) if col in filtered.columns else []
    return {"kpis":calculate_kpis(filtered), "monthly":monthly_performance(filtered), "categories":category_performance(filtered),
      "regions":regional_performance(filtered), "products":product_performance(filtered, 10), "segments":_group(filtered,"segment",["sales","profit"]) if "segment" in filtered.columns else [],
      "insights":generate_insights(filtered), "records":_records(filtered), "columns":fields,
      "filters":{"regions":options("region"),"categories":options("category"),"segments":options("segment"),
      "min_date":str(filtered.order_date.min().date()) if "order_date" in filtered.columns and filtered.order_date.notna().any() else "",
      "max_date":str(filtered.order_date.max().date()) if "order_date" in filtered.columns and filtered.order_date.notna().any() else ""}}
