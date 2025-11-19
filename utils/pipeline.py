"""
takehome.pipeline

Refactored version of the TakeHome.ipynb notebook as a reusable Python package.

Main entrypoint:
    from takehome import run_all, ProjectConfig
    run_all(ProjectConfig(data_dir="DE_DS_PracticeFiles", out_dir="out"))

This will:
    - load & clean users, products, orders, order_items
    - build enriched orders with revenue
    - compute dataset profile & date stats
    - compute lifecycle event coverage & order funnel
    - build customer-level RFM features & segments
    - write all outputs to <out_dir> as CSVs and PNGs
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class ProjectConfig:
    """
    Configuration for the pipeline.

    Attributes
    ----------
    data_dir : Path
        Directory containing USERS.csv, PRODUCTS.csv, ORDERS.csv, ORDER_ITEMS.csv.
    out_dir : Path
        Directory where all derived CSVs and plots will be written.
    show_plots : bool
        Whether to display plots (useful in notebooks).
    save_plots : bool
        Whether to save plots as PNGs under out_dir.
    """
    data_dir: Path = Path("DE_DS_PracticeFiles")
    out_dir: Path = Path("out")
    show_plots: bool = True
    save_plots: bool = True

    @property
    def users_path(self) -> Path:
        return self.data_dir / "USERS.csv"

    @property
    def products_path(self) -> Path:
        return self.data_dir / "PRODUCTS.csv"

    @property
    def orders_path(self) -> Path:
        return self.data_dir / "ORDERS.csv"

    @property
    def order_items_path(self) -> Path:
        return self.data_dir / "ORDER_ITEMS.csv"


# ---------------------------------------------------------------------------
# Utility helpers (replacement for utils_refactored)
# ---------------------------------------------------------------------------

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize column names: strip, lowercase, snake_case-ish.
    """
    df = df.copy()
    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("-", "_")
    )
    return df


def trim_strings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Strip whitespace from object/string columns.
    """
    df = df.copy()
    obj_cols = df.select_dtypes(include=["object", "string"]).columns
    for c in obj_cols:
        df[c] = df[c].astype(str).str.strip()
    return df


def find_datetime_cols(df: pd.DataFrame) -> List[str]:
    """
    Guess datetime-like columns based on column names.
    """
    candidates = []
    for c in df.columns:
        lc = c.lower()
        if any(k in lc for k in ("date", "_at", "_time", "timestamp")):
            candidates.append(c)
    return candidates


def coerce_datetimes_inplace(df: pd.DataFrame, cols: Iterable[str]) -> None:
    """
    Convert given columns to timezone-aware datetimes in place.
    """
    for c in cols:
        df[c] = pd.to_datetime(df[c], errors="coerce", utc=True)


def detect_columns(df: pd.DataFrame, patterns: Sequence[str]) -> List[str]:
    """
    Return all columns whose name contains any of the given patterns (case-insensitive).
    """
    out = []
    for c in df.columns:
        lc = c.lower()
        for p in patterns:
            if p.lower() in lc:
                out.append(c)
                break
    return out


def first_match(df: pd.DataFrame, patterns: Sequence[str]) -> Optional[str]:
    """
    Return the first column whose name contains any of the given patterns.
    """
    for c in df.columns:
        lc = c.lower()
        for p in patterns:
            if p.lower() in lc:
                return c
    return None


def pick_first(candidates: Sequence[Optional[str]], default: Optional[str] = None) -> Optional[str]:
    """
    Return the first non-null, non-empty candidate from a list.
    """
    for c in candidates:
        if c:
            return c
    return default


def numericize(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def finish_fig(
    fig: plt.Figure,
    filename: str,
    config: ProjectConfig,
) -> None:
    """
    Convenience wrapper to save and/or show a Matplotlib figure.
    """
    if config.save_plots:
        config.out_dir.mkdir(parents=True, exist_ok=True)
        path = config.out_dir / filename
        fig.savefig(path, bbox_inches="tight", dpi=150)
    if config.show_plots:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Users loader & parser
# ---------------------------------------------------------------------------

def parse_users_row(s: str) -> Dict[str, object]:
    """
    Robust parser for rows like:
      {"id":93483,"":"/firstname/Brad","":"/lastname/Pitt", ...}

    Strategy:
    - Extract numeric id
    - Extract /key/value pairs into fields
    - If overall string is valid JSON, merge that dict as well
    """
    out: Dict[str, object] = {}
    text = (s or "").strip()

    # "id": <number>
    m = re.search(r'"id"\s*:\s*(\d+)', text)
    if m:
        out["id"] = int(m.group(1))

    # /key/value pairs
    for key, val in re.findall(r'/([\w\-]+)/([^,}"\]]+)', text):
        out[key.lower()] = val.strip()

    # Merge JSON dict if parseable
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k and k.strip():
                    out[k.strip().lower()] = v
    except Exception:
        pass

    return out


def load_users(config: ProjectConfig) -> pd.DataFrame:
    """
    Load and parse USERS.csv into a clean users DataFrame.
    """
    raw = pd.read_csv(
        config.users_path,
        dtype=str,
        keep_default_na=False,
        na_values=[]
    )

    # If there is exactly one column, assume it holds the JSON-ish blob
    if raw.shape[1] == 1:
        users_json_col = raw.columns[0]
    else:
        users_json_col = "user_data" if "user_data" in raw.columns else raw.columns[0]

    parsed = [parse_users_row(s) for s in raw[users_json_col]]
    users = pd.DataFrame(parsed)

    # Common renames
    rename_map = {
        "firstname": "first_name",
        "lastname": "last_name",
        "emailaddress": "email",
    }
    users.rename(columns=rename_map, inplace=True)

    users = standardize_columns(users)
    users = trim_strings(users)
    dt_cols = find_datetime_cols(users)
    coerce_datetimes_inplace(users, dt_cols)
    users.attrs["datetime_cols"] = dt_cols

    return users


# ---------------------------------------------------------------------------
# Products loader
# ---------------------------------------------------------------------------

def load_products(config: ProjectConfig) -> pd.DataFrame:
    """
    Load PRODUCTS.csv which may contain either:
    - a JSON array as raw text
    - a one-cell CSV with JSON array
    """
    path = config.products_path
    text = Path(path).read_text(encoding="utf-8").strip()

    products: Optional[pd.DataFrame]
    products = None

    # Case 1: whole file is a JSON array
    if text.startswith("["):
        try:
            products = pd.DataFrame(json.loads(text))
        except Exception:
            products = None

    # Case 2: fallback to reading as CSV with JSON in first cell
    if products is None:
        tmp = pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[])
        first_cell = tmp.iloc[0, 0]
        products = pd.DataFrame(json.loads(first_cell))

    products = standardize_columns(products)
    products = trim_strings(products)
    dt_cols = find_datetime_cols(products)
    coerce_datetimes_inplace(products, dt_cols)
    products.attrs["datetime_cols"] = dt_cols

    return products


# ---------------------------------------------------------------------------
# Orders & order_items loaders
# ---------------------------------------------------------------------------

def load_orders_and_items(config: ProjectConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load ORDERS.csv and ORDER_ITEMS.csv and perform basic cleanup.
    """
    orders = pd.read_csv(config.orders_path, low_memory=False)
    order_items = pd.read_csv(config.order_items_path, low_memory=False)

    orders = standardize_columns(orders)
    order_items = standardize_columns(order_items)

    orders = trim_strings(orders)
    order_items = trim_strings(order_items)

    coerce_datetimes_inplace(orders, find_datetime_cols(orders))
    coerce_datetimes_inplace(order_items, find_datetime_cols(order_items))

    return orders, order_items


# ---------------------------------------------------------------------------
# Orders enrichment (revenue, event columns)
# ---------------------------------------------------------------------------

def enrich_orders(
    orders: pd.DataFrame,
    order_items: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, str]]:
    """
    Build enriched orders with revenue and a consistent set of event columns.

    Returns
    -------
    orders_enriched : pd.DataFrame
        Orders with 'order_revenue', cleaned datetime columns, and standardized event columns.
    oi_enriched : pd.DataFrame
        Order items with 'line_revenue' and quantity.
    event_cols : dict
        Mapping of logical event names to actual column names:
        {created_at, shipped_at, delivered_at, returned_at, canceled_at}
    """
    orders = orders.copy()
    oi = order_items.copy()

    # --- line-level revenue ---
    order_id_col = first_match(oi, ["order_id", "order"])
    if order_id_col is None:
        raise ValueError("Could not find order_id column in ORDER_ITEMS")

    qty_col = pick_first(detect_columns(oi, ["quantity", "qty", "item_count", "items_count", "units"]))
    price_col = pick_first(detect_columns(oi, ["sale_price", "unit_price", "price", "amount", "list_price"]))
    subtotal_col = pick_first(detect_columns(oi, ["subtotal", "line_total", "extended_price", "total"]))

    if qty_col is None:
        oi["__qty"] = 1.0
        qty_col = "__qty"
    else:
        oi[qty_col] = numericize(oi[qty_col]).fillna(1.0)

    if price_col and price_col in oi.columns:
        oi["line_revenue"] = numericize(oi[price_col]) * numericize(oi[qty_col])
    elif subtotal_col and subtotal_col in oi.columns:
        oi["line_revenue"] = numericize(oi[subtotal_col]).fillna(0.0)
    else:
        oi["line_revenue"] = 0.0

    # Filter obvious returns/cancels if we have a status or returned_at
    bad = pd.Series(False, index=oi.index)
    if "status" in oi.columns:
        bad = bad | oi["status"].astype(str).str.lower().str.contains(r"return|cancel|refund|chargeback")
    if "returned_at" in oi.columns:
        bad = bad | pd.to_datetime(oi["returned_at"], errors="coerce").notna()
    oi_good = oi.loc[~bad].copy()

    # Aggregate revenue per order
    rev_per_order = oi_good.groupby(order_id_col)["line_revenue"].sum()
    orders_enriched = orders.copy()

    order_id_orders = first_match(orders_enriched, ["order_id", "order"])
    if order_id_orders is None:
        raise ValueError("Could not find order_id column in ORDERS")

    orders_enriched["order_revenue"] = orders_enriched[order_id_orders].map(rev_per_order).fillna(0.0)

    # --- standard event columns ---
    def map_event(name: str, patterns: Sequence[str]) -> Optional[str]:
        return first_match(orders_enriched, patterns)

    event_cols = {
        "created_at":   map_event("created_at",   ["created_at", "created", "order_date"]),
        "paid_at":      map_event("paid_at",      ["paid_at", "payment_date"]),
        "shipped_at":   map_event("shipped_at",   ["shipped_at", "ship_date"]),
        "delivered_at": map_event("delivered_at", ["delivered_at", "delivery_date"]),
        "returned_at":  map_event("returned_at",  ["returned_at", "return_date"]),
        "canceled_at":  map_event("canceled_at",  ["canceled_at", "cancel_date", "cancelled_at"]),
    }

    # Ensure datetime type
    for col in event_cols.values():
        if col and col in orders_enriched.columns:
            orders_enriched[col] = pd.to_datetime(orders_enriched[col], errors="coerce", utc=True)

    return orders_enriched, oi_good, event_cols


# ---------------------------------------------------------------------------
# Dataset profile & date stats
# ---------------------------------------------------------------------------

def build_dataset_profile(
    orders_enriched: pd.DataFrame,
    users: pd.DataFrame,
    order_items: pd.DataFrame,
    event_cols: Dict[str, str],
    config: ProjectConfig,
) -> pd.DataFrame:
    """
    Build a compact dataset_profile table similar to dataset_profile.csv in the notebook.
    """
    order_id = first_match(orders_enriched, ["order_id", "order"])
    user_id = first_match(orders_enriched, ["user_id", "customer_id", "uid"])

    n_users = users[first_match(users, ["id", "user_id", "customer_id"])].nunique() if not users.empty else 0
    n_orders = orders_enriched[order_id].nunique() if order_id else len(orders_enriched)
    # Products count is typically from products, but we can approximate from order_items
    prod_id_col = first_match(order_items, ["product_id", "sku", "asin", "item_id"])
    n_items = order_items[prod_id_col].nunique() if prod_id_col else 0
    n_lines = len(order_items)

    total_revenue = float(orders_enriched["order_revenue"].sum())
    aov = total_revenue / max(n_orders, 1)

    # Orders per user stats
    if user_id and order_id:
        orders_per_user = (
            orders_enriched.groupby(user_id)[order_id]
            .nunique()
            .rename("orders_per_user")
        )
    else:
        orders_per_user = pd.Series(dtype="float64")

    if not orders_per_user.empty:
        mean_opu = float(orders_per_user.mean())
        median_opu = float(orders_per_user.median())
        p90_opu = float(orders_per_user.quantile(0.9))
    else:
        mean_opu = median_opu = p90_opu = 0.0

    # Return rate
    if "returned_at" in orders_enriched.columns:
        returned_orders = orders_enriched["returned_at"].notna().sum()
        return_rate = float(returned_orders / max(n_orders, 1))
    else:
        return_rate = np.nan

    # Time window based on primary clock
    primary_col = event_cols.get("created_at") or event_cols.get("delivered_at") or event_cols.get("shipped_at")
    primary_key = None
    if primary_col:
        primary_key = primary_col
        data_start_utc = orders_enriched[primary_col].min()
        data_end_utc = orders_enriched[primary_col].max()
        if pd.notna(data_start_utc) and pd.notna(data_end_utc):
            window_days = (data_end_utc - data_start_utc).days + 1
            months_covered = orders_enriched[primary_col].dt.to_period("M").nunique()
        else:
            data_start_utc = data_end_utc = pd.NaT
            window_days = months_covered = np.nan
    else:
        primary_col = primary_key = None
        data_start_utc = data_end_utc = pd.NaT
        window_days = months_covered = np.nan

    window_years = window_days / 365.25 if window_days == window_days else np.nan
    window_months = window_days / 30.4375 if window_days == window_days else np.nan

    profile_rows = [
        ("unique_users", int(n_users)),
        ("unique_products", int(n_items)),
        ("unique_orders", int(n_orders)),
        ("line_items", int(n_lines)),
        ("total_revenue", round(total_revenue, 2)),
        ("aov", round(aov, 2)),
        ("orders_per_user_mean", round(mean_opu, 3)),
        ("orders_per_user_median", round(median_opu, 3)),
        ("orders_per_user_p90", round(p90_opu, 3)),
        ("return_rate_orders", round(return_rate, 4) if pd.notna(return_rate) else np.nan),
        ("window_start_utc", data_start_utc),
        ("window_end_utc", data_end_utc),
        ("months_covered", int(months_covered) if months_covered == months_covered else np.nan),
    ]
    profile = pd.DataFrame(profile_rows, columns=["metric", "value"])

    config.out_dir.mkdir(parents=True, exist_ok=True)
    profile.to_csv(config.out_dir / "dataset_profile.csv", index=False)

    return profile


def build_date_stats(
    orders: pd.DataFrame,
    event_cols: Dict[str, str],
    config: ProjectConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build:
      - date_event_summary.csv
      - dataset_date_stats.csv
    """
    orders_dt = orders.copy()

    # Ensure datetime
    for c in event_cols.values():
        if c and c in orders_dt.columns:
            orders_dt[c] = pd.to_datetime(orders_dt[c], errors="coerce", utc=True)

    # Primary clock
    primary_col = event_cols.get("created_at") or event_cols.get("delivered_at") or event_cols.get("shipped_at")
    primary_key = None
    if primary_col:
        primary_key = primary_col
        s = orders_dt[primary_col].dropna()
        data_start_utc = s.min() if not s.empty else pd.NaT
        data_end_utc = s.max() if not s.empty else pd.NaT
        if pd.notna(data_start_utc) and pd.notna(data_end_utc):
            window_days = (data_end_utc - data_start_utc).days + 1
            months_covered = orders_dt[primary_col].dt.to_period("M").nunique()
        else:
            data_start_utc = data_end_utc = pd.NaT
            window_days = months_covered = np.nan
    else:
        primary_col = primary_key = None
        data_start_utc = data_end_utc = pd.NaT
        window_days = months_covered = np.nan

    window_years = window_days / 365.25 if window_days == window_days else np.nan
    window_months = window_days / 30.4375 if window_days == window_days else np.nan

    # Event summary
    event_summary = []
    for name, col in event_cols.items():
        if col and col in orders_dt.columns:
            s = orders_dt[col].dropna()
            event_summary.append(
                {
                    "event": name,
                    "column": col,
                    "coverage_%": round(orders_dt[col].notna().mean() * 100, 2),
                    "min_utc": s.min() if not s.empty else pd.NaT,
                    "max_utc": s.max() if not s.empty else pd.NaT,
                    "span_days": (s.max() - s.min()).days + 1 if len(s) else np.nan,
                }
            )
    event_summary_df = pd.DataFrame(event_summary).sort_values("coverage_%", ascending=False)

    date_stats = pd.DataFrame(
        [
            {"metric": "primary_date_key", "value": primary_key},
            {"metric": "primary_date_column", "value": primary_col},
            {"metric": "data_start_utc", "value": data_start_utc},
            {"metric": "data_end_utc", "value": data_end_utc},
            {"metric": "window_days", "value": window_days},
            {"metric": "window_months", "value": window_months},
            {"metric": "window_years", "value": window_years},
            {
                "metric": "distinct_months_with_orders",
                "value": int(months_covered) if months_covered == months_covered else np.nan,
            },
        ]
    )

    config.out_dir.mkdir(parents=True, exist_ok=True)
    event_summary_df.to_csv(config.out_dir / "date_event_summary.csv", index=False)
    date_stats.to_csv(config.out_dir / "dataset_date_stats.csv", index=False)

    # Simple monthly orders plot
    if primary_col:
        order_id = first_match(orders_dt, ["order_id", "order"])
        monthly = (
            orders_dt.set_index(primary_col)[order_id]
            .resample("MS")
            .count()
            .rename("orders_per_month")
            .reset_index()
        )
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(monthly[primary_col], monthly["orders_per_month"], width=25, align="center")
        ax.set_title(f"Orders per Month (primary={primary_col})")
        ax.set_xlabel("Month")
        ax.set_ylabel("Orders")
        fig.autofmt_xdate()
        finish_fig(fig, "orders_per_month.png", config)

    return event_summary_df, date_stats


# ---------------------------------------------------------------------------
# Order funnel
# ---------------------------------------------------------------------------

def build_order_funnel(
    orders: pd.DataFrame,
    event_cols: Dict[str, str],
    config: ProjectConfig,
) -> pd.DataFrame:
    """
    Compute a simple created → shipped → delivered funnel plus returned/canceled.
    """
    created_col = event_cols.get("created_at")
    shipped_col = event_cols.get("shipped_at")
    delivered_col = event_cols.get("delivered_at")
    returned_col = event_cols.get("returned_at")
    canceled_col = event_cols.get("canceled_at")

    N = len(orders)
    n_created = int(orders[created_col].notna().sum()) if created_col else 0
    n_shipped = int(orders[shipped_col].notna().sum()) if shipped_col else 0
    n_delivered = int(orders[delivered_col].notna().sum()) if delivered_col else 0
    n_returned = int(orders[returned_col].notna().sum()) if returned_col else 0
    n_canceled = int(orders[canceled_col].notna().sum()) if canceled_col else 0

    funnel = pd.DataFrame(
        {
            "stage": ["created", "shipped", "delivered", "returned", "canceled"],
            "count": [n_created, n_shipped, n_delivered, n_returned, n_canceled],
        }
    )
    funnel["pct_of_all_orders"] = (funnel["count"] / max(N, 1) * 100).round(2)
    funnel["seq_conversion_%"] = [
        np.nan,
        round((n_shipped / n_created * 100), 2) if n_created else np.nan,
        round((n_delivered / n_shipped * 100), 2) if n_shipped else np.nan,
        round((n_returned / n_delivered * 100), 2) if n_delivered else np.nan,
        round((n_canceled / n_created * 100), 2) if n_created else np.nan,
    ]

    funnel.to_csv(config.out_dir / "order_funnel.csv", index=False)

    # Simple bar plot (share of all orders)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(funnel["stage"], funnel["pct_of_all_orders"])
    ax.set_xlabel("Stage")
    ax.set_ylabel("Share of all orders (%)")
    ax.set_title("Order Funnel")
    for i, v in enumerate(funnel["pct_of_all_orders"]):
        ax.text(i, v + 1, f"{v:.1f}%", ha="center", va="bottom", fontsize=8)
    finish_fig(fig, "order_funnel_bar.png", config)

    return funnel


# ---------------------------------------------------------------------------
# RFM segmentation & customer features
# ---------------------------------------------------------------------------

def build_rfm(
    orders: pd.DataFrame,
    user_id_col: str,
    order_date_col: str,
    revenue_col: str = "order_revenue",
) -> pd.DataFrame:
    """
    Build standard RFM features from orders.
    """
    df = orders.copy()
    df[order_date_col] = pd.to_datetime(df[order_date_col], errors="coerce", utc=True)

    snapshot_date = df[order_date_col].max() + pd.Timedelta(days=1)

    grouped = df.groupby(user_id_col).agg(
        recency_last=(order_date_col, "max"),
        frequency=(order_date_col, "nunique"),
        monetary=(revenue_col, "sum"),
    )
    grouped = grouped[grouped["frequency"] > 0]

    grouped["recency_days"] = (snapshot_date - grouped["recency_last"]).dt.days
    grouped["monetary"] = grouped["monetary"] / grouped["frequency"].replace(0, np.nan)

    rfm = grouped[["recency_days", "frequency", "monetary"]].copy()

    # Quintiles for R, F, M
    q_r = rfm["recency_days"].quantile([0.2, 0.4, 0.6, 0.8])
    q_f = rfm["frequency"].quantile([0.2, 0.4, 0.6, 0.8])
    q_m = rfm["monetary"].quantile([0.2, 0.4, 0.6, 0.8])

    def r_score(x):
        if x <= q_r.iloc[0]:
            return 5
        elif x <= q_r.iloc[1]:
            return 4
        elif x <= q_r.iloc[2]:
            return 3
        elif x <= q_r.iloc[3]:
            return 2
        else:
            return 1

    def fm_score(x, q):
        if x <= q.iloc[0]:
            return 1
        elif x <= q.iloc[1]:
            return 2
        elif x <= q.iloc[2]:
            return 3
        elif x <= q.iloc[3]:
            return 4
        else:
            return 5

    rfm["r"] = rfm["recency_days"].apply(r_score)
    rfm["f"] = rfm["frequency"].apply(lambda x: fm_score(x, q_f))
    rfm["m"] = rfm["monetary"].apply(lambda x: fm_score(x, q_m))

    # Segment assignment
    def segment_row(row):
        r, f, m = row["r"], row["f"], row["m"]
        if r >= 4 and f >= 4 and m >= 4:
            return "VIP"
        if r >= 4 and (f >= 4 or m >= 4):
            return "Loyal"
        if r >= 3 and f >= 3:
            return "Potential Loyalist"
        if r <= 2 and (f >= 3 or m >= 3):
            return "At Risk"
        if r <= 2 and f <= 2 and m <= 2:
            return "Dormant"
        return "Regular"

    rfm["segment"] = rfm.apply(segment_row, axis=1)

    rfm = rfm.reset_index().rename(columns={user_id_col: "user_id"})
    return rfm


def build_customer_features(
    orders: pd.DataFrame,
    users: pd.DataFrame,
    rfm: pd.DataFrame,
    config: ProjectConfig,
) -> pd.DataFrame:
    """
    Join RFM segments with simple demographics features (gender, age, country, email_domain).
    """
    user_id_orders = first_match(orders, ["user_id", "customer_id", "uid"])
    user_id_users = first_match(users, ["id", "user_id", "customer_id"])

    if not user_id_orders or not user_id_users:
        raise ValueError("Could not align user id columns between orders and users")

    # Basic age / email domain from users
    users_demo = users.copy()

    # Try to build age_years (DOB or explicit age)
    dob = pick_first(detect_columns(users_demo, ["dob", "birth", "birthday", "date_of_birth"]))
    if dob:
        users_demo[dob] = pd.to_datetime(users_demo[dob], errors="coerce")
        users_demo["age_years"] = (pd.Timestamp.today(tz="UTC") - users_demo[dob]).dt.days / 365.25
    if "age_years" not in users_demo.columns or users_demo["age_years"].isna().all():
        age_col = pick_first(detect_columns(users_demo, ["age_years", "age"]))
        if age_col:
            users_demo["age_years"] = numericize(users_demo[age_col])

    if "email" in users_demo.columns:
        users_demo["email_domain"] = (
            users_demo["email"].astype(str).str.lower().str.extract(r"@(.+)$")[0]
        )

    gender_col = pick_first(detect_columns(users_demo, ["gender", "sex"]))
    country_col = pick_first(detect_columns(users_demo, ["country", "country_code", "nation", "geo_country"]))

    user_cols_keep = [c for c in [user_id_users, gender_col, country_col, "age_years", "email_domain"] if c]
    demo = users_demo[user_cols_keep].drop_duplicates()

    # Aggregate order-level stats per user (lifetime revenue, last order date, etc.)
    orders_copy = orders.copy()
    order_date_col = pick_first(detect_columns(orders_copy, ["created_at", "order_date", "date"]))
    if not order_date_col:
        raise ValueError("Could not find an order date column")

    orders_copy[order_date_col] = pd.to_datetime(orders_copy[order_date_col], errors="coerce", utc=True)

    agg = (
        orders_copy.groupby(user_id_orders)
        .agg(
            total_revenue=("order_revenue", "sum"),
            num_orders=("order_revenue", "count"),
            first_order_date=(order_date_col, "min"),
            last_order_date=(order_date_col, "max"),
        )
        .reset_index()
        .rename(columns={user_id_orders: "user_id"})
    )
    agg["avg_order_value"] = agg["total_revenue"] / agg["num_orders"].replace(0, np.nan)
    agg["lifetime_days"] = (agg["last_order_date"] - agg["first_order_date"]).dt.days

    # Join RFM + agg + demo
    feat = (
        rfm.merge(agg, on="user_id", how="left")
        .merge(demo.rename(columns={user_id_users: "user_id"}), on="user_id", how="left")
    )

    feat["value_group"] = np.where(feat["segment"].isin(["VIP", "Loyal"]), "valuable", "other")
    feat["risk_group"] = np.where(feat["segment"].eq("At Risk"), "at_risk", "not_at_risk")

    feat.to_csv(config.out_dir / "customer_features_segmented.csv", index=False)
    return feat


# ---------------------------------------------------------------------------
# Main entrypoint
# ---------------------------------------------------------------------------

def run_all(config: ProjectConfig) -> None:
    """
    Run the full pipeline with the given configuration.
    """
    config.out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Load data
    users = load_users(config)
    products = load_products(config)
    orders, order_items = load_orders_and_items(config)

    # 2) Enrich orders with revenue & event columns
    orders_enriched, oi_enriched, event_cols = enrich_orders(orders, order_items)

    # 3) Dataset profile & date stats
    _ = build_dataset_profile(orders_enriched, users, oi_enriched, event_cols, config)
    _, _ = build_date_stats(orders_enriched, event_cols, config)

    # 4) Order funnel
    _ = build_order_funnel(orders_enriched, event_cols, config)

    # 5) RFM + customer features
    user_id_orders = first_match(orders_enriched, ["user_id", "customer_id", "uid"])
    order_date_col = event_cols.get("created_at") or first_match(orders_enriched, ["created_at", "order_date", "date"])
    if not user_id_orders or not order_date_col:
        print("Skipping RFM: missing user_id or order date column")
        return

    rfm = build_rfm(orders_enriched.rename(columns={user_id_orders: "user_id"}), "user_id", order_date_col)
    feat = build_customer_features(orders_enriched, users, rfm, config)

    # 6) Simple segment plots (counts & revenue share)
    seg_counts = feat["segment"].value_counts().rename_axis("segment").reset_index(name="customers")
    fig = plt.figure(figsize=(8, 4))
    plt.bar(seg_counts["segment"], seg_counts["customers"])
    plt.xticks(rotation=45, ha="right")
    plt.xlabel("Segment")
    plt.ylabel("Customers")
    plt.title("Customers by Segment")
    finish_fig(fig, "segment_counts.png", config)

    seg_rev = feat.groupby("segment", as_index=False)["total_revenue"].sum().sort_values("total_revenue", ascending=False)
    fig = plt.figure(figsize=(8, 4))
    plt.bar(seg_rev["segment"], seg_rev["total_revenue"])
    plt.xticks(rotation=45, ha="right")
    plt.xlabel("Segment")
    plt.ylabel("Revenue")
    plt.title("Revenue by Segment")
    finish_fig(fig, "segment_revenue.png", config)

    print(f"Pipeline completed. Outputs written to: {config.out_dir}")


if __name__ == "__main__":
    # Basic CLI entrypoint
    cfg = ProjectConfig()
    run_all(cfg)
