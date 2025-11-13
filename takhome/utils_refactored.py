# ==== utils_refactored.py (drop-in) ====
from __future__ import annotations
import os
import re
from pathlib import Path
from typing import Iterable, List, Sequence, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# Name normalization & picking
# -----------------------------
def _norm_name(s: str) -> str:
    """Lowercase, remove non-alphanumerics, collapse to a-z0-9."""
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def detect_columns(df: pd.DataFrame, patterns: Sequence[str]) -> List[str]:
    """
    Case/spacing/punctuation-insensitive substring match over column names.
    Returns a de-duplicated list of ORIGINAL column names preserving order.
    """
    norm_map = {_norm_name(c): c for c in df.columns}
    out: List[str] = []
    for p in patterns:
        npat = _norm_name(p)
        for ncol, orig in norm_map.items():
            if npat in ncol:
                out.append(orig)
    # de-dup while preserving order
    seen = set()
    uniq = []
    for c in out:
        if c not in seen:
            uniq.append(c)
            seen.add(c)
    return uniq


def first_match(df: pd.DataFrame, pats: Sequence[str]) -> Optional[str]:
    """Return the first detected column or None."""
    cols = detect_columns(df, pats)
    return cols[0] if cols else None


def pick_first(lst: Sequence, default=None):
    """Safe first() with default."""
    return lst[0] if (lst and len(lst) > 0) else default


# -----------------------------
# DataFrame cleaning helpers
# -----------------------------
def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a copy with standardized snake_case column names:
    - strip, lower
    - non-word -> underscore
    - trim leading/trailing underscores
    """
    def _clean(c: str) -> str:
        c = str(c).strip().lower()
        c = re.sub(r"[^\w]+", "_", c)
        c = re.sub(r"(^_+|_+$)", "", c)
        return c
    out = df.copy()
    out.columns = [_clean(c) for c in out.columns]
    return out


def trim_strings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Trim whitespace in object columns and coerce common empties to NaN.
    Converts '', 'nan', 'none', 'null' (any case) to NaN.
    """
    out = df.copy()
    obj_cols = out.select_dtypes(include=["object"]).columns
    if len(obj_cols) == 0:
        return out
    empties = re.compile(r"^(?:nan|none|null)?$", flags=re.IGNORECASE)
    for c in obj_cols:
        s = out[c].astype(str).str.strip()
        out[c] = s.mask(s.map(lambda x: bool(empties.match(x))))
    return out


def find_datetime_cols(df: pd.DataFrame) -> List[str]:
    """
    Heuristic: include columns that are already datetime64
    or whose names contain common time tokens.
    """
    tokens = ("date", "time", "timestamp", "created", "updated", "placed", "shipped", "delivered", "returned")
    cols = []
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            cols.append(c)
            continue
        lc = str(c).lower()
        if any(tok in lc for tok in tokens):
            cols.append(c)
    return cols


def coerce_datetimes_inplace(df: pd.DataFrame, dt_cols: Iterable[str]) -> None:
    """
    Parse columns to datetime (UTC-aware), then make them timezone-naive (UTC).
    Safe even if parsing fails (column left unchanged).
    """
    for c in dt_cols:
        try:
            s = pd.to_datetime(df[c], errors="coerce", utc=True)
            # Convert to explicit UTC then drop tz info (naive in UTC)
            df[c] = s.dt.tz_convert("UTC").dt.tz_localize(None)
        except Exception:
            # leave as-is if incompatible
            pass


def numericize(df: pd.DataFrame, cols: Iterable[str]) -> None:
    """Inplace: coerce listed columns to numeric with NaN on errors."""
    for c in cols:
        if c and c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")


# -----------------------------
# Plot helper
# -----------------------------
def finish_fig(
    fig: plt.Figure,
    filename: Optional[str] = None,
    *,
    out_dir: Optional[str | Path] = None,
    show: Optional[bool] = None,
    save: Optional[bool] = None,
    dpi: int = 150
) -> None:
    """
    Save &/or show a Matplotlib figure, then close it.
    - If `save`/`show` are None, fall back to globals SAVE_PLOTS/SHOW_PLOTS.
    - If saving and `out_dir` is None, fall back to global OUT_DIR or current dir.
    """
    if show is None:
        show = bool(globals().get("SHOW_PLOTS", True))
    if save is None:
        save = bool(globals().get("SAVE_PLOTS", False))

    if save and filename:
        base = out_dir if out_dir is not None else globals().get("OUT_DIR", ".")
        path = Path(base) / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, bbox_inches="tight", dpi=dpi)

    if show:
        plt.show()

    plt.close(fig)
