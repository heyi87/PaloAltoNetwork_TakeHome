# Loyalty & Growth — Data → Segments → Predictive CLV

End-to-end pipeline to:

1) **Clean** messy commerce data (USERS / PRODUCTS / ORDERS / ORDER_ITEMS)  
2) **Explore** coverage, timeline, geo/age/category mix  
3) **Segment** customers with **RFM + AOV + Tenure** (+ personalized churn window)  
4) **Predict** 90-day CLV with **BG/NBD + Gamma-Gamma**, **Tweedie GLM**, and a **Blend**  
5) **Publish** a presentation (PPTX) using your actual charts

Works out-of-the-box on typical “take-home” CSVs; robust to missing columns, mixed schemas, and partial timestamps.

---

## Project structure

```
.
├─ DE_DS_PracticeFiles/           # INPUT: USERS.csv, PRODUCTS.csv, ORDERS.csv, ORDER_ITEMS.csv
├─ notebooks/
│  └─ 01_takehome_end_to_end.ipynb  # (optional) single-notebook version
├─ src/
│  ├─ cleaning.py                 # standardize/trim; date detection & UTC coercion; messy USERS parsing
│  ├─ orders_completed.py         # build orders_completed with purchase timestamp + net revenue
│  ├─ exploration.py              # coverage tables, timeline, funnel, latency plots
│  ├─ segmentation.py             # RFM + cadence; segments & risk overlays; plots
│  ├─ clv.py                      # BG/NBD + GG baseline; Tweedie ML; blend; metrics & calibration
│  ├─ slides.py                   # PPTX deck from the exported PNGs & metrics
│  └─ utils.py                    # tolerant column detection, helpers
├─ out/                           # OUTPUT: all CSVs, PNGs, and PPTX land here
├─ requirements.txt
└─ README.md
```

> Prefer a notebook? Use `notebooks/01_takehome_end_to_end.ipynb`. Otherwise run the CLI (below).

---

## Setup

- Python **3.9+** recommended

```bash
# venv
python -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt

# or conda
conda create -n loyalty python=3.10 -y
conda activate loyalty
pip install -r requirements.txt
```

**requirements.txt**
```
pandas
numpy
matplotlib
lifetimes
scikit-learn
python-pptx
```

---

## Data inputs

Place the four CSVs here:

```
DE_DS_PracticeFiles/
  ├─ USERS.csv
  ├─ PRODUCTS.csv
  ├─ ORDERS.csv
  └─ ORDER_ITEMS.csv
```

Schema is **tolerant** (case/spacing/underscore-insensitive). Examples it detects:

- Customer id: `user_id`, `id`, `customer_id`, `buyer_id`, `account_id`
- Product id: `product_id`, `sku`, `asin`
- Timestamps: `created_at`, `paid_at`, `shipped_at`, `delivered_at`, `returned_at`, `canceled_at` (aliases supported)
- Revenue: if **order_revenue** is missing, it’s rebuilt from items (price×qty or subtotal)

---

## Quick start (one command)

```bash
python -m src.clv   --data-dir DE_DS_PracticeFiles   --out-dir out   --horizon 90
```

This runs the full pipeline:

- Cleans & normalizes data (UTC timestamps; numeric revenue)  
- Builds `orders_completed` (purchase timestamp priority: **delivered → shipped → created → paid**)  
- Saves exploration charts/CSVs  
- Computes **RFM + risk** → `customer_features_segmented.csv`  
- Trains **BG/NBD + Gamma-Gamma** baseline and **Tweedie** ML model, plus a **Blend**  
- Writes:
  - `out/predicted_clv_90d.csv`
  - `out/clv_metrics_table.csv`, `out/clv_deciles.csv`, `out/clv_gains_curve.csv`
  - `out/clv_decile_calibration.png`, `out/clv_gains_curve.png`, `out/clv_scatter_log1p.png`, `out/clv_residual_hist.png`
- Builds a PPTX deck: **`out/Loyalty_Growth_Deck.pptx`**

---

## Run step-by-step

### 1) Build `orders_completed`
```bash
python -m src.orders_completed --data-dir DE_DS_PracticeFiles --out-dir out
```
- Purchase timestamp priority: delivered → shipped → created → paid  
- Filters cancels/returns where detectable  
- Ensures **order_revenue** exists (reconstructed if needed)  
- **Output:** `out/orders_completed.csv`

### 2) Data exploration
```bash
python -m src.exploration --data-dir DE_DS_PracticeFiles --out-dir out
```
**Outputs**
- `date_event_summary.csv`, `dataset_date_stats.csv` (coverage, min/max, span)  
- `orders_per_month.png`, `order_timeline_coverage_full.png`  
- `order_funnel.csv`, `order_funnel_pct_of_all.png`  
- Latency histograms: created→shipped, shipped→delivered, delivered→returned

### 3) Customer segmentation (RFM + cadence + risk)
```bash
python -m src.segmentation --data-dir DE_DS_PracticeFiles --out-dir out
```
**Per-customer features**
- `recency_days`, `frequency` (orders), `monetary` (sum revenue), `aov` (= M/F), `tenure_days`  
- `interpurchase_median_days` (median time between orders)  
- **Personalized churn window:** `clip(2 × interpurchase_median_days, 30..180)`

**Quintile scoring & labels**
- **R (recency_days, lower=better)**; **F (frequency, higher=better)**; **M (monetary, higher=better)** → scores 1..5  
- **Segment rules (in order):**  
  - **VIP:** r≥4 ∧ f≥4 ∧ m≥4  
  - **Loyal:** r≥4 ∧ (f≥4 ∨ m≥4)  
  - **Potential Loyalist:** r≥3 ∧ f≥3  
  - **At Risk:** r≤2 ∧ (f≥3 ∨ m≥3)  
  - **Dormant:** r≤2 ∧ f≤2 ∧ m≤2  
  - **Regulars:** otherwise

**Risk overlays (independent)**
- `is_churned = recency_days > churn_window_days`  
- `is_at_risk = (~is_churned) & (recency_days > 0.75 × churn_window_days)`

**Outputs**
- `customer_features_segmented.csv`  
- Plots: `seg_counts.png`, `seg_revenue.png`, RF monetary heatmap, risk funnel, pareto

### 4) Predictive CLV (90-day horizon)
```bash
python -m src.clv --data-dir DE_DS_PracticeFiles --out-dir out --horizon 90
```
**Modeling**
- **Baseline**: BG/NBD → expected purchases; Gamma-Gamma → expected AOV  
- **ML**: Tweedie GLM on RFM + engagement + demographics (NaN-safe imputers)  
- **Blend**: 0.5 × (BG/NBD) + 0.5 × (ML)

**Evaluation**
- **MAE / RMSE** (primary), **Median APE** on **non-zero** rows only  
- **Decile calibration** (actual vs predicted per customer, by score decile)  
- **Gains curve** (cumulative revenue captured as you target more customers)

**Outputs**
- `predicted_clv_90d.csv` (id, exp_purch_T, exp_aov, p_alive, clv_T_bgnbd, clv_T_ml, clv_T_blend, actual_clv_T)  
- `clv_metrics_table.csv`, `clv_deciles.csv`, `clv_gains_curve.csv`  
- Plots: `clv_decile_calibration.png`, `clv_gains_curve.png`, `clv_scatter_log1p.png`, `clv_residual_hist.png`

> **Why not MAPE?** 90-day CLV is **zero-inflated** (many customers spend $0). MAPE explodes on small actuals; use **MAE/RMSE + gains/deciles** instead.

### 5) Slides (PPTX)
```bash
python -m src.slides --out-dir out
```
- Builds **`out/Loyalty_Growth_Deck.pptx`** using exported charts & metrics  
- **Import to Canva:** Home → **Create a design → Import file** → select the PPTX  
  - Or drag the PPTX onto the Canva **Home** page

---

## Methodology (short)

### Cleaning & normalization
- Tolerant column detection (case/spacing/underscore-insensitive)  
- Standardize headers; trim strings; **numericize** revenue fields  
- Parse **all date-like** columns → **UTC tz-aware** → tz-naive (safe arithmetic)  
- Choose **primary clock** by **coverage** (not by guess)

### Exploration
- Coverage % per event; min/max date; span  
- **Funnel** (created→shipped→delivered; returned/canceled where available)  
- Lead-time distributions (created→shipped, shipped→delivered, delivered→returned)

### Segmentation
- RFM quintiles (scores 1..5) → rule-based segments with precedence  
- Personalized churn windows from median inter-purchase cadence  
- Risk overlays (`is_at_risk`, `is_churned`)

### Predictive CLV
- **BG/NBD + Gamma-Gamma** baseline (stable on sparse repeat buying)  
- **Tweedie GLM** (zero-heavy, positive target; NaN-safe pipeline)  
- **Blend** for stability; judge with **MAE/RMSE + gains/deciles**

---

## Using the results (what to do now)

- **At-Risk Save (highest ROI):** trigger when `recency_days > 0.75 × churn_window`; two-step ladder: value-add (expedited shipping / exchange-first) → small voucher **≤10–15%** of predicted **CLV_90d**.  
- **Second-Order Accelerator:** nudge at **IPT + 7d** with starter bundles; free-ship threshold slightly **above AOV**.  
- **VIP Protection:** early access, exchange-first returns, **2-day** SLA (avoid blanket discounts).  
- **Ops → loyalty:** reduce the slowest ship→deliver cohorts; streamline returns in top-return categories.  
- **Targeting:** rank by **Blend**; size budget with the **gains curve** and **deciles**.

---

## Configuration

- `--data-dir` (default: `DE_DS_PracticeFiles/`)  
- `--out-dir` (default: `./out`)  
- `--horizon` (default: `90`)  
- Random seeds: sklearn `random_state=42` where applicable

---

## Troubleshooting

- **`orders_completed` missing** → run the build step or the full pipeline.  
- **`lifetimes` arg mismatch** (`transaction_data` vs `transactions`) → handled automatically; upgrade `lifetimes` if needed.  
- **scikit-learn RMSE** (`mean_squared_error(..., squared=False)` not supported) → the code uses a compatibility wrapper; or upgrade sklearn.  
- **Pandas tz warnings** → harmless; timestamps are converted version-safely.  
- **Canva won’t import PPTX** → use **Home → Create a design → Import file**, or **drag & drop** onto the Home page.

---

## Privacy

Use **non-PII** or anonymized data. Do **not** commit raw customer data. Add `DE_DS_PracticeFiles/` to `.gitignore` if needed.

---

## License

MIT (or your preferred license).

---

## Acknowledgements

- [`lifetimes`](https://github.com/CamDavidsonPilon/lifetimes) for BG/NBD & Gamma-Gamma  
- `scikit-learn` for Tweedie GLM & preprocessing  
- `python-pptx` for slide generation
