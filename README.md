# Loyalty & Growth: Customer Segmentation & 90-Day CLV Modeling

This project analyzes an e-commerce dataset to answer a core business question:

> **How can we boost customer loyalty and accelerate sales?**

Using noisy, semi-structured transactional data, the notebook `TakeHome.ipynb` ingests and cleans raw files, performs exploratory data analysis (EDA), builds RFM-based customer segments, and trains a 90-day Customer Lifetime Value (CLV) model. The slide deck `Presentation.pdf` summarizes the key findings and recommendations for business stakeholders.

---

## Repository Structure

- `TakeHome.ipynb`  
  End-to-end analysis notebook:
  - Data ingestion & cleaning  
  - Exploratory data analysis (EDA)  
  - RFM + AOV + tenure segmentation  
  - 90-day CLV modeling and evaluation  
  - Strategic recommendations

- `Presentation.pdf`  
  Executive presentation of the project: problem framing, visuals from EDA, model results, and recommendations.

- `DE_DS_PracticeFiles/`  
  Raw input data:
  - `USERS.csv` – user demographics (JSON-ish text per row)  
  - `PRODUCTS.csv` – product catalog (JSON array)  
  - `ORDERS.csv` – order-level metadata  
  - `ORDER_ITEMS.csv` – line-item detail

- `utils_refactored.py`  
  Helper utilities for parsing, cleaning, and plotting (standardizing columns, trimming strings, datetime coercion, etc.).

- `out/`  
  Folder where the notebook saves key plots (e.g., `seg_counts.png`, `seg_revenue.png`, `top_categories_revenue.png`, `order_timeline_coverage_full.png`, etc.).

---

## Environment & How to Run

**Dependencies**

- Python 3.9+
- `pandas`, `numpy`
- `matplotlib`
- `scikit-learn`
- `lifetimes`
- `jupyter` / `jupyterlab`

You can install them with:

```bash
pip install -r requirements.txt
# or
pip install pandas numpy matplotlib scikit-learn lifetimes jupyter
```

## 1. Problem Framing & Approach

**Business problem**

The business wants to increase revenue by improving customer loyalty in an online retail setting. The key questions are:

- Who are our most and least valuable customers?
- Which customer segments drive revenue today?
- How can we predict future customer value to better target retention and offers?
- Where in the order lifecycle do we risk losing customers (e.g., shipping delays, returns)?

**Analytical strategy**

### Understand customer behavior through EDA

- Quantify basic dataset properties (users, orders, products, revenue, timeframe).
- Explore engagement (orders per user, AOV, order lifecycle, returns, geos, categories).

### Build interpretable customer segments (RFM + AOV + tenure)

- Compute **Recency, Frequency, Monetary, Average Order Value, and Tenure** at a fixed snapshot date.
- Turn R/F/M into quintile scores (1–5) and assign intuitive segments such as **VIP, Loyal, Potential Loyalist, At Risk, Dormant, Regulars** using rule-based thresholds.
- Use these segments to understand which customers drive revenue and which are at risk of churn.

### Predict 90-day CLV instead of pure churn

- I initially considered a binary churn model, but the positive class (“churn within horizon”) was extremely rare, yielding weak precision–recall performance and limited business actionability.
- Instead, I framed the problem as predicting spend in the next 90 days (`CLV_90d`) and using that to rank customers for targeting.

### Overlay model predictions on segments to drive actions

- For every customer, compute **RFM segment + `CLV_90d` prediction**.
- Use the combination (segment × predicted value) to prioritize offers, win-back campaigns, and lifecycle strategies.

This flow is also summarized on page 3 of the presentation (Data → Segment → Predict → Act).

---

## 2. Data Engineering & EDA

**Data cleaning & integration**

The project integrates four heterogeneous sources:

- **USERS**: rows contain JSON-like strings such as `{"id":..., "":"/firstname/Brad", ...}`.
  - Implemented a robust parser (`parse_users_row`) to extract the core `id` and `/key/value` pairs.
  - Standardized key names (e.g., `firstname` → `first_name`, `lastname` → `last_name`) and coerced datetime fields.

- **PRODUCTS**: stored as a JSON array inside a text file or a single-cell CSV.
  - Read as raw text; if it begins with `[`, parse as JSON; otherwise treat first cell as a JSON array and load into a DataFrame.

- **ORDERS & ORDER_ITEMS**: standard CSVs with mixed types.
  - Standardized column names, trimmed strings, coerced datetime columns, and joined with users/products on `user_id` and product IDs.

**Key engineering steps**

- Filtered out cancelled / returned / refunded items to compute **net revenue**, relying on status text and `returned_at` timestamps.
- Derived `order_date` from `created_at` and per-line **revenue** as `sale_price × quantity`.
- Computed order-level aggregates (order revenue, AOV) and per-customer aggregates used for RFM and CLV features.

### EDA – key findings & visuals

The EDA uncovers the overall shape of the business (see pages 4–8 of the presentation).

**Dataset size & revenue**

- ~100k users, 82k+ orders, ~29k products, and ~64k line items.
- Total revenue ≈ **$573k**, with an **average order value ≈ $6.95**.
- Data spans from **2019-01-10 to 2025-08-27** (~6.5 years).

**Customer engagement**

- ~68% of customers place exactly **1 order** and ~22% place **2 orders**; only ~10% place **3+ orders**.
- This is a low-frequency environment where the biggest lever is turning one-time buyers into repeat customers.

**Order lifecycle**

- Most loss occurs **before shipment**; among delivered orders, ~19% are returned.
- Typical door-to-door time is ~4 days; orders that take much longer are likely to feel risky or frustrating.
- Returns cluster soon after delivery, indicating post-delivery experience (fit, quality) is crucial.

**Category & geography**

- Top categories include **underwear, jeans, socks, swim trunks, and travel accessories**; there is no single “hero SKU”, suggesting a long-tail catalog.
- Revenue is concentrated in a few key countries (e.g., **United States and China**), with smaller contributions from others.

**Key charts saved to `out/` include:**

- `order_timeline_coverage_full.png` – coverage by lifecycle event.
- `top_categories_revenue.png` – revenue by category.
- `top_products_by_quantity.png` – best-selling SKUs by units.
- `top_countries_customers.png`, `top_countries_revenue.png` – geographic mix.

---

## 3. Analysis & Modeling Findings

### 3.1 Justify Your Methodology

#### RFM + AOV + Tenure segmentation

For each customer at snapshot date **S = last order date + 1 day**, I compute:

- **Recency (`recency_days`)** – days since last order (smaller is better).
- **Frequency** – number of distinct orders.
- **Monetary** – net revenue (excluding returns/cancels).
- **Average Order Value (AOV)** – monetary / frequency.
- **Tenure (`tenure_days`)** – days since first order.

Each of R, F, M is binned into quintile scores (1–5). Segment rules (first match wins) are:

- **VIP**: `r ≥ 4` AND `f ≥ 4` AND `m ≥ 4`
- **Loyal**: `r ≥ 4` AND (`f ≥ 4` OR `m ≥ 4`)
- **Potential Loyalist**: `r ≥ 3` AND `f ≥ 3`
- **At Risk**: `r ≤ 2` AND (`f ≥ 3` OR `m ≥ 3`) – strong history, stale recency
- **Dormant**: `r ≤ 2` AND `f ≤ 2` AND `m ≤ 2`
- **Regulars**: everyone else

This approach is:

- **Appropriate** because it is simple, interpretable, and widely used in CRM.
- **Strength**: segments are explainable to non-technical stakeholders, and rules can be tuned.
- **Limitation**: purely snapshot-based; it doesn’t capture channel, product preferences, or seasonality.

#### 90-day CLV model (Tweedie regression)

To move beyond static segments, I build a **90-day CLV** model:

- **Label (target)**: total spend from S → S+90 days (`CLV_90d`).

**Features**

- RFM features: recency, frequency, monetary, AOV, tenure.
- Basic demographics: age, gender, state, country, city (where available).

**Model**

- `TweedieRegressor` (GLM with log link).
- Designed for **zero-inflated, highly skewed** targets (many zeros, few high spenders).
- Predicts expected spend directly in dollars.

**Preprocessing**

- Numeric features are imputed (median) and standardized.
- Categorical features are imputed and one-hot encoded via a `ColumnTransformer` inside a `Pipeline`.

**Strengths**

- Handles zero vs. positive spend naturally.
- Produces a smooth, monotonic relationship between features and predicted value.
- Fast to train and easy to interpret globally (feature importance / partial effects).

**Limitations**

- Uses only aggregate behavioral features; no sequence modeling or product affinity.
- 90-day horizon captures near-term value but not full lifetime.
- Assumes relatively stable behavior across the train/test window.

### 3.2 Present Your Findings

#### Segment performance

From the RFM segmentation (see pages 9–12):

- **VIP**
  - ~10% of customers, ~28% of revenue.
  - Frequency ≈ 1.8× and spend ≈ 2.6× the overall average.

- **At Risk**
  - ~23% of customers, ~34% of revenue historically; strong past value but stale recency.
  - AOV ≈ 1.7× average; recency ≈ 1.9× worse (staler).

- **Regulars**
  - ~30% of customers, ~11% of revenue; mostly one-order customers with modest value.

- **Dormant**
  - ~17% of customers and effectively 0% of revenue in the recent window.

Bubble plots of **Recency vs Frequency** show:

- VIP and Loyal segments are recent & frequent.
- At Risk customers have solid frequency/spend but very stale recency.
- Regulars cluster around one order and moderate recency.
- Dormant customers are extremely stale with no recent revenue.

#### CLV model performance & alignment with segments

On the test set (pages 14–15):

- **Average 90-day revenue per customer**: \$0.81.

- **Top 10% by predicted `CLV_90d`:**
  - Actual \$1.92 per customer (~2.4× the average).
  - Captures ~24% of 90-day revenue (vs 10% at random).

- **Top 20% by predicted `CLV_90d`:**
  - Captures ~37% of 90-day revenue (vs 20% at random).

Gains curves and decile calibration plots show that the model:

- Clearly separates high- vs low-value customers.
- Concentrates future revenue in the highest-scored deciles, ideal for targeting promotions.

**Alignment with segments**

- **VIP / Loyal**: highest actual and predicted `CLV_90d` – true top-value customers.
- **At Risk**: historically valuable, but much lower predicted `CLV_90d` – “value at risk”.
- **Regulars**: moderate spend and moderate predicted CLV – core volume engine.
- **Dormant**: low historical and predicted `CLV_90d` – not worth heavy investment.

---

## 4. Strategic Recommendations

### 4.1 Actionable Insights

Based on the segmentation + CLV model, I recommend the following actions:

**Protect and grow VIP & Loyal customers**

- ~16% of customers contribute ~29% of 90-day revenue.
- Offer priority service, early access to new products, and personalized bundles.
- Use `CLV_90d` to rank within these segments for ultra-VIP treatment.

**Strengthen Regulars (the core revenue engine)**

- ~40% of customers generating ~51% of 90-day revenue.
- Trigger post-purchase flows (thank-you, usage tips, cross-sell recommendations) to drive the second and third orders.
- Promote low-friction re-order experiences for frequently purchased categories (e.g., underwear, socks).

**Targeted save program for At Risk customers**

- Segment has strong historical spend but lower predicted `CLV_90d`.
- Use CLV scores to focus win-back offers only on high-value At Risk customers (e.g., top 30–40% by predicted `CLV_90d`).
- Test incentive levels: soft nudges (reminders, new arrivals) vs. limited-time discounts.

**Low-touch strategy for Dormant customers**

- Minimal near-term value; keep them on automated, low-cost communication.
- Re-engage only if they show new activity signals (site visit, email open, add-to-cart).

**Operational improvements along the order lifecycle**

- Reduce shipping delays beyond ~5 days and address the ~19% return rate post-delivery.
- Experiment with better size/fit guidance, clearer product photos, and an “exchange-first” returns flow.

### 4.2 Business Impact & Measurement

The recommendations are designed to both **boost loyalty** and **grow revenue** while protecting margins.

**Key KPIs to track (pages 17–18):**

- **VIP & Loyal uplift**
  - 90-day repeat purchase rate vs. baseline.
  - `CLV_90d` uplift for VIP/Loyal vs. control.

- **At Risk save program**
  - Reactivation rate (any purchase in 90 days).
  - Incremental `CLV_90d` vs. holdout group.

- **Regulars → Loyal/VIP progression**
  - % of Regulars that become Loyal/VIP within 6–12 months.
  - Their `CLV_90d` before vs. after progression.

- **Promo ROI & discount governance**
  - Promotion ROI = profit / promo spend by segment.
  - Average discount as % of predicted `CLV_90d`, to ensure we are not over-discounting low-value customers.

---

**Limitations & Next Steps**

- Incorporate channel and product affinity features (e.g., email engagement, category preferences).
- Move from one-shot CLV to sequence / time-series models for repeat purchases.
- Add uplift modeling to optimize “who to treat” rather than just “who is valuable”.
- Productionize as a weekly or monthly scoring pipeline feeding into CRM and experimentation platforms.

For a visual walkthrough of these steps and findings, please see `Presentation.pdf`.
