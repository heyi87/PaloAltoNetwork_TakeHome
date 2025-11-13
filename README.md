# TakeHome Customer Analytics Project

This repository contains a refactored version of the original `TakeHome.ipynb` notebook, turned into a reusable Python package plus a small analysis toolkit.

The project:

- Cleans and standardizes messy transactional data (`USERS`, `PRODUCTS`, `ORDERS`, `ORDER_ITEMS`)
- Builds an enriched order table with **revenue** and **lifecycle events**
- Produces core **descriptive analytics** (profile, funnel, coverage, time window)
- Performs **RFM-based customer segmentation**
- Exports **CSV tables and plots** that are easy to reuse in a slide deck or further modeling

There is also a **slide deck** `Presentation.pdf` in the repo that explains the overall logic and presents key insights in a business-friendly way.

---

## 1. Data & Problem Overview

The dataset describes an e-commerce–style business with:

- **Users** (customers)
- **Products**
- **Orders**
- **Order items** (line-level detail)

From the analysis (after cleaning):

- **100,000 unique users**
- **82,480 unique orders**
- **29,120 unique products**
- **63,987 line items**
- **Total revenue:** ≈ **572,956.67** (dataset currency)
- **Average order value (AOV):** ≈ **6.95**

**Time coverage**

- Primary date: **`created_at`** (order creation time)
- Data window: **2019-01-10 → 2025-08-27**
  - ≈ **2,422 days**
  - ≈ **6.63 years**
  - **80 distinct months with orders** (essentially continuous activity)

**What someone reading the original notebook would want to know:**

1. **Is the data usable and how clean is it?**  
2. **What is the size and shape of the business?** (users, orders, revenue, time coverage)  
3. **How do orders progress through the lifecycle (created → shipped → delivered → returned/canceled)?**  
4. **Which countries, categories, brands, and products drive the business?**  
5. **How are customers segmented (RFM), and what makes “good” customers different from the rest?**  
6. **What artifacts are produced for downstream modeling/analysis?**

This package is organized to answer exactly those questions.
## Code structure

```text
takehome-project/
├─ README.md
├─ pyproject.toml           # optional, if you want to install as a package
└─ takehome/
   ├─ __init__.py
   └─ pipeline.py
   └─ utils_refactored.py
