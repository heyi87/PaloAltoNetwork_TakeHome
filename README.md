# TakeHome Customer Analytics Package

This is a refactored version of the original `TakeHome.ipynb` notebook, turned into a reusable Python package.

The goal is to:

- Load and clean the raw transactional data (`USERS.csv`, `PRODUCTS.csv`, `ORDERS.csv`, `ORDER_ITEMS.csv`)
- Build enriched orders with revenue and lifecycle events
- Compute dataset-level profile and time coverage
- Build a simple order funnel and other EDA outputs
- Compute customer-level RFM features and segments
- Export clean CSVs and plots under an `out/` directory

## Project structure

```text
takehome-project/
├─ README.md
├─ pyproject.toml           # optional, if you want to install as a package
└─ takehome/
   ├─ __init__.py
   └─ pipeline.py
   └─ utils_refactored.py
