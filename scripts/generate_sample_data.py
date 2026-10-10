"""Generate the reproducible sample datasets shipped in ``sample_data/``.

The data is synthetic but deliberately *realistic*: it has seasonality, growth,
regional differences, under-performing products, a few dominant customers and a
set of documented, injected anomalies so the anomaly detector has something
real to find.

Run:  python scripts/generate_sample_data.py
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
OUT = Path(__file__).resolve().parents[1] / "sample_data"
START, END = date(2023, 1, 1), date(2024, 12, 31)

REGIONS = {"West": 1.35, "North": 1.0, "East": 1.15, "South": 0.85, "Central": 0.7}
CHANNELS = {"Online": 0.5, "Retail": 0.3, "Partner": 0.2}

PRODUCTS = [
    # name, category, list_price, cost_ratio, popularity, trend_2024
    ("UltraBook Pro 14", "Electronics", 1299.0, 0.72, 1.2, 1.10),
    ("UltraBook Air 13", "Electronics", 999.0, 0.70, 1.4, 1.15),
    ("4K Monitor 27\"", "Electronics", 429.0, 0.62, 1.6, 1.05),
    ("Curved Monitor 34\"", "Electronics", 649.0, 0.64, 0.9, 1.20),
    ("Wireless Earbuds X", "Electronics", 149.0, 0.45, 2.2, 1.25),
    ("Noise-Cancel Headset", "Electronics", 249.0, 0.50, 1.3, 1.00),
    ("DVD Writer X2", "Electronics", 59.0, 0.55, 0.5, 0.35),
    ("Ergo Office Chair", "Furniture", 389.0, 0.58, 1.1, 1.05),
    ("Standing Desk Duo", "Furniture", 699.0, 0.60, 0.8, 1.30),
    ("Meeting Table 8-Seat", "Furniture", 1490.0, 0.65, 0.25, 0.95),
    ("Filing Cabinet Steel", "Furniture", 219.0, 0.52, 0.6, 0.55),
    ("Bookshelf Oak", "Furniture", 179.0, 0.50, 0.7, 0.90),
    ("Copy Paper A4 (Box)", "Office Supplies", 39.0, 0.40, 3.0, 1.00),
    ("Gel Pens (24 pk)", "Office Supplies", 14.0, 0.35, 2.6, 0.95),
    ("Whiteboard 6ft", "Office Supplies", 129.0, 0.45, 0.8, 1.00),
    ("Label Printer LX", "Office Supplies", 99.0, 0.55, 0.9, 1.05),
    ("Fax Machine Classic", "Office Supplies", 189.0, 0.60, 0.4, 0.30),
    ("Desk Organizer Set", "Office Supplies", 34.0, 0.38, 1.5, 1.00),
    ("Cloud Backup (1yr)", "Software", 119.0, 0.15, 1.6, 1.35),
    ("Security Suite (1yr)", "Software", 89.0, 0.12, 1.8, 1.20),
    ("Office Suite License", "Software", 249.0, 0.18, 1.2, 1.10),
    ("Design Studio Pro", "Software", 599.0, 0.20, 0.5, 1.15),
    ("Legacy CRM Desktop", "Software", 399.0, 0.25, 0.45, 0.40),
    ("Wireless Mouse M3", "Accessories", 39.0, 0.40, 2.8, 1.00),
    ("Mechanical Keyboard K8", "Accessories", 119.0, 0.48, 1.5, 1.10),
    ("USB-C Dock 12-in-1", "Accessories", 179.0, 0.50, 1.3, 1.20),
    ("Laptop Sleeve 14\"", "Accessories", 29.0, 0.35, 1.7, 0.95),
    ("Wi-Fi 6 Router", "Networking", 199.0, 0.55, 1.0, 1.10),
    ("Mesh Wi-Fi Kit (3)", "Networking", 349.0, 0.58, 0.8, 1.25),
    ("24-Port Switch", "Networking", 459.0, 0.60, 0.4, 1.00),
]

COMPANY_A = ["Apex", "Blue", "Cedar", "Delta", "Evergreen", "Falcon", "Granite", "Harbor", "Iris",
             "Juniper", "Keystone", "Lumen", "Maple", "Nimbus", "Orion", "Pioneer", "Quartz",
             "Redwood", "Summit", "Titan", "Unity", "Vertex", "Willow", "Zenith", "Atlas", "Beacon"]
COMPANY_B = ["Analytics", "Logistics", "Dynamics", "Health", "Labs", "Partners", "Systems",
             "Foods", "Energy", "Media", "Retail", "Capital", "Works", "Studios", "Group"]
COMPANY_C = ["Ltd", "Inc", "LLC", "Co", "PLC"]
CITIES = {
    "West": [("USA", "San Francisco"), ("USA", "Seattle"), ("USA", "Los Angeles")],
    "North": [("Canada", "Toronto"), ("USA", "Chicago"), ("USA", "Minneapolis")],
    "East": [("USA", "New York"), ("USA", "Boston"), ("UK", "London")],
    "South": [("USA", "Austin"), ("USA", "Atlanta"), ("USA", "Miami")],
    "Central": [("USA", "Denver"), ("USA", "Kansas City"), ("USA", "Dallas")],
}
FIRST = ["Alex", "Priya", "Jordan", "Mei", "Sam", "Fatima", "Liam", "Aisha", "Noah", "Sofia",
         "Ethan", "Zara", "Lucas", "Chloe", "Arjun", "Emma", "Omar", "Grace", "Ravi", "Hana"]
LAST = ["Patel", "Smith", "Garcia", "Chen", "Khan", "Johnson", "Silva", "Nguyen", "Brown",
        "Kim", "Singh", "Martin", "Lopez", "Wilson", "Ali", "Taylor"]


def make_products(rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for i, (name, cat, price, cost_ratio, _pop, _trend) in enumerate(PRODUCTS, start=1):
        launch = date(2019, 1, 1) + timedelta(days=int(rng.integers(0, 1400)))
        rows.append({
            "product_id": f"P{i:03d}",
            "product_name": name,
            "category": cat,
            "list_price": price,
            "unit_cost": round(price * cost_ratio, 2),
            "launch_date": launch.isoformat(),
        })
    return pd.DataFrame(rows)


def make_customers(rng: np.random.Generator, n: int = 320) -> pd.DataFrame:
    rows, used = [], set()
    region_names = list(REGIONS)
    for i in range(1, n + 1):
        segment = rng.choice(["Enterprise", "SMB", "Consumer"], p=[0.15, 0.45, 0.40])
        region = rng.choice(region_names, p=[0.24, 0.2, 0.22, 0.18, 0.16])
        country, city = CITIES[region][int(rng.integers(0, 3))]
        if segment == "Consumer":
            name = f"{rng.choice(FIRST)} {rng.choice(LAST)}"
        else:
            name = f"{rng.choice(COMPANY_A)} {rng.choice(COMPANY_B)} {rng.choice(COMPANY_C)}"
        while name in used:
            name = name + " II"
        used.add(name)
        rows.append({
            "customer_id": f"C{i:04d}",
            "customer_name": name,
            "segment": segment,
            "region": region,
            "country": country,
            "city": city,
            "signup_date": (date(2020, 1, 1) + timedelta(days=int(rng.integers(0, 1460)))).isoformat(),
        })
    return pd.DataFrame(rows)


def season(d: date) -> float:
    # Q4 peak, summer dip, mild January slump.
    return {1: 0.85, 2: 0.9, 3: 1.0, 4: 0.98, 5: 1.0, 6: 0.92, 7: 0.88, 8: 0.93,
            9: 1.02, 10: 1.08, 11: 1.3, 12: 1.4}[d.month]


def make_orders(rng: np.random.Generator, products: pd.DataFrame, customers: pd.DataFrame) -> pd.DataFrame:
    prod_pop = np.array([p[4] for p in PRODUCTS])
    prod_trend = np.array([p[5] for p in PRODUCTS])
    # A handful of "whale" customers drive a disproportionate share of revenue.
    cust_weight = np.where(customers["segment"].eq("Enterprise"), 3.0,
                           np.where(customers["segment"].eq("SMB"), 1.0, 0.45))
    cust_weight = cust_weight * rng.lognormal(0, 0.6, len(customers))
    whales = rng.choice(np.flatnonzero(customers["segment"].eq("Enterprise")), 5, replace=False)
    cust_weight[whales] *= 6
    cust_weight = cust_weight / cust_weight.sum()
    cust_region = customers["region"].to_numpy()

    rows = []
    order_no = 100001
    d = START
    total_days = (END - START).days
    while d <= END:
        t = (d - START).days / total_days
        growth = 1 + 0.35 * t
        weekday = 0.75 if d.weekday() >= 5 else 1.05
        lam = 7.5 * growth * season(d) * weekday
        n_orders = rng.poisson(lam)
        for _ in range(n_orders):
            ci = rng.choice(len(customers), p=cust_weight)
            region = cust_region[ci]
            # Regional mix: West sells more per order.
            pw = prod_pop.copy()
            if d.year == 2024:
                pw = pw * prod_trend  # under-performers decline in 2024
            pw = pw / pw.sum()
            pi = rng.choice(len(products), p=pw)
            prod = products.iloc[pi]
            seg = customers.iloc[ci]["segment"]
            base_qty = {"Enterprise": 9, "SMB": 4, "Consumer": 1.5}[seg]
            qty = max(1, int(rng.poisson(base_qty * REGIONS[region] ** 0.5)))
            price = round(float(prod["list_price"]) * rng.uniform(0.97, 1.03), 2)
            discount = float(rng.choice([0, 0, 0, 0.05, 0.1, 0.15, 0.2], p=[.3, .15, .1, .15, .15, .1, .05]))
            if seg == "Enterprise":
                discount = min(0.3, discount + 0.05)
            channel = rng.choice(list(CHANNELS), p=list(CHANNELS.values()))
            status = rng.choice(["Completed", "Completed", "Completed", "Completed", "Returned", "Cancelled"],
                                p=[0.3, 0.3, 0.2, 0.15, 0.03, 0.02])
            rows.append({
                "order_id": f"SO-{order_no}",
                "order_date": d.isoformat(),
                "customer_id": customers.iloc[ci]["customer_id"],
                "product_id": prod["product_id"],
                "region": region,
                "sales_channel": channel,
                "quantity": qty,
                "unit_price": price,
                "discount": discount,
                "order_status": status,
            })
            order_no += 1
        d += timedelta(days=1)

    df = pd.DataFrame(rows)

    # --- Regional outage: South almost no orders in March 2024 -----------------------
    mask = (df["region"] == "South") & df["order_date"].str.startswith("2024-03")
    drop_idx = df[mask].sample(frac=0.85, random_state=SEED).index
    df = df.drop(drop_idx).reset_index(drop=True)

    # --- Black-Friday style spike on 2024-11-29 -------------------------------------
    spike = df[df["order_date"].str.startswith("2024-11")].sample(70, random_state=SEED).copy()
    spike["order_date"] = "2024-11-29"
    spike["order_id"] = [f"SO-{order_no + i}" for i in range(len(spike))]
    order_no += len(spike)
    df = pd.concat([df, spike], ignore_index=True)

    # --- Bulk orders (extreme quantities) --------------------------------------------
    bulk = df.sample(4, random_state=7).index
    df.loc[bulk, "quantity"] = [480, 520, 600, 450]

    # --- Data-entry errors: negative quantities and a unit-price typo ----------------
    neg = df.drop(bulk).sample(3, random_state=11).index
    df.loc[neg, "quantity"] = [-3, -5, -2]
    typo = df.drop(bulk).drop(neg).sample(2, random_state=13).index
    df.loc[typo, "unit_price"] = df.loc[typo, "unit_price"] * 100

    # --- Missing discounts (~1.5%) ---------------------------------------------------
    miss = df.sample(frac=0.015, random_state=17).index
    df["discount"] = df["discount"].astype("float64")
    df.loc[miss, "discount"] = np.nan

    df["revenue"] = (df["quantity"] * df["unit_price"] * (1 - df["discount"].fillna(0))).round(2)

    # --- Exact duplicate rows (double-submitted orders) ------------------------------
    dups = df.sample(3, random_state=19)
    df = pd.concat([df, dups], ignore_index=True)

    df = df.sort_values(["order_date", "order_id"]).reset_index(drop=True)
    cols = ["order_id", "order_date", "customer_id", "product_id", "region", "sales_channel",
            "quantity", "unit_price", "discount", "revenue", "order_status"]
    return df[cols]


def make_targets(rng: np.random.Generator, orders: pd.DataFrame) -> pd.DataFrame:
    """Monthly revenue targets per region, set from the prior-year run-rate plus planned growth.

    Central is given an over-ambitious 2024 plan, so it misses target most months.
    """
    o = orders.assign(month=orders["order_date"].str[:7])
    actual = o.groupby(["region", "month"])["revenue"].sum()
    base = actual.groupby("region").median()
    rows = []
    for m in pd.period_range("2023-01", "2024-12", freq="M"):
        for region in REGIONS:
            growth = 1.0 if m.year == 2023 else 1.12
            if region == "Central" and m.year == 2024:
                growth = 1.45
            target = base[region] * season(m.to_timestamp().date()) * growth * rng.uniform(0.95, 1.05)
            rows.append({"region": region, "month": str(m), "revenue_target": round(target, -2)})
    return pd.DataFrame(rows)


def make_marketing(rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    months = pd.period_range("2023-01", "2024-12", freq="M")
    for m in months:
        for region, w in REGIONS.items():
            for ch, base in {"Search": 9000, "Social": 6000, "Email": 1500, "Events": 4000}.items():
                spend = base * w * season(m.to_timestamp().date()) * rng.uniform(0.85, 1.15)
                rows.append({"month": str(m), "region": region, "marketing_channel": ch,
                             "spend": round(spend, 2),
                             "leads": int(spend / rng.uniform(35, 60))})
    return pd.DataFrame(rows)


def make_messy() -> str:
    """A deliberately messy CSV used to demonstrate validation & data-quality checks."""
    return (
        "Employee ID, Full Name ,Department,Salary,Start Date,Bonus %,Rating,Rating\n"
        "E001,Ana Lopez,Sales,\"$72,500.00\",2021-03-15,10%,4.5,A\n"
        "E002,Ben Carter,Engineering,\"$98,000\",15/07/2020,12%,4.1,B\n"
        "E003,  Chen Wei ,Engineering,N/A,2019-11-01,,3.9,B\n"
        "E004,Dara Singh,Marketing,\"$64,250.50\",03/02/2022,8%,,C\n"
        ",,,,,,,\n"
        "E005,Eli Brooks,Sales,\"$71,000\",2022-08-30,9%,4.8,A\n"
        "E005,Eli Brooks,Sales,\"$71,000\",2022-08-30,9%,4.8,A\n"
        "E006,Farah Khan,HR,\"$58,900\",2018-01-09,5%,3.2,C\n"
        "E007,Gus Moreno,Engineering,\"$1,250,000\",2023-05-22,15%,4.0,B\n"
        "E008,Hiro Tanaka,engineering,\"$101,500\",2020-10-12,12%,4.3,A\n"
        "E009,Isla Novak,Sales,\"-$5,000\",2024-01-08,7%,2.9,D\n"
        "E010,Jon Petrov,Marketing,\"$66,800\",unknown,8%,3.7,B\n"
    )


def main() -> None:
    rng = np.random.default_rng(SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    products = make_products(rng)
    customers = make_customers(rng)
    orders = make_orders(rng, products, customers)
    marketing = make_marketing(rng)
    targets = make_targets(rng, orders)

    products.to_csv(OUT / "products.csv", index=False)
    customers.to_csv(OUT / "customers.csv", index=False)
    orders.to_csv(OUT / "sales_orders.csv", index=False)
    marketing.to_csv(OUT / "marketing_spend.csv", index=False)
    targets.to_csv(OUT / "regional_targets.csv", index=False)
    (OUT / "messy_hr_example.csv").write_text(make_messy(), encoding="utf-8")

    print(f"products: {len(products)}  customers: {len(customers)}  "
          f"orders: {len(orders)}  marketing: {len(marketing)}  targets: {len(targets)}")


if __name__ == "__main__":
    main()
