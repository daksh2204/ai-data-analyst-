# Eval report: offline / heuristic-planner

- Run: 2026-10-10T15:09:59
- Pass rate: **12/12 (100%)**
- Latency: median 0.03s, max 0.13s
- Tokens: 0 in / 0 out
- Tool errors (self-corrected or not): 0

| Case | Result | Time | Tools | Notes |
|---|---|---|---|---|
| `top_region` | ✅ | 0.02s | run_sql, create_chart | - |
| `monthly_trend` | ✅ | 0.02s | run_sql, create_chart | - |
| `underperforming_products` | ✅ | 0.03s | run_sql, create_chart | - |
| `top_customers` | ✅ | 0.03s | run_sql, create_chart | - |
| `sql_for_previous` | ✅ | 0.03s | run_sql, create_chart | - |
| `anomalies` | ✅ | 0.13s | detect_anomalies | - |
| `targets` | ✅ | 0.02s | run_sql, create_chart | - |
| `forecast` | ✅ | 0.04s | forecast | - |
| `category_join` | ✅ | 0.02s | run_sql, create_chart | - |
| `data_quality` | ✅ | 0.01s | profile_table, profile_table, profile_table, profile_table | - |
| `distinct_customers` | ✅ | 0.01s | run_sql | - |
| `follow_up_by_month` | ✅ | 0.04s | run_sql, create_chart | - |
