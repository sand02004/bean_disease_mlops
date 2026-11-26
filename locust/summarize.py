"""
locust/summarize.py
-------------------
Turns the Locust CSV results into one comparison table (for the README).

Run from the project folder after the tests:
    python locust/summarize.py
"""
import glob
import os
import re

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
files = sorted(glob.glob(os.path.join(ROOT, "locust", "results", "containers_*_stats.csv")))
if not files:
    raise SystemExit("No results found in locust/results/. Run the Locust tests first.")

rows = []
for path in files:
    n = int(re.search(r"containers_(\d+)_stats", path).group(1))
    df = pd.read_csv(path)
    for name in ["/predict", "Aggregated"]:
        r = df[df["Name"] == name]
        if r.empty:
            continue
        r = r.iloc[0]
        rows.append({
            "Containers": n,
            "Endpoint": "All requests" if name == "Aggregated" else name,
            "Requests": int(r["Request Count"]),
            "Failures": int(r["Failure Count"]),
            "Avg (ms)": round(r["Average Response Time"]),
            "Median (ms)": round(r["Median Response Time"]),
            "95th pct (ms)": round(r["95%"]),
            "Max (ms)": round(r["Max Response Time"]),
            "Requests/s": round(r["Requests/s"], 2),
        })

table = pd.DataFrame(rows).sort_values(["Endpoint", "Containers"], ascending=[False, True])
print(table.to_string(index=False))
md_path = os.path.join(ROOT, "locust", "results", "summary.md")
with open(md_path, "w") as f:
    cols = list(table.columns)
    f.write("| " + " | ".join(cols) + " |\n")
    f.write("|" + "|".join(["---"] * len(cols)) + "|\n")
    for _, row in table.iterrows():
        f.write("| " + " | ".join(str(row[c]) for c in cols) + " |\n")
print(f"\nMarkdown table saved to {md_path} (paste it into the README).")