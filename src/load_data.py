# cd ~/m3/m3-forecasting
# source m3env/bin/activate
import pandas as pd
import numpy as np

def load_monthly_finance_data(
    data_path="data/M3C.xls",
    sheet_name="M3Month",
    category="FINANCE",
    horizon=18
):
    df = pd.read_excel(data_path, sheet_name=sheet_name)
    df.columns = df.columns.astype(str).str.strip()

    if "Category" not in df.columns:
        raise KeyError(f"'Category' column not found. Columns: {list(df.columns)}")
    cats = df["Category"].astype(str).str.strip().str.upper()
    df = df.assign(Category=cats)
    df = df[df["Category"] == category.upper()]

    series_data = []
    for _, row in df.iterrows():
        values = row.iloc[5:].dropna().astype(float).to_numpy()
        if len(values) <= horizon:
            continue
        series_data.append({
            "id": str(row.get("Series", "")),
            "category": row["Category"],
            "train": values[:-horizon],
            "test": values[-horizon:]
        })

    print(f"Prepared {len(series_data)} series for modeling.")
    return series_data

if __name__ == "__main__":
    data = load_monthly_finance_data()
    print(f"Loaded {len(data)} series total.")
    if data:
        print(f"Example ID: {data[0]['id']}")
        print(f"Train length: {len(data[0]['train'])}, Test length: {len(data[0]['test'])}")