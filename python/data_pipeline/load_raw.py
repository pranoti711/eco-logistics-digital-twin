from pathlib import Path
import pandas as pd
from sqlalchemy import create_engine

# CHANGE these three lines
USER = "postgres"
PASSWORD = "IASTOBE7"
PORT = 5432

HOST = "localhost"
DB = "eco_logistics_twin"

engine = create_engine(f"postgresql+psycopg2://{USER}:{PASSWORD}@{HOST}:{PORT}/{DB}")

raw_dir = Path(__file__).resolve().parents[2] / "data" / "raw"

for csv in sorted(raw_dir.glob("*.csv")):
    table = csv.stem  # e.g. olist_orders_dataset
    print(f"Loading {csv.name} ...")
    first = True
    # read everything as text so raw data is stored exactly as it is
    for chunk in pd.read_csv(csv, dtype=str, chunksize=100_000):
        chunk.to_sql(
            table, engine, schema="raw",
            if_exists="replace" if first else "append",
            index=False,
        )
        first = False
    print(f"  done: {table}")

print("All files loaded.")