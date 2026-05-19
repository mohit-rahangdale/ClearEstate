import os
import requests
import pandas as pd
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text

# 1. Load Database Credentials from GitHub Secrets (Environment)
DB_HOST = os.environ.get("DB_HOST")
DB_PORT = os.environ.get("DB_PORT", "5432")
DB_NAME = os.environ.get("DB_NAME")
DB_USER = os.environ.get("DB_USER")
DB_PASSWORD = os.environ.get("DB_PASSWORD")

# Create SQLAlchemy Engine
DB_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
engine = create_engine(DB_URL)

def setup_database():
    """Creates the properties table if it doesn't exist."""
    create_table_query = """
    CREATE TABLE IF NOT EXISTS properties (
        property_id VARCHAR(50) PRIMARY KEY,
        title TEXT,
        description TEXT,
        price NUMERIC,
        bedrooms INT,
        bathrooms INT,
        sqft NUMERIC,
        neighborhood VARCHAR(100),
        fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """
    with engine.begin() as conn:
        conn.execute(text(create_table_query))
        print("Database schema verified.")

def purge_old_data(days=30):
    """Deletes records older than the specified rolling window."""
    purge_query = text("DELETE FROM properties WHERE fetched_at < NOW() - INTERVAL ':days days'")
    with engine.begin() as conn:
        result = conn.execute(purge_query, {"days": days})
        print(f"Purged {result.rowcount} old records.")

def fetch_and_store_new_data():
    """Fetches new listings from API and inserts them into RDS."""
    # Replace this URL with a real RapidAPI or open data endpoint for real estate
    API_URL = "https://raw.githubusercontent.com/datasets/house-prices-uk/master/data/data.csv" 
    
    print("Fetching new property data...")
    # For demonstration, we simulate an API call by reading a public CSV directly into a DataFrame
    # In production with a JSON API, you would use requests.get().json() and pd.DataFrame(data)
    try:
        df = pd.read_csv(API_URL).head(500) # Fetch latest 500 for the interval
    except Exception as e:
        print(f"Failed to fetch data: {e}")
        return

    # Standardize columns to match our database schema
    # (Mapping logic will depend on the exact API you choose)
    df = df.rename(columns={
        "id": "property_id",
        "Price": "price",
        "Town/City": "neighborhood"
    })
    
    # Add timestamp and fill missing columns for the schema
    df["fetched_at"] = datetime.now()
    if "description" not in df.columns:
         df["description"] = "Standard property in " + df["neighborhood"].astype(str)
    
    expected_cols = ["property_id", "title", "description", "price", "bedrooms", "bathrooms", "sqft", "neighborhood", "fetched_at"]
    
    # Ensure missing columns exist with nulls to prevent DB errors
    for col in expected_cols:
        if col not in df.columns:
            df[col] = None
            
    df = df[expected_cols]

    # Push to RDS (PostgreSQL)
    try:
        # 'append' adds to table, 'index=False' prevents writing pandas index
        df.to_sql("properties", engine, if_exists="append", index=False, method="multi")
        print(f"Successfully inserted {len(df)} new records into RDS.")
    except Exception as e:
        # Catch duplicate primary key errors if the API returns already-seen properties
        print(f"Insertion skipped or failed (possibly duplicates): {e}")

if __name__ == "__main__":
    setup_database()
    purge_old_data(days=30)
    fetch_and_store_new_data()