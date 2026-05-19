import os
import joblib
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

# Load DB Engine
DB_HOST = os.environ.get("DB_HOST")
DB_PORT = os.environ.get("DB_PORT", "5432")
DB_NAME = os.environ.get("DB_NAME")
DB_USER = os.environ.get("DB_USER")
DB_PASSWORD = os.environ.get("DB_PASSWORD")

DB_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
engine = create_engine(DB_URL)

def load_raw_data():
    """Pulls all available data from the rolling RDS window."""
    query = "SELECT * FROM properties"
    df = pd.read_sql(query, engine)
    return df

def engineer_targets(df):
    """Generates targets for both regression and classification."""
    # Ensure no missing values in key columns
    df = df.dropna(subset=["price", "neighborhood", "sqft"])
    
    # Regression Target
    y_reg = df["price"].values
    
    # Classification Target: Price vs Neighborhood Median
    # Group by neighborhood and compute median price per sqft
    df["price_per_sqft"] = df["price"] / (df["sqft"] + 1e-5)
    med_per_neighborhood = df.groupby("neighborhood")["price_per_sqft"].transform("median")
    
    # Ratio of current property price per sqft to neighborhood median
    ratio = df["price_per_sqft"] / (med_per_neighborhood + 1e-5)
    
    conditions = [
        (ratio < 0.9),
        (ratio >= 0.9) & (ratio <= 1.1),
        (ratio > 1.1)
    ]
    choices = [0, 1, 2] # 0: Bargain, 1: Fair, 2: Overpriced
    y_clf = np.select(conditions, choices, default=1)
    
    return df, y_reg, y_clf

def build_preprocessing_pipeline():
    """Defines how columns are handled based on their data type."""
    
    # Numeric pipeline: Scales features like square footage and room counts
    num_features = ["bedrooms", "bathrooms", "sqft"]
    
    # Categorical pipeline: Encodes structural location data
    cat_features = ["neighborhood"]
    
    # NLP / Text pipeline: Vectorizes descriptions to capture sentiment/amenities
    text_feature = "description"

    # Assemble unified feature transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_features),
            ("nlp", TfidfVectorizer(max_features=200, stop_words="english"), text_feature)
        ]
    )
    return preprocessor

def prepare_and_save_data():
    """Main execution block for feature engineering and dataset splitting."""
    df = load_raw_data()
    
    if len(df) < 10:
        print("Not enough data points in database to split. Skipping.")
        return None
        
    df, y_reg, y_clf = engineer_targets(df)
    
    # Drop features we won't feed to the pipeline matrix
    X = df[["bedrooms", "bathrooms", "sqft", "neighborhood", "description"]]
    
    # Train-test split (stratified on classification to maintain balanced subsets)
    X_train, X_test, y_train_reg, y_test_reg, y_train_clf, y_test_clf = train_test_split(
        X, y_reg, y_clf, test_size=0.2, random_state=42, stratify=y_clf
    )
    
    # Fit processor on training data only to prevent data leakage
    preprocessor = build_preprocessing_pipeline()
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)
    
    # Save the fitted transformer artifact locally so training and API code can use it
    os.makedirs("artifacts", exist_ok=True)
    joblib.dump(preprocessor, "artifacts/preprocessor.joblib")
    print("Feature engineering complete. Preprocessor saved to artifacts/preprocessor.joblib")
    
    # For execution simplicity across automated scripts, we pass processed matrices back as a dict
    return {
        "X_train": X_train_transformed,
        "X_test": X_test_transformed,
        "y_train_reg": y_train_reg,
        "y_test_reg": y_test_reg,
        "y_train_clf": y_train_clf,
        "y_test_clf": y_test_clf
    }

if __name__ == "__main__":
    prepare_and_save_data()