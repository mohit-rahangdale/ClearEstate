import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Security, Depends
from fastapi.security.api_key import APIKeyHeader
from starlette.status import HTTP_403_FORBIDDEN
from api.schema import PropertyInput, PredictionResponse

# Setup Authentication Secret
API_KEY_NAME = "X-API-KEY"
API_KEY = os.environ.get("API_AUTH_SECRET", "default_fallback_secret_for_ci_tests")
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

app = FastAPI(
    title="Real Estate Intelligence & Sentiment Engine",
    description="Production ML API combining Tabular Ensembles and NLP Sentiment Vectors.",
    version="1.0.0"
)

# Global Variables to hold models in memory
preprocessor = None
reg_model = None
clf_model = None

# Price tier decoder map
TIER_MAP = {0: "Bargain", 1: "Fairly Priced", 2: "Overpriced"}

@app.on_event("startup")
def load_artifacts():
    """Loads all model components into memory once when the container initializes."""
    global preprocessor, reg_model, clf_model
    try:
        # Check current working directory to adapt to local vs docker run paths
        base_path = "artifacts" if os.path.exists("artifacts") else "../artifacts"
        
        preprocessor = joblib.load(os.path.join(base_path, "preprocessor.joblib"))
        reg_model = joblib.load(os.path.join(base_path, "regression_model.joblib"))
        clf_model = joblib.load(os.path.join(base_path, "classification_model.joblib"))
        print("All production model artifacts loaded successfully.")
    except Exception as e:
        print(f"Error loading model artifacts during startup: {e}")

async def get_api_key(header_key: str = Depends(api_key_header)):
    """Secures endpoints with an API token gate."""
    if header_key == API_KEY:
        return header_key
    raise HTTPException(status_code=HTTP_403_FORBIDDEN, detail="Invalid or missing API Key credentials.")

@app.get("/health", tags=["Monitoring"])
def health_check():
    """Endpoint for load balancers to monitor service availability."""
    if preprocessor and reg_model and clf_model:
        return {"status": "healthy", "artifacts_loaded": True}
    return {"status": "degraded", "artifacts_loaded": False}

@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
async def predict_property_metrics(payload: PropertyInput, api_key: str = Depends(get_api_key)):
    """Applies preprocessing pipelines and infers multi-task metrics."""
    if not (preprocessor and reg_model and clf_model):
        raise HTTPException(status_code=503, detail="Models are not loaded or initialized yet.")

    try:
        # Convert Pydantic payload directly to Pandas DataFrame structure expected by transformer
        raw_data = pd.DataFrame([{
            "bedrooms": payload.bedrooms,
            "bathrooms": payload.bathrooms,
            "sqft": payload.sqft,
            "neighborhood": payload.neighborhood,
            "description": payload.description
        }])

        # Process features (Zero Leakage - uses saved vocabulary/scalers)
        transformed_features = preprocessor.transform(raw_data)

        # Simultaneous multi-task inference
        predicted_price = float(reg_model.predict(transformed_features)[0])
        tier_code = int(clf_model.predict(transformed_features)[0])
        value_category = TIER_MAP.get(tier_code, "Unknown")

        return PredictionResponse(
            predicted_price=round(predicted_price, 2),
            value_category=value_category,
            status="success"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference Engine failure: {str(e)}")