# from fastapi import FastAPI, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# import joblib
# import pandas as pd
# import numpy as np
# from rainfall_utils import get_rainfall

# app = FastAPI(title="Crop Recommendation API")

# # CORS
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Load Models and Artifacts
# try:
#     model_y = joblib.load("models/model_yield")
#     model_p = joblib.load("models/model_production")
#     scaler = joblib.load("models/scaler.pkl")
#     columns = joblib.load("models/columns.pkl")
#     unique_crops = joblib.load("models/unique_crops.pkl")
# except FileNotFoundError as e:
#     raise RuntimeError(f"Missing model file: {e}")

# numeric_features = ['Area', 'Annual_Rainfall', 'Fertilizer', 'Pesticide']

# class CropConditions(BaseModel):
#     Crop_Year: int
#     Season: str
#     State: str
#     Area: float
#     Fertilizer: float
#     Pesticide: float
#     Annual_Rainfall: float | None = None

# @app.post("/recommend")
# def recommend_crops(conditions: CropConditions):
#     cond = conditions.dict()

#     # Input Validation
#     if cond['Area'] <= 0:
#         raise HTTPException(status_code=400, detail="Area must be positive")
#     if cond['Fertilizer'] < 0:
#         raise HTTPException(status_code=400, detail="Fertilizer must be non-negative")
#     if cond['Pesticide'] < 0:
#         raise HTTPException(status_code=400, detail="Pesticide must be non-negative")
#     if cond['Crop_Year'] < 1900 or cond['Crop_Year'] > 2100:
#         raise HTTPException(status_code=400, detail="Invalid Crop_Year")

#     # Auto-fill Rainfall
#     if cond["Annual_Rainfall"] is None:
#         try:
#             cond["Annual_Rainfall"] = get_rainfall(cond["State"], cond["Crop_Year"], cond["Season"])
#         except Exception as e:
#             raise HTTPException(status_code=500, detail=f"Error fetching rainfall: {e}")

#     # Create Test DataFrame
#     test_df = pd.DataFrame({
#         'Crop': unique_crops,
#         'Crop_Year': [cond['Crop_Year']] * len(unique_crops),
#         'Season': [cond['Season']] * len(unique_crops),
#         'State': [cond['State']] * len(unique_crops),
#         'Area': [cond['Area']] * len(unique_crops),
#         'Annual_Rainfall': [cond['Annual_Rainfall']] * len(unique_crops),
#         'Fertilizer': [cond['Fertilizer']] * len(unique_crops),
#         'Pesticide': [cond['Pesticide']] * len(unique_crops)
#     })

#     # One-Hot Encoding
#     test_df = pd.get_dummies(test_df, columns=['Crop', 'Season', 'State'], drop_first=True)

#     # Add Missing Columns
#     for col in columns:
#         if col not in test_df.columns:
#             test_df[col] = 0

#     # Reorder Columns
#     test_df = test_df[columns]

#     # Preprocess Numeric Features
#     test_df[numeric_features] = np.log1p(test_df[numeric_features])
#     test_df[numeric_features] = scaler.transform(test_df[numeric_features])

#     # Convert Dummy Columns to Integer
#     dummy_columns = [col for col in test_df.columns if col not in numeric_features + ['Crop_Year']]
#     test_df[dummy_columns] = test_df[dummy_columns].astype(int)

#     # Predictions
#     y_pred = model_y.predict(test_df)
#     p_pred = model_p.predict(test_df)

#     # Output DataFrame
#     df_out = pd.DataFrame({
#         "Crop": unique_crops,
#         "Predicted_Yield": y_pred,
#         "Predicted_Production": p_pred
#     })
#     df_out["Score"] = 0.5 * (df_out["Predicted_Yield"] + df_out["Predicted_Production"])

#     # Top 3 Crops
#     top_crops = df_out.sort_values("Score", ascending=False).head(3)

#     return top_crops.to_dict(orient="records")

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from rainfall_utils import get_rainfall

app = FastAPI(title="Crop Recommendation API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Model and Artifacts
try:
    model = joblib.load("models/crop_classifier")
    scaler = joblib.load("models/scaler.pkl")
    columns = joblib.load("models/columns.pkl")
    unique_crops = joblib.load("models/unique_crops.pkl")
except FileNotFoundError as e:
    raise RuntimeError(f"Missing model file: {e}")

numeric_features = ['Area', 'Annual_Rainfall', 'Fertilizer', 'Pesticide']
dummy_columns = [col for col in columns if col not in numeric_features + ['Crop_Year']]
cat_features = [columns.index(col) for col in dummy_columns]  # Categorical feature indices

class CropConditions(BaseModel):
    Crop_Year: int
    Season: str
    State: str
    Area: float
    Fertilizer: float
    Pesticide: float
    Annual_Rainfall: float | None = None

@app.post("/recommend")
def recommend_crops(conditions: CropConditions):
    cond = conditions.dict()

    # Input Validation
    if cond['Area'] <= 0:
        raise HTTPException(status_code=400, detail="Area must be positive")
    if cond['Fertilizer'] < 0:
        raise HTTPException(status_code=400, detail="Fertilizer must be non-negative")
    if cond['Pesticide'] < 0:
        raise HTTPException(status_code=400, detail="Pesticide must be non-negative")
    if cond['Crop_Year'] < 1900 or cond['Crop_Year'] > 2100:
        raise HTTPException(status_code=400, detail="Invalid Crop_Year")

    # Auto-fill Rainfall
    if cond["Annual_Rainfall"] is None:
        try:
            cond["Annual_Rainfall"] = get_rainfall(cond["State"], cond["Crop_Year"], cond["Season"])
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error fetching rainfall: {e}")

    # Create Test DataFrame (one instance, as classification predicts crop probabilities)
    test_df = pd.DataFrame({
        'Crop_Year': [cond['Crop_Year']],
        'Season': [cond['Season']],
        'State': [cond['State']],
        'Area': [cond['Area']],
        'Annual_Rainfall': [cond['Annual_Rainfall']],
        'Fertilizer': [cond['Fertilizer']],
        'Pesticide': [cond['Pesticide']]
    })

    # One-Hot Encoding
    test_df = pd.get_dummies(test_df, columns=['Season', 'State'], drop_first=True)

    # Add Missing Columns
    for col in columns:
        if col not in test_df.columns:
            test_df[col] = 0

    # Reorder Columns
    test_df = test_df[columns]

    # Preprocess Numeric Features
    test_df[numeric_features] = np.log1p(test_df[numeric_features])
    test_df[numeric_features] = scaler.transform(test_df[numeric_features])

    # Convert Dummy Columns to Integer
    dummy_columns = [col for col in test_df.columns if col not in numeric_features + ['Crop_Year']]
    test_df[dummy_columns] = test_df[dummy_columns].astype(int)

    # Predictions: Get probabilities for all classes (crops)
    proba = model.predict_proba(test_df)[0]  # Probabilities for the single instance
    crop_probs = pd.DataFrame({
        "Crop": model.classes_,  # All unique crops from training
        "Probability": proba
    })

    # Top 3 Crops by probability
    top_crops = crop_probs.sort_values("Probability", ascending=False).head(3)

    # Format as dict (adjust if you want to include yield/production; for now, just crop and score as probability)
    return top_crops.to_dict(orient="records")