from pathlib import Path
import json
import numpy as np
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data_utils import load_data, prepare_data

def main():
    df = load_data()
    cleaned, feature_cols = prepare_data(df)

    X = cleaned[feature_cols]
    y = cleaned["price"]  # Bengaluru dataset target is typically in lakh rupees.

    categorical = ["location"]
    numeric = [c for c in feature_cols if c not in categorical]

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]), numeric),
            ("categorical", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore"))
            ]), categorical)
        ]
    )

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(
            n_estimators=250,
            max_depth=20,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        ))
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    metrics = {
        "MAE_lakh": float(mean_absolute_error(y_test, pred)),
        "RMSE_lakh": float(np.sqrt(mean_squared_error(y_test, pred))),
        "R2": float(r2_score(y_test, pred)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "features": feature_cols
    }

    Path("models").mkdir(exist_ok=True)
    Path("reports").mkdir(exist_ok=True)
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    joblib.dump(model, "models/house_price_pipeline.joblib")
    cleaned.to_csv("data/processed/cleaned_house_data.csv", index=False)
    with open("reports/model_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    print("\nModel evaluation (price unit: lakh rupees)")
    for key, value in metrics.items():
        print(f"{key}: {value}")
    print("\nSaved model to models/house_price_pipeline.joblib")

if __name__ == "__main__":
    main()
