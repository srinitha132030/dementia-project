import os
import joblib

os.makedirs("model", exist_ok=True)

joblib.dump(model, "model/dementia_model.pkl")
joblib.dump(scaler, "model/scaler.pkl")

print("✅ Files created!")