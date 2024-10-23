import xgboost as xgb

# Load the old model (the one that causes issues)
old_model = xgb.Booster()
old_model.load_model("xgb_model.bin")  # Update the path to your old model file

# Save the model in the new XGBoost format
old_model.save_model("xgb_model.bin")  # This will save it in the newer format
