from fastapi import FastAPI
import pandas as pd
import pickle
import numpy as np
from pydantic import BaseModel
from main import prepare_input_opt
app = FastAPI()

class CustomerInput(BaseModel):
    CreditScore: int
    Geography: str
    Gender: str
    Age: int
    Tenure: int
    Balance: float
    NumOfProducts: int
    HasCrCard: bool
    IsActiveMember: bool
    EstimatedSalary: float

def load_model(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)

# Load your models
xgboost_model = load_model('xgb_model.pkl')

@app.post("/predict")
def predict(customer: CustomerInput):
    input_dict = customer.dict()
    
    input_df = pd.DataFrame([input_dict])
    
    # Prepare input based on your previous function
    input_df = prepare_input_opt(input_df)
    
    # Make predictions
    probabilities = {
        'XGBoost': xgboost_model.predict_proba(input_df)[0][1],
    }
    
    avg_probability = np.mean(list(probabilities.values()))
    
    return {"Churn_Probability": avg_probability}
