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
async def predict(customer: CustomerInput):
    input_dict = customer.model_dump()
    
    input_df = pd.DataFrame([input_dict])
    
    # Prepare input based on your previous function
    input_df = prepare_input_opt(input_df)
    
    # Define the columns that were used during model training
    expected_columns = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 
                    'HasCrCard', 'IsActiveMember', 'EstimatedSalary', 
                    'Geography_France', 'Geography_Germany', 'Geography_Spain', 
                    'Gender_Female', 'Gender_Male']

    # Ensure input_df has only the expected columns
    input_df = input_df[expected_columns]
    
    # Make predictions using XGBoost
    probabilities = xgboost_model.predict_proba(input_df)[:, 1]  # Get probability of churn
    
    # Convert numpy array to Python list for JSON serialization
    churn_probabilities = probabilities.tolist()
    
    return {"Churn_Probability": churn_probabilities[0]}  # Assuming you're predicting for a single customer
