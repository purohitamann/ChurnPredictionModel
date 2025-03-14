
   <img width="1042" height="545" alt="Screenshot 2025-03-14 at 3 23 51 PM" src="https://github.com/user-attachments/assets/10635a7d-c422-49fc-82ca-51ba721ebe36" />



# Customer Churn Prediction Model

This repository contains a machine learning model that predicts customer churn. It uses several machine learning classifiers like XGBoost, Random Forest, and K-Nearest Neighbors, along with techniques for hyperparameter tuning using Grid Search and Randomized Search.

<img width="368" alt="Screenshot 2025-03-14 at 3 22 21 PM" src="https://github.com/user-attachments/assets/2e90ebbb-cff4-4312-b925-b6174e8cf446" />
<img width="436" height="292" alt="Screenshot 2025-03-14 at 3 23 06 PM" src="https://github.com/user-attachments/assets/c3ec73bf-7711-41aa-a16b-05c005862d48" />



## Features

- Predicts customer churn using machine learning models.
- Provides explanations for churn predictions using OpenAI API.
- Includes hyperparameter tuning for optimizing model performance.
- Interactive web app built using Streamlit.
- Visualizations for customer metrics and model performance.

## Installation

To install the project and its dependencies, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/ChurnPredictionModel.git
   cd ChurnPredictionModel
   Create and activate a virtual environment (optional but recommended):
   ```

```bash

python3 -m venv .venv
source .venv/bin/activate # For Linux/Mac
Install the dependencies:
```
```bash

pip install -r requirements.txt
Usage
Running the Streamlit App
You can run the interactive web app locally using Streamlit:
```
```bash

streamlit run main.py
This will launch a local server, and you can open the app in your browser.
```

Model Details
The project uses the following machine learning classifiers:

XGBoost Classifier
Random Forest Classifier
K-Nearest Neighbors
The models are optimized using hyperparameter tuning (Grid Search and Randomized Search) to enhance performance.

