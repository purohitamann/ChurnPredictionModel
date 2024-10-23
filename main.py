import streamlit as st
import pandas as pd
import pickle
import numpy as np
import os
import utils as ut

from openai import OpenAI

import scipy.stats as stats

client = OpenAI(base_url="https://api.groq.com/openai/v1",
                api_key=os.environ['GROQ_API_KEY'])


def prepare_input_opt(input_df):
    # Add derived features if they are not already present
    input_df['CLV'] = input_df['Balance'] * input_df['EstimatedSalary'] / 100000
    input_df['TenureAgeRatio'] = input_df['Tenure'] / input_df['Age']

    # Create age group features
    input_df['AgeGroup_MiddleAge'] = np.where((input_df['Age'] >= 40) & (input_df['Age'] < 60), 1, 0)
    input_df['AgeGroup_Senior'] = np.where((input_df['Age'] >= 60), 1, 0)
    input_df['AgeGroup_Elderly'] = np.where((input_df['Age'] >= 75), 1, 0)

    # Define the correct order of the features
    expected_columns = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 
                        'IsActiveMember', 'EstimatedSalary', 'Geography_France', 
                        'Geography_Germany', 'Geography_Spain', 'Gender_Female', 
                        'Gender_Male', 'CLV', 'TenureAgeRatio', 'AgeGroup_MiddleAge', 
                        'AgeGroup_Senior', 'AgeGroup_Elderly']

    # Ensure the input_df has all expected columns, and reorder them accordingly
    input_df = input_df.reindex(columns=expected_columns, fill_value=0)

    return input_df
def calculate_percentiles(selected_customer, df):
    # Calculate percentiles for each relevant metric
    percentiles = {
        'CreditScore Percentile':
        stats.percentileofscore(df['CreditScore'],
                                selected_customer['CreditScore']),
        'Age Percentile':
        stats.percentileofscore(df['Age'], selected_customer['Age']),
        'Tenure Percentile':
        stats.percentileofscore(df['Tenure'], selected_customer['Tenure']),
        'Balance Percentile':
        stats.percentileofscore(df['Balance'], selected_customer['Balance']),
        'NumOfProducts Percentile':
        stats.percentileofscore(df['NumOfProducts'],
                                selected_customer['NumOfProducts']),
        'EstimatedSalary Percentile':
        stats.percentileofscore(df['EstimatedSalary'],
                                selected_customer['EstimatedSalary']),
    }

    # Return the dictionary of percentiles
    return percentiles


def explain_prediction(probability, input_dict, surname):
    prompt = f"""You are an expert data scientist at a bank, ehre you specialize in interpreting and explianing predictions of the machine learning models. 
    
    Your machine leatrning model has predicted that a customer named {surname} has a {round(probability*100,1)}% probability of churning, based on the information provided below. 
    
    Here is the Customer's Information:
    {input_dict}
    
    Here are the machine learning model's top 10 most important features for predicitng churn:
    	feature	        |   importance
------------------------------------------
	NumOfProducts.      |	0.323888
	IsActiveMember	    |   0.164146
	Age	                |   0.109550
	Geography_Germany	|   0.091373
	Balance             | 	0.052786
	Geography_France	|   0.046463
	Gender_Female	    |   0.045283
	Geography_Spain	    |   0.036855
	CreditScore	        |   0.035005
	EstimatedSalary     |	0.032655
	HasCrCard	        |   0.031940
	Tenure	            |   0.030054
	Gender_Male         |	0.000000

{pd.set_option('display.max_columns', None)}

Here are summary statistics of the Churned Customer's information:
{df[df['Exited']==1].describe()}

-If the Cutomer has over a 40% risk of churning, generate a 3 sentence explantion of why tehy are at the risk of churining.
-If the customer has less than a 40% risk of churning, generate a 3 sentence explanation of why they might not be at risk of churning.
- Your explanation should be based on the customer's information, the sunmmary statistics of churned amnd non-churned cuystomers, and the feaure importance provided.

Don't mention the probability of churning, or the machine learning model, or say anaything like "Based on the machine learning model's prediction and top 10m most importnat features, just explain the prediciton. 
"""
    print("EXPLAINATION PROMPT: ", prompt)
    raw_response = client.chat.completions.create(model="llama-3.2-3b-preview",
                                                  messages=[{
                                                      "role": "user",
                                                      "content": prompt
                                                  }])
    return raw_response.choices[0].message.content


def load_model(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)


def prepare_input(credit_score, location, gender, age, tenure, balance,
                  num_products, has_credit_card, is_active_member,
                  estimated_salary):
    input_dict = {
        'CreditScore': credit_score,
        'Age': age,
        'Tenure': tenure,
        'Balance': balance,
        'NumOfProducts': num_products,
        'HasCrCard': int(has_credit_card),
        'IsActiveMember': int(is_active_member),
        'EstimatedSalary': estimated_salary,
        'Geography_France': 1 if location == 'France' else 0,
        'Geography_Germany': 1 if location == 'Germany' else 0,
        'Geography_Spain': 1 if location == 'Spain' else 0,
        'Gender_Male': 1 if gender == 'Male' else 0,
        'Gender_Female': 1 if gender == 'Female' else 0,
    }

    input_df = pd.DataFrame([input_dict])
    return input_df, input_dict


def make_predictions(input_df, input_dict):
    probabilities = {
        'XGBoost': xgboost_model.predict_proba(input_df)[0][1],
        'Random Forest': random_forest_model.predict_proba(input_df)[0][1],
        'K-Nearest Neighbour': knn_model.predict_proba(input_df)[0][1],
    }
    avg_probability = np.mean(list(probabilities.values()))
    # st.markdown('### Model Probabilties')
    # for model, prob in probabilities.items():
    #     st.write(f'- {model}: {prob}')
    # st.write(f"Average Probability: {avg_probability}")
    col1, col2 = st.columns(2)
    with col1:
        fig = ut.create_gauge_chart(avg_probability)
        st.plotly_chart(fig, use_container_width=True)
        st.write(
            f"The customer has a {avg_probability:.2%} probability of churning."
        )
    with col2:
        fig_probs = ut.create_model_probability_chart(probabilities)
        st.plotly_chart(fig_probs, use_container_width=True)
    return avg_probability


def generate_email(probability, input_dict, explanation, surname):
    prompt = f"""You are a manager at National Bank of Europe. You are responsible for ensuring customer stay with the bank and are incentivized with various offers.
    
    You noticed a customer named {surname} has a {round(probability * 100,1)}% probability of churning. 
    
    Here is the customer's information:
    {input_dict}

    Here is some explanation as to why the customer might be at risk of churning: 
    {explanation}

    Generate an email to the customer based on their information, asking them to stay if they are at the risk of churning, or offering them incentives so that they beocme more loyal to the bank.

    Make sure to list out a set of incentives to stay based on their information, in bullet point format. Don't ever mention the probaility of churining, or the machine learning model to the customer. again format it well with bullets and grammer.
    Make sure the email is well formated, like the incentives are on a new bulleted line.
    the salutation fro the mail is also formal and easy to read. make it easier for the customer to understand and easy for manger to just copy the email  \n.
    
"""
    raw_response = client.chat.completions.create(model="llama-3.2-3b-preview",
                                                  messages=[{
                                                      "role": "user",
                                                      "content": prompt
                                                  }])
    print("\n\nEMAIL PROMPT: ", prompt)
    return raw_response.choices[0].message.content


xgboost_model = load_model('xgb_model.pkl')
naive_bayes_model = load_model('nb_model.pkl')
random_forest_model = load_model('rf_model.pkl')
decision_tree_model = load_model('dt_model.pkl')
svm_model = load_model('svm_model.pkl')
knn_model = load_model('knn_model.pkl')
# voting_classifiers_model = load_model('voting_classifier.pkl')
# xgboost_SMOTE_model = load_model('xgboost-SMOTE.pkl')
# xgboost_featureEngineered_model = load_model(
#     'xgboost-featureEngineered_model.pkl')

st.title("Customer Churn Prediction")
df = pd.read_csv("churn.csv")

customers = [
    f"{row['CustomerId']} - {row['Surname']}" for _, row in df.iterrows()
]

selected_customer_option = st.selectbox("Select a customer", customers)

if selected_customer_option:
    selected_customer_id = int(selected_customer_option.split(" - ")[0])
    selected_surname = selected_customer_option.split(" - ")[1]

    print("Selected Customer ID", selected_customer_id)
    print("Selected Surname", selected_surname)

    selected_customer = df.loc[df["CustomerId"] ==
                               selected_customer_id].iloc[0]
    print("Selected Customer", selected_customer)

    col1, col2 = st.columns(2)
    with col1:
        credit_score = st.number_input("Credit Score",
                                       min_value=300,
                                       max_value=850,
                                       value=int(
                                           selected_customer['CreditScore']))
        location = st.selectbox("Location", ["Spain", "France", "Germany"],
                                index=["Spain", "France", "Germany"
                                       ].index(selected_customer['Geography']))
        gender = st.radio(
            "Gender", ["Male", "Female"],
            index=0 if selected_customer['Gender'] == 'Male' else 1)
        age = st.number_input("Age",
                              min_value=0,
                              max_value=100,
                              value=int(selected_customer['Age']))
        tenure = st.number_input("Tenure (Years)",
                                 min_value=0,
                                 max_value=50,
                                 value=int(selected_customer['Tenure']))
        with col2:
            balance = st.number_input("Balance",
                                      min_value=0.0,
                                      value=float(
                                          selected_customer['Balance']))
            num_products = st.number_input(
                "Number of Products",
                min_value=1,
                max_value=10,
                value=int(selected_customer['NumOfProducts']))
            has_credit_card = st.checkbox("Has Credit Card",
                                          value=bool(
                                              selected_customer['HasCrCard']))
            is_active_member = st.checkbox(
                "Is Active Member",
                value=bool(selected_customer['IsActiveMember']))
            estimated_salary = st.number_input(
                "Estimated Salary",
                min_value=0.0,
                value=float(selected_customer['EstimatedSalary']))
    input_df, input_dict = prepare_input(credit_score, location, gender, age,
                                         tenure, balance, num_products,
                                         has_credit_card, is_active_member,
                                         estimated_salary)
    avg_probaility = make_predictions(input_df, input_dict)

    explanation = explain_prediction(avg_probaility, input_dict,
                                     selected_customer['Surname'])

    percentiles = calculate_percentiles(selected_customer, df)

    # Display the percentiles in the Streamlit app
    st.subheader("Customer Percentiles")
    # for key, value in percentiles.items():
    #     st.write(f"{key}: {value:.2f}th percentile")
    per_chart = ut.plot_percentiles(percentiles)
    st.plotly_chart(per_chart, use_container_width=True)
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Prediction Results for Optimized Grid Search Model")
        gridSearch_Eval = load_model('gridSearch-eval.pkl')
        proba_grid = gridSearch_Eval.predict_proba(prepare_input_opt(input_df))
        fig = ut.create_gauge_chart(proba_grid[0][1])
        st.plotly_chart(fig, use_container_width=True)
        st.write(f"The customer has a {proba_grid[0][1]:.2%} probability of churning.")

    with col2:
        hyperameter_tuning_model = load_model('xgb_Hyper-eval.pkl')
        st.subheader("Prediction Results for Optimized Random Search Model")
        random = hyperameter_tuning_model.predict_proba(prepare_input_opt(input_df))
        fig = ut.create_gauge_chart(random[0][1])
        st.plotly_chart(fig, use_container_width=True)
        st.write(f"The customer has a {random[0][1]:.2%} probability of churning.")
    st.markdown("---")
    st.subheader("Explanation of Prediction")
    st.markdown(explanation)

    email = generate_email(avg_probaility, input_dict, explanation,
                           selected_customer['Surname'])
    st.markdown("---")
    st.subheader("Personalized Email")
    st.markdown(email)

    