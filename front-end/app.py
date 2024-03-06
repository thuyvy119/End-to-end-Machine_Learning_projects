import streamlit as st
import pickle
import numpy as np
import requests

# define API URL endpoint
Api_url = "http://127.0.0.1:8000/predict"

# function to make prediction using the API
def prediction(data):
    response = requests.post(Api_url, json= data)
    if response.status_code == 200:
        result= response.json()
        return result["prediction"], result["probability"]
    else:
        st.error("Sorry! An error occurred while making decision.")

# create web app
def main():
    # set app title and description
    st.title("Stroke Prediction")
    st.write("Please provide information to predict the likelihood of stroke.")    

    # create input fields for users typing in 
    gender = st.selectbox("Biological sex", ("Male", "Female"))
    age = st.number_input("Age", min_value = 1, max_value = 120, value = 40)
    hypertension = st.selectbox("Hypertension", ("Yes", "No"))
    heart_disease = st.selectbox("Heart disease", ("Yes", "No"))
    ever_married = st.selectbox("Do you get married?", ("Yes", "No"))
    work_type = st.selectbox("Work type", ("Private", "Self-employed", "Children", "Government job", "Never worked"))
    Residence_type = st.selectbox("Residence type", ("Urban", "Rural"))
    avg_glucose_level = st.number_input("Average glucose level", min_value = 0.0, value = 80.0)
    bmi = st.number_input("BMI", min_value = 0.0, value = 20.0)    
    smoking_status = st.selectbox("Smoking status", ("Never smoked", "Unknown", "Formerly smoked", "Smokes"))

    # convert categorical inputs to numerical data
    age = int(age)
    avg_glucose_level= float(avg_glucose_level)
    bmi = float(bmi)
    gender = 1 if gender == "Male" else 0
    hypertension = 1 if hypertension == "Yes" else 0
    heart_disease = 1 if heart_disease == "Yes" else 0
    ever_married = 1 if ever_married == "Yes" else 0
    Residence_type = 1 if Residence_type == "Urban" else 0

    # map work_type to numerical values
    work_type_map = {
        "Government job": 0,
        "Never worked": 1,
        "Private": 2,
        "Self-employed": 3,
        "Children": 4
    }
    work_type = work_type_map[work_type]

    # map smoking status to numerical values
    smoking_status_map = {
        "Unknown": 0,
        "Formerly smoked": 1,
        "Never smoked": 2,
        "Smokes": 3
    }
    smoking_status = smoking_status_map[smoking_status]

    # create prediction button
    if st.button("Predict Stroke"):
        # collect input features
        input_data = [gender, age, hypertension,
                    heart_disease, ever_married,
                    work_type, Residence_type, 
                    avg_glucose_level, bmi, smoking_status]

        # predict stroke and probability
        prediction, probability = prediction(input_data)

        # display result
        if prediction == 0:
            st.write("Congratulations! You do not have a risk of stroke.")
        else:
            st.write("You may be at risk of stroke with the probability of: ", probability)
            st.write("You should go to the nearest clinic for a further check-up.")

#run web app
if __name__ == "__main__":
    main()
