import pandas as pd
import numpy as np
import pickle  
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

# initiate app instance
app = FastAPI(title = "Stroke Prediction", version = '1.0',
            description = 'Stroke prediction based on Random Forest model')

# load model using pickle
with open("saved_model.pkl", "rb") as file:
    model = pickle.load(file)

# define data structures expected by the predict endpoint
class Data(BaseModel):
    gender: str
    age: int
    hypertension: str
    heart_disease: str
    ever_married: str
    work_type: str
    Residence_type: str
    avg_glucose_level: float
    bmi: float
    smoking_status: str

class PredictionOut(BaseModel):
    prediction: int
    probability: float

# API root/ home endpoint  
@app.get('/')
@app.get('/home')
def read_home():
    return {'message': 'System is ok'}

# predict endpoint to make prediction after receiving request from user
@app.post("/predict", response_model= PredictionOut)
def predict(data: Data):
    prediction, probability = model.predict(pd.DataFrame(columns= ['gender', 'age', 'hypertension',
                                        'heart_disease', 'ever_married', 'work_type', 
                                        'Residence_type', 'avg_glucose_level', 
                                        'bmi', 'smoking_status'],
                                        data= np.array([data.gender, data.age, data.hypertension,
                                                        data.heart_disease, data.ever_married,
                                                        data.work_type, data.work_type, 
                                                        data.Residence_type, data.avg_glucose_level,
                                                        data.bmi, data.smoking_status]).reshape(1, 10)))[0]
    return {"prediction": prediction, "probability": probability}

if __name__ == '__main__':
    uvicorn.run("main:app", host = "127.0.0.1", port = 8000, reload = True)
