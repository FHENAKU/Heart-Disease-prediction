"""
Created on Tue Nov 17 21:40:41 2020

@author: win10
"""

# 1. Library imports
import uvicorn
from fastapi import FastAPI
from HeartDiseases import HeartDisease
import numpy as np
import pickle
import pandas as pd
# 2. Create the app object
app = FastAPI()
pickle_in = open("heart_ds_log_regr_model.pkl","rb")
classifier=pickle.load(pickle_in)

# 3. Index route, opens automatically on http://127.0.0.1:8000
@app.get('/')
def index():
    return {'message': 'Hello, World'}

# 4. Route with a single parameter, returns the parameter within a message
#    Located at: http://127.0.0.1:8000/AnyNameHere
@app.get('/{name}')
def get_name(name: str):
    return {'Welcome To Heart Disease App': f'{name}'}

# 3. Expose the prediction functionality, make a prediction from the passed
#    JSON data and return the predicted Heart disease with the confidence
@app.post('/predict')
def predict_heartDisease(data:HeartDisease):
        data = data.dict()
        age=data['age']
        sex=data['sex'] 
        cp=data['cp']
        trestbps=data['trestbps']
        chol=data['chol']
        fbs=data['fbs']
        restecg=data['restecg']
        thalach=data['thalach']
        exang=data['exang']
        oldpeak=data['oldpeak']
        slope=data['slope']
        ca=data['ca']
        thal=data['thal']
    
       # print(classifier.predict([[age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]]))
        prediction = classifier.predict([[age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]])
        if(prediction[0]>0.5):
            prediction="Heart disease"
        else:
             prediction="No Heart Disease"
             
        return {'prediction': prediction}
    
    # 5. Run the API with uvicorn
#    Will run on http://127.0.0.1:8000
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
    
#uvicorn app:app --reload