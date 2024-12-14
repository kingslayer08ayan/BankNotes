import uvicorn
from fastapi import FastAPI
from Banknotes import BankNote
import numpy as np
import pandas as pd
import pickle
app=FastAPI()
pickle_in=open('classifier.pkl','rb')
classifier=pickle.load(pickle_in)

@app.get('/')
def index():
    return {'message': 'Hello, stranger'}

@app.get('/{name}')
def get_name(name:str):
    return {'message': f'Hello, {name}'}

@app.post('/predict')
def predict_type(data:BankNote):
    data=data.model_dump()
    variance=data['variance']
    skewness=data['skewness']
    curtosis=data['curtosis']
    entropy=data['entropy']
    prediction=classifier.predict([[variance,skewness,curtosis,entropy]])
    if(prediction[0]>0.5):
        prediction="Fake note"
    else:
        prediction="Real note"
    return {
        'prediction':prediction
    }

#will run on http://127.0.0.1:8000

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1',port=8000)
#uvicorn app:app --reload



