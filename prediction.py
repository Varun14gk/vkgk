import pickle
import joblib

with open('rf_model.pkl' , 'rb') as f:
    model = joblib.load(f)   

with open('rf_model.pkl' , 'rb') as f:
    model = pickle.load(f)   


def predict_(data, model = model):   
    return model.predict(data)
