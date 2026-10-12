import streamlit as st
import tensorflow as tf
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder
import pandas as pd
import pickle

model=tf.keras.load_model('model.h5')

with open('label_encoder_gender.pkl','rb') as file:
    label_encoder_gender=pickle.load(file)
with open('ohe_geo_encoder.pkl','rb') as file:
    ohe_geo_encoder=pickle.load(file)
with open('scaler.pkl','rb') as file:
    scaler=pickle.load(file)

##Streamlit app
st.title('Customer churn report')

