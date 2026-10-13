
import streamlit as st
import tensorflow as tf
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder
import pandas as pd
import pickle

model=tf.keras.models.load_model('model_1.h5')

with open('label_encoder_gender.pkl','rb') as file:
    label_encoder_gender=pickle.load(file)
with open('ohe_geo_encoder.pkl','rb') as file:
    ohe_geo_encoder=pickle.load(file)
with open('scaler.pkl','rb') as file:
    scaler=pickle.load(file)

st.title('Customer churn report')

geography=st.selectbox('Geography',ohe_geo_encoder.categories_[0])
gender=st.selectbox('Gender',label_encoder_gender.classes_)
age=st.slider('Age',18,100)
balance=st.number_input('Balance')
credit_score=st.number_input('Credit Score')
estimated_salary=st.number_input('Estimated Salary')
tenure=st.slider('Tenure',0,10)
num_of_pro=st.slider('Number of Products',1,4)
has_cr_card=st.selectbox('Has Credit Card',[0,1])
active=st.selectbox('Is Active',[0,1])

input_data={
    'CreditScore':[credit_score],
    'Gender':[label_encoder_gender.transform([gender])[0]],
    'Age':[age],
    'Tenure':[tenure],
    'Balance':[balance],
    'NumOfProducts':[num_of_pro],
    'HasCrCard':[has_cr_card],
    'IsActiveMember':[active],
    'EstimatedSalary':[estimated_salary]
}

geo_encoded=ohe_geo_encoder.transform([[geography]]).toarray()
geo_encoded_df=pd.DataFrame(geo_encoded,columns=ohe_geo_encoder.get_feature_names_out(['Geography']))

input_data=pd.DataFrame(input_data)
input_data=pd.concat([input_data,geo_encoded_df],axis=1)

input_data_scaled=scaler.transform(input_data)

prediction=model.predict(input_data_scaled)
prob=prediction[0][0]

st.write(f"Churn probability:{prob:.2%}")

if prob>=0.5:
    st.write('Customer is likely to churn.')
else:
    st.write('Customer is unlikely to churn.')
