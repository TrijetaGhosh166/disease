import streamlit as st
import pickle
import numpy as np
import pandas as pd
df=pd.read_csv("Symptom-severity.csv")
def symptoms_weight(df):
    dis={}
    for a in df[['Symptom','weight']].values:
        dis[a[0]]=a[1]
    return dis


dis=symptoms_weight(df)

    
    

# Load the saved model
@st.cache_resource
def load_model():
    with open('knn.sav', 'rb') as file:
        model = pickle.load(file)
    return model

model = load_model()

# Streamlit App
st.title("Disease Prediction System")
st.write("Enter the following details:")

# Creating seven text input fields
input1  = st.text_input("Symptom 1")

input2 = st.text_input("Symptom 2")

input3 = st.text_input("Symptom 3")

input4 = st.text_input("Symptom 4")

input5 = st.text_input("Symptom 5")

input6 = st.text_input("Symptom 6")




# Collect inputs

# Predict button
if st.button("Predict"):
    
    input1=dis[input1]
    input2=dis[input2]
    input3=dis[input3]
    input4=dis[input4]
    input5=dis[input5]
    input6=dis[input6]

    inputs = [input1, input2, input3, input4, input5, input6]
    # Ensure all inputs are filled
    if all(inputs):
        # You may need to encode or preprocess inputs here
        # For demonstration, we assume inputs are numerical
        try:
           # inputs = [float(i) for i in inputs]
            inputs=np.array(inputs).reshape(1,-1)
            # Reshape the input to match the model's expected format
            prediction = model.predict(inputs)
            st.success(f"Predicted Output: {prediction[0]}")
        except ValueError:
            st.error("Please enter valid numerical values for all symptoms.")
    else:
        st.warning("Please fill in all symptom fields before predicting.")
