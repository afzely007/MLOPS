import streamlit as st
import pickle
import numpy as np
import os

# Load the saved Linear Regression model
with open('Gsalemodel.pkl', 'rb') as model_file:
    model = pickle.load(model_file)


# Function to predict GDP per capita using the loaded model
def predict_VGS(NA_Sales, EU_Sales, JP_Sales, Other_Sales):
    features = np.array([NA_Sales, EU_Sales, JP_Sales, Other_Sales])
    features = features.reshape(1,-1)
    VGS = model.predict(features)
    return VGS[0]

# Streamlit UI

st.title('Video Game Sales')
st.write("""  
Enter the values for the input features to predict video game sales.
""")
# Input fields for user
NA_Sales = st.number_input('NA_Sales')
EU_Sales = st.number_input('EU_Sales')
JP_Sales = st.number_input('JP_Sales')
Other_Sales = st.number_input('Other_Sales')                                          
# Prediction button
if st.button('Predict'):
    # Predict Sport
    VGS_prediction = predict_VGS(NA_Sales, EU_Sales, JP_Sales, Other_Sales)
    st.write(f"Predicted Value: {VGS_prediction}")