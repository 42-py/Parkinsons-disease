import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained model
model = joblib.load("parkinsons_model.pkl")

st.title("Parkinson's Disease Prediction App")

# Function to get user input
def user_input_features():
    fo = st.number_input('MDVP:Fo(Hz)')
    fhi = st.number_input('MDVP:Fhi(Hz)')
    flo = st.number_input('MDVP:Flo(Hz)')
    jitter_percent = st.number_input('MDVP:Jitter(%)')
    jitter_abs = st.number_input('MDVP:Jitter(Abs)')
    rap = st.number_input('MDVP:RAP')
    ppq = st.number_input('MDVP:PPQ')
    ddp = st.number_input('Jitter:DDP')
    shimmer = st.number_input('MDVP:Shimmer')
    shimmer_db = st.number_input('MDVP:Shimmer(dB)')
    apq3 = st.number_input('Shimmer:APQ3')
    apq5 = st.number_input('Shimmer:APQ5')
    apq = st.number_input('MDVP:APQ')
    dda = st.number_input('Shimmer:DDA')
    nhr = st.number_input('NHR')
    hnr = st.number_input('HNR')
    rpde = st.number_input('RPDE')
    dfa = st.number_input('DFA')
    spread1 = st.number_input('spread1')
    spread2 = st.number_input('spread2')
    d2 = st.number_input('D2')
    ppe = st.number_input('PPE')

    data = {
        'MDVP:Fo(Hz)': fo,
        'MDVP:Fhi(Hz)': fhi,
        'MDVP:Flo(Hz)': flo,
        'MDVP:Jitter(%)': jitter_percent,
        'MDVP:Jitter(Abs)': jitter_abs,
        'MDVP:RAP': rap,
        'MDVP:PPQ': ppq,
        'Jitter:DDP': ddp,
        'MDVP:Shimmer': shimmer,
        'MDVP:Shimmer(dB)': shimmer_db,
        'Shimmer:APQ3': apq3,
        'Shimmer:APQ5': apq5,
        'MDVP:APQ': apq,
        'Shimmer:DDA': dda,
        'NHR': nhr,
        'HNR': hnr,
        'RPDE': rpde,
        'DFA': dfa,
        'spread1': spread1,
        'spread2': spread2,
        'D2': d2,
        'PPE': ppe
    }

    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Predict
if st.button("Predict"):
    prediction = model.predict(input_df)
    if prediction[0] == 1:
        st.error("The person is likely to have Parkinson's disease.")
    else:
        st.success("The person is unlikely to have Parkinson's disease.")
