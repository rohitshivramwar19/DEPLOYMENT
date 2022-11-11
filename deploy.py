# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 16:48:59 2022

@author: ROHIT
"""

import pandas as pd 
import streamlit as st
from sklearn.linear_model import LogisticRegression

st.title('model deployment: logic regression')

st.sidebar.header('user input parameter')

def user_input_features():
    CLMSEX = st.sidebar.selectbox('gender',('1','0'))
    CLMINSUR = st.sidebar.selectbox('insurance',('1','0'))
    SEATBELT = st.sidebar.selectbox('seatbelt',('1','0'))
    CLMAGE = st.sidebar.number_input('insert age')
    LOSS = st.sidebar.number_input('insert loss')
    data={'CLMSEX':CLMSEX,
          'CLMINSUR':CLMINSUR,
          'SEATBELT':SEATBELT,
          'CLMAGE':CLMAGE,
          'LOSS':LOSS}
    features = pd.DataFrame(data,index = [0])
    return features

df= user_input_features()
st.subheader('user input parameter')
st.write(df)


claimants = pd.read_csv('claimants.csv')
claimants.drop(["CASENUM"],inplace=True,axis = 1)
claimants = claimants.dropna()

x = claimants.iloc[:,[1,2,3,4,5]]
y = claimants.iloc[:,0]
clf = LogisticRegression()
clf.fit(x,y)

prediction = clf.predict(df)
prediction_proba = clf.predict_proba(df)

st.subheader('Predicted Result')
st.write('Yes' if prediction_proba[0][1]>0.5 else 'No')

st.subheader('Prediction Probability')

st.write(prediction_proba)
st.write(prediction)