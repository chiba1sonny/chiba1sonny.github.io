---
layout: post
title: Streamlitを使って簡単なアプリを作る
image: /img/sl.jpg
tags: ["deploy","machine learning"]
---

# A Streamlit demo for PGV prediction
Created a demo for PGV prediction. PGV means peak ground velocity and it is an earthquake intensity. PGV is widely used in both deterministic and probabilistic seismic hazard analyses. 

PGV depends on source characteristics (e.g., earthquake magnitude), the propagation path (e.g., the shortest distance from the fault), and the local site conditions. Basiclly, PGV had been obtained by empirical equation until the machine learning and big data booming. As the development of statistical analysis methods and more ground motion records are obtained, the research of attenuation relationship of PGV has been greatly developed.

So in this repo, firstly, a XGBoost model for PGV prediction is constructed and the model is saved as json data. Then created a simple application for PGV prediction using Streamlit.

## 1.Requirements
- xgboost
- pandas
- matplotlib
- seaborn
- sklearn
- pathlib
- streamlit

## 2.Data
This repo used ground motion data obtained by K-NET (Kyoshin network) and KiK-net (Kiban Kyoshin network), which are strong-motion seismograph networks constructed by the National Research Institute for Earth Science and Disaster prevention (NIED), Japan. 6,944 ground motion records at 1,184 K-NET and KiK-net seismic observation stations which were observed during the 32 earthquakes are employed.
- PGV is used as the objective variable
- the moment magnitude (Mw), the shortest distance from the fault (Distance), the earthquake source depth (Depth) are used as the explanatory variables

## 3.Model
As a simple example, I used XGBoost Regressor to deal with the data and obtained the model, which is available in ml.py.

## 4.App
We have our model and we can use it to predict the PGV (I have attached the model as model.json). Then we can build a simple app using streamlit. It is very easy and readble. Have a look at main.py. After building it, just run main.py to get the app in action like below.

![Screenshot from 2022-10-26 18-03-24](https://user-images.githubusercontent.com/68838083/197983860-d89e74eb-409e-44fc-beb9-a24432cda24e.png)

## 5.Code

#### Constructing xgb model for my custon data

```py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics

data = pd.read_csv("Pgv.csv")
data.head(3)

column_sels=['Mw','Distance','Depth']
X=data.loc[:,column_sels]
y=data['logpgv']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X,y, test_size = 0.2, random_state = 4)

print(X_train.shape,
X_test.shape,
y_train.shape,
y_test.shape)

from xgboost import XGBRegressor

reg = XGBRegressor()

reg.fit(X_train, y_train)
reg.save_model('model.json')
```

#### Creating a simple app using Streamlit

```py
import xgboost as xgb
import streamlit as st
import pandas as pd
from pathlib import Path


model = xgb.XGBRegressor()
model.load_model('model.json')
@st.cache

def predict(Mw, Distance, Depth):
    prediction = model.predict(pd.DataFrame([[Mw, Distance, Depth]], columns=['Mw', 'Distance', 'Depth']))
    return prediction

def read_markdown_file(markdown_file):
    return Path(markdown_file).read_text()
intro_markdown = read_markdown_file("introduction.md")
st.markdown(intro_markdown, unsafe_allow_html=True)

st.write('Mw, Distance, Depthを入力してください')

Mw = st.sidebar.number_input('Mw:', min_value=1, max_value=12, value=6)
Distance = st.sidebar.number_input('Distance:', min_value=1, max_value=500, value=50)
Depth = st.sidebar.number_input('Depth:', min_value=1, max_value=150, value=10)

if st.button('log(pgv)='):
    logpgv = predict(Mw, Distance, Depth)
    st.success(f'{logpgv[0]:.2f}')

if st.button('pgv='):
    logpgv = predict(Mw, Distance, Depth)
    pgv = 10**logpgv
    st.success(f'{pgv[0]:.2f}')
```

#### [Here](https://github.com/chiba1sonny/PGV-Prediction-Streamlit-demo/blob/main/model.json) is the model.
