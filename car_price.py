import streamlit as st
import pandas as pd
import pickle
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split


car_price=pickle.load(open('car_price.pkl','rb'))
predict=pd.DataFrame(car_price)

X=predict.drop(columns='Price')
Y=predict['Price']
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=4)

ohe=OneHotEncoder()
ohe.fit(X[['name','company','fuel_type']])
column_trans=make_column_transformer((OneHotEncoder(categories=ohe.categories_),['name','company','fuel_type']),remainder='passthrough')
leg=LinearRegression()

pipe=make_pipeline(column_trans,leg)
pipe.fit(X_train,Y_train)

def prediction():
    return pipe.predict(pd.DataFrame([[option2,option1,option3,option5,option4]],columns=['name','company','year','kms_driven','fuel_type']))[0]


st.title("Welcome To Car Price Pridictor")

st.markdown (""" <style>.stTextInput > label { font-size:120%; font-weight:bold; color:black; font-align:centre }.stSelectbox > label { font-size:120%; font-weight:bold; color:black; text-align:centre } </style> """, unsafe_allow_html=True)

option1 = st.selectbox(
    'Select Company',
    predict['company'].drop_duplicates().sort_values())

names=predict[predict['company']==option1]
option2 = st.selectbox(
    'Select Model',
    names['name'].drop_duplicates().sort_values())

option3 = st.selectbox(
    'Select Year',
    predict['year'].drop_duplicates().sort_values())

option4 = st.selectbox(
    'Select Fuel',
    predict['fuel_type'].drop_duplicates().sort_values())

option5 = st.text_input('Enter No. of Kilometers travelled',placeholder='enter kilometers')
#st.write(option5)
if st.button("Predict Price",type='primary',use_container_width=True):
    if(option5.isdigit()):
        st.header(prediction())
    else:
        st.error('Error : Enter Valid No. of Kilometers', icon="🚨")

st.dataframe(predict,use_container_width=True) 

#[theme]
#base="light"
#primaryColor="#1913c3"
#secondaryBackgroundColor="#ece4ec"
#textColor="#26282b"


