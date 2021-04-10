""" streamlit_app.py """

import pandas as pd
import streamlit as st
from sidebar_value import Sidebar_value
from data_get import Data_Get
from preprocessing import Preprocessing
from create_train_test import Create_Train_Test
from basic_ml import Trial_ML
import optuna_ml as om

st.title('Titanic - Machine Learning from Disaster')

st.sidebar.subheader('Parameter')
bins = Sidebar_value().bins_value()
solution = Sidebar_value().solution_value()
test_size = Sidebar_value().test_size_value()
file_output = Sidebar_value().file_output_value()
output_name = Sidebar_value().file_output_name()
analysis = Sidebar_value().analysis_value()


st.subheader('(1) Data Structure')
alldata, test_raw = Data_Get().read_data()


st.subheader('(2) Missing Value')
st.write('Please coding your method for missing value')
alldata = Data_Get().missing_value(alldata)


st.subheader('(3) Preprocessing')
st.write('Please coding your method for preprocessing')
alldata_sum = alldata.copy()

alldata_sum = Preprocessing().name_process(alldata_sum)
alldata_sum = Preprocessing().fare_processing(alldata_sum, bins)
alldata_sum = Preprocessing().age_processing(alldata_sum, solution)
alldata_sum = Preprocessing().family_processing(alldata_sum)
alldata_sum = Preprocessing().cabin_processing(alldata_sum)
alldata_sum = Preprocessing().ticket_processing(alldata_sum)


st.subheader('(4) Create Train data')
alldata_sum = Create_Train_Test().label_encoder(alldata_sum)
alldata_sum = Create_Train_Test().dummy_df(alldata_sum)
train_feature, train_tagert, test_feature = Create_Train_Test().create_train_test(alldata_sum)

if test_raw.shape[0] != test_feature.shape[0]:
    st.warning("Bad status of test_feature.Please check your preprocessing")
X_train, X_test, y_train, y_test = Create_Train_Test().split_train_test(train_feature, train_tagert, test_size)

if analysis:
    st.subheader('(5) Machine Learning(Basic)')
    rfc, xgb, lgb, lr, svc = Trial_ML().base_ml(X_train, X_test, y_train, y_test)

    if file_output == 'Yes':
        Trial_ML().output_file(output_name, rfc, xgb, lgb, lr, svc)
