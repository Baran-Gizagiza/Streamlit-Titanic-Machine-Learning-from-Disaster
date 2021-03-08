""" main.py """
import pandas as pd
import streamlit as st
import sidebar_value
import data_get as dg
import preprocessing as pr
import create_train_test as ct
import basic_ml as ml
import optuna_ml as om

st.title('Titanic - Machine Learning from Disaster')

sidebar_Inst = sidebar_value.Sidebar_value()
bins = sidebar_Inst.bins_value()
solution = sidebar_Inst.solution_value()
test_size = sidebar_Inst.test_size_value()
file_output = sidebar_Inst.file_output_value()
analysis = sidebar_Inst.analysis_value()

if analysis:
    st.subheader('(1) Data Structure')
    dg_Inst = dg.Data_Get()
    alldata, test_raw = dg_Inst.read_data()


    st.subheader('(2) Missing Value')
    st.write('Please coding your method for missing value')
    alldata = dg_Inst.missing_value(alldata)


    st.subheader('(3) Preprocessing')
    st.write('Please coding your method for preprocessing')
    alldata_sum = alldata.copy()

    pp_Inst = pr.Preprocessing()
    alldata_sum = pp_Inst.name_process(alldata_sum)
    alldata_sum = pp_Inst.fare_processing(alldata_sum, bins)
    alldata_sum = pp_Inst.age_processing(alldata_sum, solution)
    alldata_sum = pp_Inst.family_processing(alldata_sum)
    alldata_sum = pp_Inst.cabin_processing(alldata_sum)
    alldata_sum = pp_Inst.ticket_processing(alldata_sum)


    st.subheader('(4) Create Train data')
    ct_Inst = ct.Create_Train_Test()
    alldata_sum = ct_Inst.label_encoder(alldata_sum)
    alldata_sum = ct_Inst.dummy_df(alldata_sum)
    train_feature, train_tagert, test_feature = ct_Inst.create_train_test(alldata_sum)


    if test_raw.shape[0] != test_feature.shape[0]:
        st.warning("Bad status of test_feature.Please check your preprocessing")
    X_train, X_test, y_train, y_test = ct_Inst.split_train_test(train_feature, train_tagert, test_size)


    st.subheader('(5) Machine Learning(Basic)')
    ml_Inst = ml.Trial_ML()
    rfc, xgb, lgb, lr, svc = ml_Inst.base_ml()

    if file_output == 'Yes':
        ml_Inst.output_file(rfc, xgb, lgb, lr, svc)


    # st.subheader('(6) Machine Learning(Optuna)')

    # rfc_best, xgb_best, lgb_best, lr_best, svc_best = om.optuna_ml()

    # if file_output == 'Yes':
    #     om.output_file(rfc_best, xgb_best, lgb_best, lr_best, svc_best)
