""" create_train_test.py """
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

#@st.cache
class Create_Train_Test():
    def label_encoder(self, alldata_sum):
        # カテゴリ特徴量についてlabel encoding
        target_col = ['Sex', 'Fare_bin']
        label = LabelEncoder()
        for col in target_col:
            alldata_sum.loc[:, col] = label.fit_transform(alldata_sum[col])

        return alldata_sum


    def dummy_df(self, alldata_sum):
        # カテゴリカル変数
        cat_col = ['Pclass', 'Embarked','honor','Cabin_init', 'FS_bin', 'Fare_bin']
        alldata_sum['Pclass'] = alldata_sum['Pclass'].astype('str')
        alldata_sum = pd.get_dummies(alldata_sum, drop_first=True, columns=cat_col)
        return alldata_sum

    def create_train_test(self, alldata_sum):
        # ターゲット変数と、学習に不要なカラムを定義
        target_col = 'Survived'
        drop_col = ['PassengerId','Survived', 'Name', 'Fare', 'Ticket', 'Cabin', 'train_or_test',  'Parch', 'FamilySize', 'SibSp']

        # 最初に統合したtrainとtestを分離
        train = alldata_sum[alldata_sum['train_or_test'] == 0]
        test = alldata_sum[alldata_sum['train_or_test'] == 1]

        # 学習に必要な特徴量のみを保持
        train_feature = train.drop(columns=drop_col)
        test_feature = test.drop(columns=drop_col)
        train_tagert = train[target_col]
        st.write('The size of the train data:' + str(train_feature.shape))
        st.write('The size of the test data:' + str(test_feature.shape))

        if train_feature.shape[1] != test_feature.shape[1]:
            st.warning("Note: Bad status of feature. Please check your preprocessing")

        with st.beta_container():
            col1, col2 = st.beta_columns([1, 1])
        with col1:
            st.write("Before ")
            st.write(alldata_sum.dtypes)
        with col2:
            st.write("After")
            st.write(train_feature.dtypes)

        return train_feature, train_tagert, test_feature

    def split_train_test(self, train_feature, train_tagert, test_size):
        # trainデータを分割
        X_train, X_test, y_train, y_test = train_test_split(
            train_feature, train_tagert, test_size=test_size, random_state=0, stratify=train_tagert)

        return X_train, X_test, y_train, y_test
