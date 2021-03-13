""" data_get.py """
import pandas as pd
import streamlit as st

@st.cache
class Data_Get():
    def read_data(self):
        train_raw = pd.read_csv('../data/train.csv') #学習データ
        test_raw = pd.read_csv('../data/test.csv') #テストデータ
        st.write('The size of the train data:' + str(train_raw.shape))
        st.write('The size of the test data:' + str(test_raw.shape))
        st.write('Target mean:' + str(round(train_raw["Survived"].mean(), 3)))

        df_train, df_test = train_raw.copy(), test_raw.copy()
        df_train['train_or_test'] = 0 #学習データフラグ
        df_test['train_or_test'] = 1 #テストデータフラグ
        df_test["Survived"] = 9 #テストにSalePriceカラムを仮置き
        alldata = pd.concat([df_train,df_test],sort=False,axis=0).reset_index(drop=True)

        checkbox = st.checkbox('Show data')
        if checkbox:
            checkbox_1 = st.checkbox('Train_data')
            if checkbox_1:
                st.write(df_train.describe(include='all'))
            checkbox_2 = st.checkbox('Test_data')
            if checkbox_2:
                st.write(df_test.describe(include='all'))
            checkbox_3 = st.checkbox('All_data')
            if checkbox_3:
                st.write(alldata.describe(include='all'))

        return alldata, test_raw


    def missing_value(self, alldata):
        with st.beta_container():
            col1, col2 = st.beta_columns([1, 1])
        with col1:
            st.subheader("Before")
            st.write(alldata.isnull().sum())
        # Embarkedには最頻値を代入
        alldata['Embarked'].fillna(alldata['Embarked'].mode()[0], inplace=True)
        # Fareには中央値を代入
        alldata['Fare'].fillna(alldata['Fare'].median(), inplace=True)
        with col2:
            st.subheader("After")
            st.write(alldata.isnull().sum())
        return alldata
