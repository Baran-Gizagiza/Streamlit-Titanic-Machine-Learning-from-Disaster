""" preprocessing.py """
import pandas as pd
import numpy as np
import streamlit as st
import seaborn as sns; sns.set(font='DejaVu Sans')
import matplotlib.pyplot as plt

# @st.cache
class Preprocessing():
    def name_process(self, alldata_sum):
        st.write("(3)-1. Name")
        alldata_sum['honor'] = alldata_sum['Name'].apply(lambda x: x.split(', ')[1].split('. ')[0])

        # "honor"と"train or test"のクロステーブル
        # st.write(pd.crosstab(alldata_sum['honor'], alldata_sum['train_or_test']))

        alldata_sum = alldata_sum.query("honor not in ['Capt','Don','Jonkheer','Lady','Major','Mile','the Countess']").reset_index(drop=True)

        alldata_sum['honor'].replace(['Col','Dr', 'Rev'], 'Rare',inplace=True) #少数派の敬称を統合
        alldata_sum['honor'].replace(['Mlle','Ms'], 'Miss',inplace=True) #Missに統合
        alldata_sum['honor'].replace(['Mme','Sir'], 'Mr',inplace=True) #Missに統合

        st.write(alldata_sum[alldata_sum['train_or_test'] == 0]['Survived'].groupby(alldata_sum['honor']).agg(['mean','count']))

        return alldata_sum

    def fare_processing(self, alldata_sum, bins):
        st.write("(3)-2. Fare")
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        axes[0].set_title('Common scale')
        sns.distplot(alldata_sum[alldata_sum['Survived']==1]['Fare'],kde=False,rug=False,bins=bins,label='Survived', ax=axes[0])
        sns.distplot(alldata_sum[alldata_sum['Survived']==0]['Fare'],kde=False,rug=False,bins=bins,label='Death', ax=axes[0])
        axes[0].set_ylabel('Count')
        axes[1].set_title('Log scale')
        sns.distplot(np.log1p(alldata_sum[alldata_sum['Survived']==1]['Fare']),kde=False,rug=False,bins=bins,label='Survived', ax=axes[1])
        sns.distplot(np.log1p(alldata_sum[alldata_sum['Survived']==0]['Fare']),kde=False,rug=False,bins=bins,label='Death', ax=axes[1])
        axes[1].set_ylabel('Count')
        axes[0].legend()
        axes[1].legend()
        fig.tight_layout()
        st.pyplot(fig)
        # Fareの分割
        alldata_sum.loc[:, 'Fare_bin'] = pd.qcut(alldata_sum['Fare'], bins)

        return alldata_sum


    def age_processing(self, alldata_sum, solution):
        st.write("(3)-3. Age")

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].set_title('Before')
        sns.distplot(alldata_sum[alldata_sum['Survived']==1]['Age'],kde=True,rug=False,bins=10,label='Survived', ax=axes[0])
        sns.distplot(alldata_sum[alldata_sum['Survived']==0]['Age'],kde=True,rug=False,bins=10,label='Death', ax=axes[0])
        axes[0].set_ylabel('Count')

        if solution == "Exclude":
            # Ageの欠損を除外
            alldata_sum = alldata_sum[~((alldata_sum['train_or_test'] == 0) & (alldata_sum['Age'].isnull()))]
        elif solution == "Mean+STD":
            # Ageの欠損を平均値±標準偏差で代用
            age_avg = alldata_sum['Age'].mean()
            age_std = alldata_sum['Age'].std()
            age_null_count = alldata_sum['Age'].isnull().sum()
            age_null_random_list = np.random.randint(age_avg - 2.0 * age_std, age_avg + 2.0 * age_std, size=age_null_count)
            alldata_sum = alldata_sum.copy()
            alldata_sum['Age'][np.isnan(alldata_sum['Age'])] = age_null_random_list
            alldata_sum['Age'] = alldata_sum['Age'].astype(int)
        elif solution == "Mean":
            age_avg = alldata_sum['Age'].mean()
            alldata_sum['Age'] = alldata_sum['Age'].fillna(age_avg)
        elif solution == "Median":
            age_median = alldata_sum['Age'].median()
            alldata_sum['Age'] = alldata_sum['Age'].fillna(age_median)

        axes[1].set_title('After')
        sns.distplot(alldata_sum[alldata_sum['Survived']==1]['Age'],kde=True,rug=False,bins=10,label='Survived', ax=axes[1])
        sns.distplot(alldata_sum[alldata_sum['Survived']==0]['Age'],kde=True,rug=False,bins=10,label='Death', ax=axes[1])
        axes[1].set_ylabel('Count')
        axes[0].legend()
        axes[1].legend()
        fig.tight_layout()
        st.pyplot(fig)

        return alldata_sum


    def family_processing(self, alldata_sum):
        st.write("(3)-4. FamilySize(Parch+Sibsp+1)")
        # 家族数 = Parch + SibSp + 1
        alldata_sum['FamilySize'] = alldata_sum['Parch'] + alldata_sum['SibSp'] + 1

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].set_title('Before')
        sns.countplot(alldata_sum[alldata_sum['train_or_test'] == 0]['FamilySize'],hue=alldata_sum['Survived'], ax=axes[0])
        axes[0].set_ylabel('Count')

        # FamilySizeを離散化
        alldata_sum['FS_bin'] = 'big'
        alldata_sum.loc[alldata_sum['FamilySize']==1,'FS_bin'] = 'alone'
        alldata_sum.loc[(alldata_sum['FamilySize']>=2) & (alldata_sum['FamilySize']<=4),'FS_bin'] = 'small'
        alldata_sum.loc[(alldata_sum['FamilySize']>=5) & (alldata_sum['FamilySize']<=7),'FS_bin'] = 'mediam'

        axes[1].set_title('After')
        sns.countplot(alldata_sum[alldata_sum['train_or_test'] == 0]['FS_bin'],hue=alldata_sum['Survived'], ax=axes[1])
        axes[1].set_ylabel('Count')
        axes[0].legend()
        axes[1].legend()
        fig.tight_layout()
        st.pyplot(fig)

        return alldata_sum


    def cabin_processing(self, alldata_sum):
        st.write("(3)-5. Cabin")

        alldata_sum['Cabin_init'] = alldata_sum['Cabin'].apply(lambda x: str(x)[0])

        with st.beta_container():
            col1, col2, col3 = st.beta_columns([1, 1, 1])
        with col1:
            st.write("Mean vs count")
            # Cabinの頭文字別の生存率とレコード数
            st.write(alldata_sum[alldata_sum['train_or_test'] == 0]['Survived'].groupby(alldata_sum['Cabin_init']).agg(['mean','count']))

        with col2:
            st.write("Before")
            # Cabinの頭文字別の生存率とレコード数
            st.write(pd.crosstab(alldata_sum['Cabin_init'],alldata_sum['train_or_test']))

        #少数派のCabin_initを統合
        alldata_sum['Cabin_init'].replace(['G','T'], 'Rare',inplace=True)
        with col3:
            st.write("After")
            # Cabinの頭文字別の生存率とレコード数
            st.write(pd.crosstab(alldata_sum['Cabin_init'],alldata_sum['train_or_test']))

        return alldata_sum


    def ticket_processing(self, alldata_sum):
        st.write("(3)-6. Ticket")

        # Ticket頻度別の生存率
        alldata_sum.loc[:, 'TicketFreq'] = alldata_sum.groupby(['Ticket'])['PassengerId'].transform('count')
        st.write(alldata_sum[alldata_sum['train_or_test'] == 0].groupby(['TicketFreq'])['Survived'].agg(['mean','count']).reset_index())

        return alldata_sum
