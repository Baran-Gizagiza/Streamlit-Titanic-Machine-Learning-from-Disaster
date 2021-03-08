import main
import pandas as pd
import seaborn as sns; sns.set(font='DejaVu Sans')
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.svm import SVC

file_name = 'basic_rev001'

class Trial_ML():
    def rfc(slef):
        X_train, y_train = main.X_train, main.y_train
        rfc = RandomForestClassifier(random_state=0)
        rfc.fit(X_train, y_train)
        # print('RandomForestClassifier')
        # print('accuracy of train set: {}'.format(rfc.score(X_train, y_train)))
        # print('accuracy of test set: {}'.format(rfc.score(X_test, y_test)))
        return rfc


    def xgb(slef):
        X_train, y_train = main.X_train, main.y_train
        xgb = XGBClassifier(random_state=0, use_label_encoder =False)
        xgb.fit(X_train, y_train)
        # print('XGBClassifier')
        # print('accuracy of train set: {}'.format(xgb.score(X_train, y_train)))
        # print('accuracy of test set: {}'.format(xgb.score(X_test, y_test)))
        return xgb


    def lgb(slef):
        X_train, y_train = main.X_train, main.y_train
        lgb = LGBMClassifier(random_state=0)
        lgb.fit(X_train, y_train)
        # print('LGBMClassifier')
        # print('accuracy of train set: {}'.format(lgb.score(X_train, y_train)))
        # print('accuracy of test set: {}'.format(lgb.score(X_test, y_test)))
        return lgb


    def lr(slef):
        X_train, y_train = main.X_train, main.y_train
        lr = LogisticRegression(random_state=0)
        lr.fit(X_train, y_train)
        # print('LogisticRegression')
        # print('accuracy of train set: {}'.format(lr.score(X_train, y_train)))
        # print('accuracy of test set: {}'.format(lr.score(X_test, y_test)))
        return lr


    def svc(slef):
        X_train, y_train = main.X_train, main.y_train
        svc = SVC(random_state=0)
        svc.fit(X_train, y_train)
        # print('SVC')
        # print('accuracy of train set: {}'.format(svc.score(X_train, y_train)))
        # print('accuracy of test set: {}'.format(svc.score(X_test, y_test)))
        return svc


    def base_ml(self):
        X_train, X_test, y_train, y_test = main.X_train, main.X_test, main.y_train, main.y_test

        rfc = self.rfc()
        xgb = self.xgb()
        lgb = self.lgb()
        lr = self.lr()
        svc = self.svc()

        st.success('Finishied Machine learning')

        dic = {'train': [rfc.score(X_train, y_train), xgb.score(X_train, y_train), \
                lgb.score(X_train, y_train), lr.score(X_train, y_train), svc.score(X_train, y_train)],
                'test': [rfc.score(X_test, y_test), xgb.score(X_test, y_test), \
                lgb.score(X_test, y_test), lr.score(X_test, y_test), svc.score(X_test, y_test)]}
        index=['rfc', 'xgb', 'lgb', 'lr', 'svc']

        df_cls = pd.DataFrame(dic, index=index)
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        sns.barplot(data=df_cls, x=index, y='train', ax = axes[0])
        axes[0].set_ylim([0.7, 1])
        axes[0].set_xlabel('Classifier')
        axes[0].set_ylabel('Accuracy')
        axes[0].set_title('Train')
        sns.barplot(data=df_cls, x=index, y='test', ax = axes[1])
        axes[1].set_ylim([0.7, 1])
        axes[1].set_xlabel('Classifier')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Test')
        fig.tight_layout()
        st.pyplot(fig)

        return rfc, xgb, lgb, lr, svc

    def output_file(self, *arg):
        test_feature = main.test_feature
        test_raw = main.test_raw
        pred_1 = {
            'rfc': rfc.predict(test_feature),
            'xgb': xgb.predict(test_feature),
            'lgb': lgb.predict(test_feature),
            'lr': lr.predict(test_feature),
            'svc': svc.predict(test_feature)
        }
        # ファイル出力
        for key, value in pred_1.items():
            pd.concat(
                [
                    pd.DataFrame(test_raw['PassengerId']).reset_index(drop=True),
                    pd.DataFrame(value, columns=['Survived'])
                ],
                axis=1
            ).to_csv('../output/submittion_{0}_{1}.csv'.format(key, file_name), index=False)
