""" optuna_ml.py """
import optuna
import main
import pandas as pd
import streamlit as st
import warnings
import seaborn as sns; sns.set(font='DejaVu Sans')
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate

CV = 5
trials = 20
timeout = 60
file_name = 'optuna_rev001'


def objective_rfc(trial_rfc):
    param_grid_rfc = {
        "max_depth": trial_rfc.suggest_int("max_depth", 2, 10),
        "min_samples_leaf": trial_rfc.suggest_int("min_samples_leaf", 1, 10),
        'min_samples_split': trial_rfc.suggest_int("min_samples_split", 5, 20),
        "criterion": trial_rfc.suggest_categorical("criterion", ["gini", "entropy"]),
        'max_features': trial_rfc.suggest_int("max_features", 3, 10),
        "random_state": 0
    }
    X_train, y_train = main.X_train, main.y_train
    model = RandomForestClassifier(**param_grid_rfc)

    # n-Fold CV / Accuracy でモデルを評価する
    kf = StratifiedKFold(n_splits=CV, shuffle=True, random_state=42)
    scores = cross_validate(model, X=X_train, y=y_train, cv=kf)
    # 最小化なので 1.0 からスコアを引く
    return 1.0 - scores['test_score'].mean()


def objective_xgb(trial_xgb):
    param_grid_xgb = {
        'min_child_weight': trial_xgb.suggest_int("min_child_weight", 1, 5),
        'gamma': trial_xgb.suggest_discrete_uniform("gamma", 0.1, 1.0, 0.1),
        'subsample': trial_xgb.suggest_discrete_uniform("subsample", 0.5, 2.0, 0.1),
        'colsample_bytree': trial_xgb.suggest_discrete_uniform("colsample_bytree", 0.5, 1.0, 0.1),
        'max_depth': trial_xgb.suggest_int("max_depth", 5, 15),
        "random_state": 0
    }
    X_train, y_train = main.X_train, main.y_train
    model = XGBClassifier(**param_grid_xgb)

    kf = StratifiedKFold(n_splits=CV, shuffle=True, random_state=42)
    scores = cross_validate(model, X=X_train, y=y_train, cv=kf)
    return 1.0 - scores['test_score'].mean()


def objective_lgb(trial_lgb):

    param_grid_lgb = {
        'num_leaves': trial_lgb.suggest_int("num_leaves", 3, 10),
        'learning_rate': trial_lgb.suggest_loguniform("learning_rate", 1e-8, 1.0),
        'max_depth': trial_lgb.suggest_int("max_depth", 3, 10),
        "random_state": 0
    }
    X_train, y_train = main.X_train, main.y_train
    model = LGBMClassifier(**param_grid_lgb)

    kf = StratifiedKFold(n_splits=CV, shuffle=True, random_state=42)
    scores = cross_validate(model, X=X_train, y=y_train, cv=kf)
    return 1.0 - scores['test_score'].mean()


def objective_lr(trial_lr):
    warnings.filterwarnings('ignore')
    param_grid_lr = {
        'C' : trial_lr.suggest_int("C", 1, 100),
        "random_state": 0
    }
    X_train, y_train = main.X_train, main.y_train
    model = LogisticRegression(**param_grid_lr)

    kf = StratifiedKFold(n_splits=CV, shuffle=True, random_state=42)
    scores = cross_validate(model, X=X_train, y=y_train, cv=kf)
    # 最小化なので 1.0 からスコアを引く
    return 1.0 - scores['test_score'].mean()


def objective_svc(trial_svc):
    warnings.filterwarnings('ignore')
    param_grid_svc = {
        'C' : trial_svc.suggest_int("C", 50, 200),
        'gamma': trial_svc.suggest_loguniform("gamma", 1e-4, 1.0),
        "random_state": 0,
        'kernel': 'rbf'
    }

    X_train, y_train = main.X_train, main.y_train
    model = SVC(**param_grid_svc)

    kf = StratifiedKFold(n_splits=CV, shuffle=True, random_state=42)
    scores = cross_validate(model, X=X_train, y=y_train, cv=kf)
    # 最小化なので 1.0 からスコアを引く
    return 1.0 - scores['test_score'].mean()


def study_optuna(classifier, objective):
    study = optuna.create_study()
    study.optimize(objective, n_trials=trials, timeout=timeout)
    st.write('==' + '  ' + classifier)
    st.write('Best_prameter:{}'. format(study.best_params))
    st.write('Best_value: {}'.format(study.best_value))
    # print('Best_trial: {}'.format(study.best_trial))
    return study.best_params


def claasifier_mean_std(classifer):
    train_feature, train_tagert = main.train_feature, main.train_tagert

    kf = StratifiedKFold(n_splits=CV, shuffle=True, random_state=42)
    scores = cross_validate(classifer, X=train_feature, y=train_tagert, cv=kf)
    mean = scores["test_score"].mean()
    std = scores["test_score"].std()
    return mean, std


def optuna_ml():
    train_feature, train_tagert = main.train_feature, main.train_tagert

    rfc_best_param = study_optuna("RandomForest", objective_rfc)
    rfc_best = RandomForestClassifier(**rfc_best_param)
    rfc_mean, rfc_std = claasifier_mean_std(rfc_best)
    rfc_best.fit(train_feature, train_tagert)

    xgb_best_param = study_optuna("XGBClassifier", objective_xgb)
    xgb_best = XGBClassifier(**xgb_best_param)
    xgb_mean, xgb_std = claasifier_mean_std(xgb_best)
    xgb_best.fit(train_feature, train_tagert)

    lgb_best_param = study_optuna("LGBMClassifier", objective_lgb)
    lgb_best = LGBMClassifier(**lgb_best_param)
    lgb_mean, lgb_std = claasifier_mean_std(lgb_best)
    lgb_best.fit(train_feature, train_tagert)

    lr_best_param = study_optuna("LogisticRegression", objective_lr)
    lr_best = LogisticRegression(**lr_best_param)
    lr_mean, lr_std = claasifier_mean_std(lr_best)
    lr_best.fit(train_feature, train_tagert)

    svc_best_param = study_optuna("SVC", objective_svc)
    svc_best = SVC(**svc_best_param)
    svc_mean, svc_std = claasifier_mean_std(svc_best)
    svc_best.fit(train_feature, train_tagert)

    st.success('Finishied Machine Learning!!')

    dic = {'mean': [rfc_mean, xgb_mean, lgb_mean, lr_mean, svc_mean],
        'std': [rfc_std, xgb_std, lgb_std,lr_std, svc_std]}

    index=['rfc', 'xgb', 'lgb', 'lr', 'svc']

    df_cls = pd.DataFrame(dic, index=index)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    sns.barplot(data=df_cls, x=index, y='mean', ax = axes[0])
    axes[0].set_ylim([0.8, 0.9])
    axes[0].set_xlabel('Classifier')
    axes[0].set_ylabel('Accuracy of average')
    axes[0].set_title('Mean')
    sns.barplot(data=df_cls, x=index, y='std', ax = axes[1])
    axes[1].set_ylim([0.01, 0.06])
    axes[1].set_xlabel('Classifier')
    axes[1].set_ylabel('Accuracy of STD')
    axes[1].set_title('STD')
    fig.tight_layout()
    st.pyplot(fig)

    return rfc_best, xgb_best, lgb_best, lr_best, svc_best


def output_file(rfc_best, xgb_best, lgb_best, lr_best, svc_best):
    test_feature = main.test_feature
    test_raw = main.test_raw
    pred_2 = {
    'rfc': rfc_best.predict(test_feature),
    'xgb': xgb_best.predict(test_feature),
    'lgb': lgb_best.predict(test_feature),
    'lr': lr_best.predict(test_feature),
    'svc': svc_best.predict(test_feature)
    }
    # ファイル出力
    for key, value in pred_2.items():
        pd.concat(
            [
                pd.DataFrame(test_raw['PassengerId']).reset_index(drop=True),
                pd.DataFrame(value, columns=[sel_target])
            ],
            axis=1
        ).to_csv('../output/submittion_{0}_{1}.csv'.format(key, file_name), index=False)
