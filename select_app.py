
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt

from sklearn.model_selection import cross_val_score


if __name__ == '__main__':
    """
    利用‘xgboost’模型来选出重要的app，只保留这些app就行了
    """
    sample_train = pd.read_table('./open_data/sample_train.txt')  # 训练集约1.9万
    apps_dummy = pd.read_csv('./output/apps_dummy.csv')
    app_dat = pd.merge(sample_train, apps_dummy, on='id', how='inner')

    # 下面为训练模型准备X和Y
    label = app_dat['label']
    data = app_dat.drop(['id', 'label'], axis=1)
    na_num = data.apply(lambda x: sum(x.isnull()), axis=1)
    data['na_num'] = na_num

    params = {'booster': 'gbtree', 'objective': 'binary:logistic', 'eval_metric': 'auc',
              'seed': 0, 'silent': 1, 'min_child_weight': 3, 'max_depth': 4, 'subsample': 0.7,
              'colsample_bytree': 0.8, 'learning_rate': 0.05, 'lambda': 1.1,
              'n_estimators': 100}
    xgb_model = xgb.XGBClassifier(**params)

    # 先来交叉验证看一下dat_app数据的预测能力如何：0.605的AUC
    scores = cross_val_score(xgb_model, data, label, cv=5, scoring='roc_auc', n_jobs=1)
    print(np.mean(scores))

    xgb_model.fit(data, label)

    # 下面选出重要的特征
    feature_importance = pd.DataFrame({'feature': data.columns, 'importance': xgb_model.feature_importances_})
    feature_importance.sort_values(by='importance', ascending=False, inplace=True)
    feature_importance.reset_index(drop=True, inplace=True)

    important_feature = feature_importance['feature'][feature_importance['importance'] > 0]

    score_list = []
    feature_num = list(range(31, 301, 10))
    for i in feature_num:
        scores = cross_val_score(xgb_model, data[important_feature[:i]], label, cv=5, scoring='roc_auc', n_jobs=1)
        xx = np.mean(scores)
        score_list.append(xx)
        print(i)
    score_series = pd.Series(score_list, index=feature_num)
    plt.plot(score_series)
    plt.xlabel('app_num')
    plt.ylabel('AUC')

    # 下面根据选出来的特征再进行一次交叉验证用全部important_app的话有0.626的AUC
    scores = cross_val_score(xgb_model, data[important_feature[:]], label, cv=5, scoring='roc_auc', n_jobs=1)
    print(np.mean(scores))
    # 卧槽，只用前66个特征居然有0.656的AUC

    # 输出数据
    important_feature_app = pd.DataFrame({'feature': important_feature[:66]})
    important_feature_app.to_csv('./output/important_feature_app.csv', index=False)
