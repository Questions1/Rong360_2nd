
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score


if __name__ == '__main__':
    agg_2 = pd.read_csv('./output/Louvain_result/agg_2.csv')

    sample_train = pd.read_table("./open_data/sample_train.txt")  # 训练集约1.9万
    valid_id = pd.read_table("./open_data/valid_id.txt")  # 验证集
    test_id = pd.read_table("./open_data/test_id.txt")  # 测试集

    all_id = pd.concat([sample_train[['id']], valid_id[['id']], test_id[['id']]], axis=0)

    # --------------------------------------------------------
    all_comm = pd.merge(all_id, agg_2[['id', 'agg_2_label_3']], on='id')
    all_comm['agg_2_label_3'] = [str(x) for x in all_comm['agg_2_label_3']]
    all_comm_dummy = pd.get_dummies(all_comm['agg_2_label_3'])
    all_comm_dummy['id'] = all_comm['id']

    sample_comm = pd.merge(sample_train, all_comm_dummy, on='id')

    data = sample_comm.drop(['id', 'label'], axis=1)
    label = sample_comm['label']
    # --------------------------------------------------------

    params = {'booster': 'gbtree', 'objective': 'binary:logistic', 'eval_metric': 'auc',
              'seed': 0, 'silent': 1, 'min_child_weight': 3, 'max_depth': 4, 'subsample': 0.7,
              'colsample_bytree': 0.8, 'learning_rate': 0.05, 'lambda': 1.1,
              'n_estimators': 100}
    xgb_model = xgb.XGBClassifier(**params)

    # scores = cross_val_score(xgb_model, data, label, cv=5, scoring='roc_auc', n_jobs=1)
    # print(np.mean(scores))

    xgb_model.fit(data, label)

    # 下面选出重要的特征
    feature_importance = pd.DataFrame({'feature': data.columns, 'importance': xgb_model.feature_importances_})
    feature_importance.sort_values(by='importance', ascending=False, inplace=True)
    feature_importance.reset_index(drop=True, inplace=True)

    important_feature = feature_importance['feature'][feature_importance['importance'] > 0]
    print(len(important_feature))

    # 选前47可以达到0.556的AUC
    # 直接用weight的效果不如用log的效果哎
    # 在那个加权label上也用一下试试
    scores = cross_val_score(xgb_model, data[important_feature[:]], label, cv=5, scoring='roc_auc', n_jobs=1)
    print(np.mean(scores))

    score_list = []
    feature_num = list(range(45, 75, 2))
    for i in feature_num:
        scores = cross_val_score(xgb_model, data[important_feature[:i]], label, cv=5, scoring='roc_auc', n_jobs=1)
        x = np.mean(scores)
        score_list.append(x)
        print(i)
    score_series = pd.Series(score_list, index=feature_num)
    plt.plot(score_series)

    agg_2_important = all_comm_dummy[['id'] + list(important_feature)]

    # important_feature_cluster = pd.DataFrame({'feature': important_feature[:47]})
    agg_2_important.to_csv('./output/agg_2_important.csv', index=False)




