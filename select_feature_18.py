
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt

from sklearn.model_selection import cross_val_score


def get_new_df_feature_18():
    """
    对数值型数据进行“排序化”和“离散化”
    其中离散化的方式为利用Mean-shift进行聚类，得到类别特征
    """
    # 先找出数值型的数据
    data_num = df_feature_18.drop(['id', 'out_longer_5', 'out_longer_10', 'in_longer_5', 'in_longer_10'], axis=1)

    # 数据排序化
    data_num_rank = data_num.rank(axis=0)  # 这个就是排序的特征
    data_num_rank.columns = ['rank_%s' % x for x in data_num_rank.columns]

    # 数据离散化
    # data_num.fillna(0, inplace=True)
    # data_discrete = box_split(data_num, 50, 'discrete')

    # 把排序后的数据和离散化后的数据拼接起来，并且把id给加上
    the_new_df_feature_18 = pd.concat([df_feature_18[['id', 'out_longer_5', 'out_longer_10',
                                                      'in_longer_5', 'in_longer_10']],
                                       data_num_rank], axis=1)

    return the_new_df_feature_18


if __name__ == '__main__':
    sample_train = pd.read_table('./open_data/sample_train.txt')  # 训练集约1.9万
    df_feature_18 = pd.read_csv('./output/df_feature_18.csv')

    both_feature = df_feature_18.columns[[x.startswith('both') for x in df_feature_18.columns]]
    df_feature_18 = df_feature_18[['id'] + list(both_feature)]

    new_df_feature_18 = get_new_df_feature_18()

    df_feature_18_ready = pd.merge(sample_train, df_feature_18, on='id', how='inner')
    # 下面为训练模型准备X和Y
    label = df_feature_18_ready['label']
    data = df_feature_18_ready.drop(['id', 'label'], axis=1)
    na_num = data.apply(lambda x: sum(x.isnull()), axis=1)
    data['na_num'] = na_num

    params = {'booster': 'gbtree', 'objective': 'binary:logistic', 'eval_metric': 'auc',
              'seed': 0, 'silent': 1, 'min_child_weight': 3, 'max_depth': 4, 'subsample': 0.7,
              'colsample_bytree': 0.8, 'learning_rate': 0.05, 'lambda': 1.1,
              'n_estimators': 100}
    xgb_model = xgb.XGBClassifier(**params)

    # 先来交叉验证看一下"df_feature_18_ready"数据的预测能力如何：
    scores = cross_val_score(xgb_model, data, label, cv=5, scoring='roc_auc', n_jobs=1)
    print(np.mean(scores))

    xgb_model.fit(data, label)

    # 下面选出重要的特征
    feature_importance = pd.DataFrame({'feature': data.columns, 'importance': xgb_model.feature_importances_})
    feature_importance.sort_values(by='importance', ascending=False, inplace=True)
    feature_importance.reset_index(drop=True, inplace=True)

    important_feature = feature_importance['feature'][feature_importance['importance'] > 0]

    score_list = []
    for i in range(1, len(important_feature)):
        scores = cross_val_score(xgb_model, data[important_feature[:i]], label, cv=5, scoring='roc_auc', n_jobs=1)
        xx = np.mean(scores)
        score_list.append(xx)
        print(i)
    plt.plot(score_list)

    # 下面根据选出来的特征再进行一次交叉验证
    scores = cross_val_score(xgb_model, data[important_feature[:]], label, cv=5, scoring='roc_auc', n_jobs=1)
    print(np.mean(scores))

    important_feature_18 = pd.DataFrame({'feature': important_feature[:]})
    important_feature_18.to_csv('./output/important_feature_18.csv', index=False)
