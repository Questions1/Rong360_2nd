
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt

from sklearn.model_selection import cross_val_score
from functools import reduce


if __name__ == '__main__':
    """
    这个‘dat_symbol’直接预测居然0.5的AUC，应该是用来算它的一度联系人的这个的特征的情况吧
    
    light_gbm还是不如xgboost更精细啊
    
    所有数据的一步亲属有3173091个，与所有的‘dat_symbol’的交集有312055个，平均每个ID有10个有效数据
    
    按照这个道理来说的话，'dat_risk'也应该是如此才对
    'dat_risk'长度为7437689，但是与所有的'one_step'的交集只有103822个，平均每个人3个
    """
    sample_train = pd.read_table('./open_data/sample_train.txt')

    sample_in_first = pd.read_csv('./output/sample_in_first.csv')
    sample_in_second = pd.read_csv('./output/sample_in_second.csv')
    sample_in_both = pd.read_csv('./output/sample_in_both.csv')

    dat_symbol = reduce(lambda x, y: pd.merge(x, y, on='id', how='left'),
                        [sample_in_first.drop('label', axis=1),
                         sample_in_second.drop('label', axis=1),
                         sample_in_both.drop('label', axis=1)])
    dat_symbol = pd.merge(dat_symbol, sample_train, on='id')
    # 下面为训练模型准备X和Y
    label = dat_symbol['label']
    data = dat_symbol.drop(['id', 'label'], axis=1)
    na_num = data.apply(lambda x: sum(x.isnull()), axis=1)
    data['na_num'] = na_num

    params = {'booster': 'gbtree', 'objective': 'binary:logistic', 'eval_metric': 'auc',
              'seed': 0, 'silent': 1, 'min_child_weight': 3, 'max_depth': 4, 'subsample': 0.7,
              'colsample_bytree': 0.8, 'learning_rate': 0.05, 'lambda': 1.1,
              'n_estimators': 100}
    xgb_model = xgb.XGBClassifier(**params)

    scores = cross_val_score(xgb_model, data, label, cv=5, scoring='roc_auc', n_jobs=1)
    print(np.mean(scores))

    xgb_model.fit(data, label)

    feature_importance = pd.DataFrame({'feature': data.columns,
                                       'importance': xgb_model.feature_importances_})
    feature_importance.sort_values(by='importance', ascending=False, inplace=True)
    feature_importance.reset_index(drop=True, inplace=True)

    important_feature = feature_importance['feature'][feature_importance['importance'] > 0]

    # 看起来选前7个特征最好了
    scores = cross_val_score(xgb_model, data[important_feature[:7]], label, cv=5, scoring='roc_auc', n_jobs=1)
    print(np.mean(scores))

    score_list = []
    feature_num = list(range(2, 28, 1))
    for i in feature_num:
        scores = cross_val_score(xgb_model, data[important_feature[:i]], label, cv=5, scoring='roc_auc', n_jobs=1)
        xx = np.mean(scores)
        score_list.append(xx)
        print(i)

    score_series = pd.Series(score_list, index=feature_num)
    plt.plot(score_series)
    plt.xlabel('symbol_num')
    plt.ylabel('AUC')

    # 不过我们还是把28个symbol都输出了吧
    important_feature_symbol = pd.DataFrame({'feature': important_feature[:]})
    important_feature_symbol.to_csv('./output/important_feature_symbol.csv', index=False)
