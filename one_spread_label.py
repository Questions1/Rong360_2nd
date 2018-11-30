
import os
import numpy as np
import pandas as pd


def cal_fea(df):
    """
    计算出df里边的平均黑度之类的特征
    """
    df['weight_label'] = df['weight_sum_x'] * df['weight_sum_y'] * df['label']
    agg_df = df.groupby(by='id')[['weight_label', 'label']].sum()
    weight_label_mean = agg_df['weight_label'].mean()
    label_mean = agg_df['label'].mean()
    label_sign_mean = np.sign(agg_df['label']).mean()
    length = len(agg_df)

    return [weight_label_mean, label_mean, label_sign_mean, length]


def get_1_spread(one_id):
    """
    计算每一个ID的儿子、父亲的一度联系人(儿子、父亲)的平均黑度
    """
    son_df = son[son['from_id'] == one_id]
    son_one_spread_son = pd.merge(son_df[['to_id', 'weight_sum']].rename(columns={'to_id': 'from_id'}),
                                  data_louvain_output, on='from_id')
    if len(son_one_spread_son) == 0:
        son_son = [np.nan] * 4
    else:
        son_one_spread_son_1 = pd.merge(son_one_spread_son, sample_train,
                                        left_on='to_id', right_on='id')
        son_one_spread_son_2 = son_one_spread_son_1[son_one_spread_son_1['id'] != one_id].copy()
        if len(son_one_spread_son_2) == 0:
            son_son = [np.nan] * 4
        else:
            son_son = cal_fea(son_one_spread_son_2)

    son_one_spread_father = pd.merge(son_df[['to_id', 'weight_sum']], data_louvain_output, on='to_id')
    if len(son_one_spread_father) == 0:
        son_father = [np.nan] * 4
    else:
        son_one_spread_father_1 = pd.merge(son_one_spread_father, sample_train,
                                           left_on='from_id', right_on='id')
        son_one_spread_father_2 = son_one_spread_father_1[son_one_spread_father_1['id'] != one_id].copy()
        if len(son_one_spread_father_2) == 0:
            son_father = [np.nan] * 4
        else:
            son_father = cal_fea(son_one_spread_father_2)

    # 上面计算son_son, son_father,下面计算father_father, father_son

    father_df = father[father['to_id'] == one_id]
    father_one_spread_father = pd.merge(father_df[['from_id', 'weight_sum']].rename(columns={'from_id': 'to_id'}),
                                        data_louvain_output, on='to_id')
    if len(father_one_spread_father) == 0:
        father_father = [np.nan] * 4
    else:
        father_one_spread_father_1 = pd.merge(father_one_spread_father, sample_train,
                                              left_on='from_id', right_on='id')
        father_one_spread_father_2 = father_one_spread_father_1[father_one_spread_father_1['id'] != one_id].copy()
        if len(father_one_spread_father_2) == 0:
            father_father = [np.nan] * 4
        else:
            father_father = cal_fea(father_one_spread_father_2)

    father_one_spread_son = pd.merge(father_df[['from_id', 'weight_sum']], data_louvain_output, on='from_id')
    if len(father_one_spread_son) == 0:
        father_son = [np.nan] * 4
    else:
        father_one_spread_son_1 = pd.merge(father_one_spread_son, sample_train,
                                           left_on='to_id', right_on='id')
        father_one_spread_son_2 = father_one_spread_son_1[father_one_spread_son_1['id'] != one_id].copy()
        if len(father_one_spread_son_2) == 0:
            father_son = [np.nan] * 4
        else:
            father_son = cal_fea(father_one_spread_son_2)

    return son_son + son_father + father_father + father_son


if __name__ == '__main__':
    input_path = './'
    sample_train = pd.read_table(os.path.join(input_path, "open_data/sample_train.txt"))  # 训练集约1.9万
    valid_id = pd.read_table(os.path.join(input_path, "open_data/valid_id.txt"))  # 验证集
    test_id = pd.read_table(os.path.join(input_path, "open_data/test_id.txt"))  # 测试集
    all_id = pd.concat([sample_train[['id']], valid_id[['id']], test_id[['id']]], axis=0)

    son = pd.read_csv('./output/son.csv')
    father = pd.read_csv('./output/father.csv')
    data_louvain_output = pd.read_csv('./output/data_louvain_output.csv')
    data_louvain_output['weight_sum'] = np.log(data_louvain_output['weight_sum'] + 1.01)

    one_spread_result = list(map(get_1_spread, all_id['id'].values))
    columns = ['weight_label_mean', 'label_mean', 'label_sign_mean', 'length']
    df_columns = (['son_son_%s' % x for x in columns]
                  + ['son_father_%s' % x for x in columns]
                  + ['father_father_%s' % x for x in columns]
                  + ['father_son_%s' % x for x in columns])
    one_spread_label = pd.DataFrame(one_spread_result, columns=df_columns)
    one_spread_label['id'] = all_id['id'].values
    one_spread_label.to_csv('./output/one_spread_label.csv', index=False)
