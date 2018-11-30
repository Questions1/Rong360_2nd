
import os
import numpy as np
import pandas as pd
from functools import reduce
from data_preprocess import time_pass


@time_pass
def read_big_csv(path):
    reader = pd.read_csv(path, chunksize=10000)
    data = pd.concat(reader, axis=0, ignore_index=True)

    return data


def get_1_step(one_id):
    """
    计算每一个ID的一度联系人的黑度
    """
    def cal_fea(column):
        """
        :param column的取值为'from_id'或'to_id'
        """
        if column == 'from_id':
            one_step = better_son[better_son[column] == one_id]
        if column == 'to_id':
            one_step = better_father[better_father[column] == one_id]
        if len(one_step) == 0:
            return [np.nan] * 4

        black_num_label = one_step['label'].sum()
        black_mean_label = one_step['label'].mean()

        black_num_weight_label = one_step['weight_label'].sum()
        black_mean_weight_label = one_step['weight_label'].mean()

        return [black_num_label, black_mean_label,
                black_num_weight_label, black_mean_weight_label]

    result_from = cal_fea('from_id')
    result_to = cal_fea('to_id')
    result_both = [np.nanmean((x, y)) for x, y in zip(result_from, result_to)]

    return result_from + result_to + result_both


if __name__ == '__main__':
    input_path = './'
    sample_train = pd.read_table(os.path.join(input_path, "open_data/sample_train.txt"))  # 训练集约1.9万
    valid_id = pd.read_table(os.path.join(input_path, "open_data/valid_id.txt"))  # 验证集
    test_id = pd.read_table(os.path.join(input_path, "open_data/test_id.txt"))  # 测试集

    file_names = os.listdir('./output/dat_edge_feature')
    dat_edge_feature = reduce(lambda x, y: x.append(y),
                              (read_big_csv('./output/dat_edge_feature/%s' % z) for z in file_names))

    all_id = pd.concat([sample_train[['id']], valid_id[['id']], test_id[['id']]], axis=0)

    son = pd.merge(all_id, dat_edge_feature, left_on='id', right_on='from_id')
    son.to_csv('./output/son.csv', index=False)
    good_son = pd.merge(sample_train, son, left_on='id', right_on='to_id')
    better_son = good_son[['from_id', 'to_id', 'weight_sum', 'times_sum', 'label']].copy()
    better_son['weight_label'] = better_son['weight_sum'] * better_son['label']
    better_son.to_csv('./output/better_son.csv', index=False)

    father = pd.merge(all_id, dat_edge_feature, left_on='id', right_on='to_id')
    father.to_csv('./output/father.csv', index=False)
    good_father = pd.merge(sample_train, father, left_on='id', right_on='from_id')
    better_father = good_father[['from_id', 'to_id', 'weight_sum', 'times_sum', 'label']].copy()
    better_father['weight_label'] = better_father['weight_sum'] * better_father['label']
    better_father.to_csv('./output/better_father.csv', index=False)

    better_son = pd.read_csv('./output/better_son.csv')
    better_father = pd.read_csv('./output/better_father.csv')

    one_step_result = list(map(get_1_step, all_id['id'].values))
    columns = ['black_num_label', 'black_mean_label',
               'black_num_weight_label', 'black_mean_weight_label']
    df_columns = (['%s_from' % x for x in columns]
                  + ['%s_to' % x for x in columns]
                  + ['%s_both' % x for x in columns])
    one_step_label = pd.DataFrame(one_step_result, columns=df_columns)
    one_step_label['id'] = all_id['id'].values
    one_step_label.to_csv('./output/one_step_label.csv', index=False)
