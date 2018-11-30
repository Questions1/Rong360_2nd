
import os
import pandas as pd
import numpy as np


def cal_two_step_feature(id_loc):
    id_dat = two_step_label[int(id_loc[1]):int(id_loc[2])]

    label = id_dat['label']
    weighted_label = id_dat['weighted_label']

    # 开始计算特征
    length = len(id_dat)

    # 开始计算label特征
    zero_num = np.sum(label == 0)
    one_num = np.sum(label != 0)
    zero_ratio = zero_num / length
    one_ratio = one_num / length

    # 开始计算weighted_label特征
    p_0 = np.min(weighted_label)
    p_25 = np.percentile(weighted_label, 25)
    p_50 = np.percentile(weighted_label, 50)
    p_75 = np.percentile(weighted_label, 75)
    p_100 = np.max(weighted_label)
    p_mean = np.mean(weighted_label)

    # 开始计算按照id_y汇总之后的特征
    id_dat_agg = id_dat.groupby(by='id_y')['label', 'weighted_label'].sum()

    label_agg = id_dat_agg['label']
    weighted_label_agg = id_dat_agg['weighted_label']

    length_agg = len(id_dat_agg)

    # 先计算label_agg的
    zero_num_agg = np.sum(label_agg == 0)
    one_num_agg = np.sum(label_agg != 0)
    zero_ratio_agg = zero_num_agg / length_agg
    one_ratio_agg = one_num_agg / length_agg

    # 然后计算weighted_label_agg的特征
    p_0_agg = np.min(weighted_label_agg)
    p_25_agg = np.percentile(weighted_label_agg, 25)
    p_50_agg = np.percentile(weighted_label_agg, 50)
    p_75_agg = np.percentile(weighted_label_agg, 75)
    p_100_agg = np.max(weighted_label_agg)
    p_mean_agg = np.mean(weighted_label_agg)

    result = ([length, zero_num, one_num, zero_ratio, one_ratio]
              + [p_0, p_25, p_50, p_75, p_100, p_mean]
              + [length_agg, zero_num_agg, one_num_agg, zero_ratio_agg, one_ratio_agg]
              + [p_0_agg, p_25_agg, p_50_agg, p_75_agg, p_100_agg, p_mean_agg])

    return result


if __name__ == '__main__':
    input_path = './'
    sample_train = pd.read_table(os.path.join(input_path, "open_data/sample_train.txt"))  # 训练集约1.9万
    valid_id = pd.read_table(os.path.join(input_path, "open_data/valid_id.txt"))  # 验证集
    test_id = pd.read_table(os.path.join(input_path, "open_data/test_id.txt"))  # 测试集
    all_id_num = pd.concat([sample_train[['id']], valid_id[['id']], test_id[['id']]], axis=0)

    data_louvain_output = pd.read_csv('./output/data_louvain_output.csv')
    data_louvain_output = data_louvain_output[['from_id', 'to_id', 'weight_sum']]

    data_louvain_output['weight_sum'] = data_louvain_output['weight_sum'] ** (1 / 3)

    # 提取grand_son的信息
    son = pd.merge(all_id_num, data_louvain_output, left_on='id', right_on='from_id', how='inner')

    grand_son = pd.merge(
        son[['id', 'to_id', 'weight_sum']].rename(columns={'to_id': 'from_id', 'weight_sum': 'weight_sum_x'}),
        data_louvain_output, on='from_id', how='inner')
    good_grand_son = pd.merge(grand_son, sample_train.rename(columns={'id': 'to_id'}), on='to_id')

    good_grand_son['weighted_label'] = good_grand_son['label'] * good_grand_son['weight_sum_x'] * good_grand_son[
        'weight_sum']

    good_grand_son.drop(['from_id', 'weight_sum_x', 'weight_sum'], axis=1, inplace=True)
    good_grand_son = good_grand_son[good_grand_son['id'] != good_grand_son['to_id']]
    good_grand_son.rename(columns={'to_id': 'id_y'}, inplace=True)

    # 提取son_father的信息
    son_father = pd.merge(son[['id', 'to_id', 'weight_sum']].rename(columns={'weight_sum': 'weight_sum_x'}),
                          data_louvain_output, on='to_id')
    good_son_father = pd.merge(son_father, sample_train.rename(columns={'id': 'from_id'}), on='from_id')
    good_son_father['weighted_label'] = good_son_father['label'] * good_son_father['weight_sum_x'] * good_son_father[
        'weight_sum']
    good_son_father.drop(['to_id', 'weight_sum_x', 'weight_sum'], axis=1, inplace=True)
    good_son_father = good_son_father[good_son_father['id'] != good_son_father['from_id']]
    good_son_father.rename(columns={'from_id': 'id_y'}, inplace=True)

    # 提取father_son的信息
    father = pd.merge(all_id_num, data_louvain_output, left_on='id', right_on='to_id', how='inner')

    father_son = pd.merge(father[['id', 'from_id', 'weight_sum']].rename(columns={'weight_sum': 'weight_sum_x'}),
                          data_louvain_output, on='from_id')
    good_father_son = pd.merge(father_son, sample_train.rename(columns={'id': 'to_id'}), on='to_id')
    good_father_son['weighted_label'] = good_father_son['weight_sum_x'] * good_father_son['weight_sum'] * \
                                        good_father_son['label']
    good_father_son.drop(['from_id', 'weight_sum_x', 'weight_sum'], axis=1, inplace=True)
    good_father_son = good_father_son[good_father_son['id'] != good_father_son['to_id']]
    good_father_son.rename(columns={'to_id': 'id_y'}, inplace=True)

    two_step_label = pd.concat([good_grand_son, good_son_father, good_father_son], axis=0, ignore_index=True)
    two_step_label.sort_values(by='id', inplace=True)
    two_step_label.reset_index(drop=True, inplace=True)

    id_counts = two_step_label['id'].value_counts(sort=False).sort_index().cumsum()

    id_loc_info = pd.DataFrame({'id': id_counts.index,
                                'start': [0] + list(id_counts.values)[:-1],
                                'stop': list(id_counts.values)})

    two_step_result = list(map(cal_two_step_feature, id_loc_info.values))

    columns = (['length', 'zero_num', 'one_num', 'zero_ratio', 'one_ratio']
               + ['p_0', 'p_25', 'p_50', 'p_75', 'p_100', 'p_mean']
               + ['length_agg', 'zero_num_agg', 'one_num_agg', 'zero_ratio_agg', 'one_ratio_agg']
               + ['p_0_agg', 'p_25_agg', 'p_50_agg', 'p_75_agg', 'p_100_agg', 'p_mean_agg'])

    two_step_label = pd.DataFrame(two_step_result, columns=['two_step_%s' % x for x in columns])
    two_step_label['id'] = id_loc_info['id']

    two_step_label.to_csv('./output/two_step_label.csv', index=False)







