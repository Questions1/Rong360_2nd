
import os
import pandas as pd
from functools import reduce
from data_preprocess import time_pass


def read_big_csv(path):
    reader = pd.read_csv(path, chunksize=10000)
    data = pd.concat(reader, axis=0, ignore_index=True)

    return data


@time_pass
def circle_2(id_in, data, output=False):
    """
    找到在两步转账记录以内可以转回id_in里边的记录
    找儿子和找爸爸的含义是一样的
    """
    self_set = set(id_in['id'])

    son = pd.merge(id_in, data, left_on='id', right_on='from_id', how='inner')
    print('son.shape[0]: %s' % son.shape[0])
    grand_son = pd.merge(pd.DataFrame({'id': son['to_id'].unique()}),
                         data, left_on='id', right_on='from_id', how='inner')
    print('grand_son.shape[0]: %s' % grand_son.shape[0])
    inter_set_s = self_set.intersection(set(grand_son['to_id']))
    print(len(inter_set_s))
    back_grand_son = pd.merge(pd.DataFrame({'id': list(inter_set_s)}),
                              grand_son.drop('id', axis=1),
                              left_on='id', right_on='to_id', how='inner')

    back_son = pd.merge(pd.DataFrame({'id': list(set(back_grand_son['from_id']))}),
                        son.drop('id', axis=1), left_on='id', right_on='to_id', how='inner')

    result_1 = set(back_son['index']).union(set(back_grand_son['index']))

    if not output:
        father = pd.merge(id_in, data, left_on='id', right_on='to_id', how='inner')
        result_2 = set(son['index']).union(set(father['index']))

        return list(result_1.union(result_2))
    if output:
        return list(result_1)


if __name__ == '__main__':
    """选出起着桥接作用的一度联系人（data_louvain_output）和所有的一度联系人（data_louvain）"""
    input_path = './'
    sample_train = pd.read_table(os.path.join(input_path, "open_data/sample_train.txt"))  # 训练集约1.9万
    valid_id = pd.read_table(os.path.join(input_path, "open_data/valid_id.txt"))  # 验证集
    test_id = pd.read_table(os.path.join(input_path, "open_data/test_id.txt"))  # 测试集

    file_names = os.listdir('./output/dat_edge_feature')
    dat_edge_feature = reduce(lambda x, y: x.append(y),
                              (read_big_csv('./output/dat_edge_feature/%s' % z) for z in file_names))

    all_id = pd.concat([sample_train[['id']], valid_id[['id']], test_id[['id']]], axis=0)

    # 加入转账记录的index，并找出这些记录
    dat_edge_feature['index'] = range(len(dat_edge_feature))

    # 选出起着桥接作用的一度联系人
    needed_index_output = circle_2(all_id, dat_edge_feature[['from_id', 'to_id', 'index']], output=True)
    pd.DataFrame({'index': needed_index_output}).to_csv('./output/needed_index_output.csv', index=False)

    needed_index_output = pd.read_csv('./output/needed_index_output.csv')
    data_louvain_output = dat_edge_feature.iloc[needed_index_output['index'].values, :]
    data_louvain_output.to_csv('./output/data_louvain_output.csv', index=False)

    # 选出所有的一度联系人
    needed_index = circle_2(all_id, dat_edge_feature[['from_id', 'to_id', 'index']], output=False)
    pd.DataFrame({'index': needed_index}).to_csv('./output/needed_index.csv', index=False)

    needed_index = pd.read_csv('./output/needed_index.csv')
    data_louvain = dat_edge_feature.iloc[needed_index['index'].values, :]
    data_louvain.to_csv('./output/data_louvain.csv', index=False)
