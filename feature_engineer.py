
import os
import numpy as np
import pandas as pd

from sklearn import manifold
from sklearn.cluster import KMeans
from functools import reduce
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.decomposition import PCA
from data_preprocess import time_pass


@time_pass
def box_split(data_num, n_cluster, string):
    """
    利用KMeans对数值型数据离散化，分成10个区间
    """
    def var_split(series):
        """
        对某一个series进行KMeans聚类，然后分成10个离散的值
        """
        if len(series.unique()) < n_cluster:
            return series
        ms = KMeans(n_clusters=n_cluster).fit(series.values.reshape(-1, 1))
        cluster_centers = ms.cluster_centers_.reshape(1, -1)[0]
        cut_points = list((np.sort(cluster_centers)[:-1] + np.sort(cluster_centers)[1:]) / 2)
        new_series = np.ones(len(series)) * (n_cluster - 1)
        for i in list(reversed(range(n_cluster-1))):
            new_series[series < cut_points[i]] = i

        return new_series

    data_bins = data_num.apply(var_split, axis=0)
    data_bins.columns = ['box_%s_%s' % (string, x) for x in data_bins.columns]

    return data_bins


@time_pass
def col_cluster(x, n_cluster, string):
    """
    对列变量进行聚类，然后每一个在算出每一类列变量里取值为1的数量，这个函数只使用用0-1型矩阵
    """
    core_data = x.fillna(0)
    col_labels = KMeans(n_clusters=n_cluster).fit(core_data.T).labels_
    new_columns = pd.MultiIndex.from_arrays([list(col_labels), list(core_data.columns)],
                                            names=['cluster', 'original'])
    core_data.columns = new_columns
    new_core_data = core_data.groupby(level='cluster', axis=1).sum()
    new_core_data.columns = ['cluster_%s_%s' % (string, x) for x in new_core_data.columns]

    return new_core_data


@time_pass
def use_part_label(data, string):
    new_data = pd.merge(all_id[['id']], data, on='id', how='inner')
    new_data[new_data.columns[-1]] = [str(x) for x in new_data[new_data.columns[-1]]]

    new_data_dummy = pd.get_dummies(new_data[new_data.columns[-1]])
    new_data_dummy_part = new_data_dummy[[str(x) for x in important_feature_cluster['feature']]].copy()
    new_data_dummy_part.columns = ['%s_%s' % (string, x) for x in new_data_dummy_part.columns]
    new_data_dummy_part['id'] = new_data['id']

    return new_data_dummy_part


@time_pass
def reduce_data(data, string, easy=True):
    """
    利用PCA确定降维至只保留原始信息的80%
    参数i表示数据是从第几列开始的

    同时利用LDA把数据降到1维，因为降至的维数必须小于类别数

    同时利用局部线性嵌入LLE把数据降到5维
    """
    # ---------------pca降维-----------------------------------
    pca = PCA(n_components=0.8).fit_transform(data.drop(['id', 'label'], axis=1))
    df_pca = pd.DataFrame(pca, columns=['%s_pca_%s' % (string, x) for x in range(pca.shape[1])])

    bandwidth = estimate_bandwidth(data.drop(['id', 'label'], axis=1), quantile=0.2, n_samples=500)
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    ms.fit(data.drop(['id', 'label'], axis=1))
    df_labels = pd.DataFrame(ms.labels_, columns=['%s_ms' % string])

    result = pd.concat([data[['id']], df_pca, df_labels], axis=1)
    # ---------------局部线性嵌入LLE降维-----------------------------------
    if easy:
        lle = manifold.LocallyLinearEmbedding(n_neighbors=30, n_components=5, method='standard')
        trans_data = lle.fit_transform(data.drop(['id', 'label'], axis=1))
        df_lle = pd.DataFrame(trans_data, columns=['%s_lle_%x' % (string, x) for x in range(trans_data.shape[1])])
        # ---------------TSNE降维-----------------------------------
        tsne = manifold.TSNE(n_components=2).fit_transform(data.drop(['id', 'label'], axis=1))
        df_tsne = pd.DataFrame(tsne, columns=['%s_tsne_%x' % (string, x) for x in range(tsne.shape[1])])

        result = pd.concat([data[['id']], df_pca, df_lle, df_tsne], axis=1)

    return result


@time_pass
def get_new_df_feature_18():
    """
    对数值型数据进行“排序化”和“离散化”
    其中离散化的方式为利用Mean-shift进行聚类，得到类别特征
    """
    # 先找出数值型的数据
    data_num = df_feature_18.drop('id', axis=1)

    # 数据排序化
    data_num_rank = data_num.rank(axis=0)  # 这个就是排序的特征
    data_num_rank.columns = ['rank_%s' % x for x in data_num_rank.columns]

    # 数据离散化
    # data_num.fillna(0, inplace=True)
    # data_discrete = box_split(data_num, 50, 'discrete')

    # 把排序后的数据和离散化后的数据拼接起来，并且把id给加上
    the_new_df_feature_18 = pd.concat([df_feature_18[['id']], data_num_rank], axis=1)

    return the_new_df_feature_18


def get_rank(data):
    tmp = data.drop('id', axis=1)
    tmp_rank = tmp.rank(axis=0)
    tmp_rank['id'] = data['id']

    return tmp_rank


def get_box(data):
    tmp = data.drop('id', axis=1)
    tmp.fillna(0, inplace=True)
    data_discrete = box_split(tmp, 50, 'discrete')
    data_discrete['id'] = data['id']

    return data_discrete


@time_pass
def get_minor_major(data, string, minority_ratio, majority_ratio):
    """
    data是一个0-1型DataFrame

    以用户安装app的数据为例，这个函数计算出了：
    1.每个用户安装小众软件的数量
    2.每个用户安装大众软件的数量
    3.每个用户安装任何软件的数量

    """
    by_column = data.apply(sum, axis=0)

    minority_num = np.percentile(by_column.values, minority_ratio)
    majority_num = np.percentile(by_column.values, majority_ratio)

    minority_features = by_column[by_column <= minority_num].index  # 小众特征
    majority_features = by_column[by_column > majority_num].index  # 大众特征

    minority_features_num = data[minority_features].apply(sum, axis=1)  # 每个用户小众特征的数量
    majority_features_num = data[majority_features].apply(sum, axis=1)  # 每个用户大众特征的数量
    all_features_num = data.apply(sum, axis=1)  # 每个用户所有特征的数量

    the_minor_major = pd.concat([minority_features_num, majority_features_num,
                                 all_features_num], axis=1)
    the_minor_major.columns = ['minor_major_%s_%s' % (string, x) for
                               x in the_minor_major.columns]

    return the_minor_major


@time_pass
def get_symbol_final():
    """
    对data_symbol进行聚类，并计算出一个横向加和作为特征
    """
    dat_symbol = reduce(lambda x, y: pd.merge(x, y, on='id', how='left'),
                        [sample_in_first.drop('label', axis=1),
                         sample_in_second.drop('label', axis=1),
                         sample_in_both.drop('label', axis=1)])

    symbol_data_part = dat_symbol[['id'] + list(important_feature_symbol['feature'])]

    return symbol_data_part


@time_pass
def get_apps_final(the_apps_dummy):
    """
    计算出：
    1.每个用户安装app的数量；
    2.每个用户安装小众app的数量；
    3.每个用户安装大众app的数量；
    4.根据每个用户安装app的向量进行Mean-shift聚类的结果
    """
    core_data = the_apps_dummy.drop(['id'], axis=1)

    the_apps_final = get_minor_major(core_data, 'apps', 5, 90)

    # new_core_data = col_cluster(core_data, n_cluster, 'app')

    # the_apps_final = pd.concat([apps_minor_major, new_core_data], axis=1)
    the_apps_final['id'] = the_apps_dummy['id']

    return the_apps_final


if __name__ == '__main__':
    input_path = './'

    sample_train = pd.read_table(os.path.join(input_path, "open_data/sample_train.txt"))  # 训练集
    valid_id = pd.read_table(os.path.join(input_path, "open_data/valid_id.txt"))  # 验证集
    test_id = pd.read_table(os.path.join(input_path, "open_data/test_id.txt"))  # 测试集

    df_feature_18 = pd.read_csv(os.path.join(input_path, "output/df_feature_18.csv"))
    sample_dat_risk = pd.read_csv(os.path.join(input_path, "output/sample_dat_risk.csv"))
    sample_in_first = pd.read_csv(os.path.join(input_path, "output/sample_in_first.csv"))
    sample_in_second = pd.read_csv(os.path.join(input_path, "output/sample_in_second.csv"))
    sample_in_both = pd.read_csv(os.path.join(input_path, "output/sample_in_both.csv"))
    one_step_apps_dummy = pd.read_csv(os.path.join(input_path, "output/one_step_apps_dummy.csv"))
    # times_sum = pd.read_csv(os.path.join(input_path, "output/Louvain_result/times_sum.csv"))
    agg_2 = pd.read_csv(os.path.join(input_path, "output/Louvain_result/agg_2.csv"))
    apps_dummy = pd.read_csv('./output/apps_dummy.csv')

    important_feature_cluster = pd.read_csv('./output/important_feature_cluster.csv')
    important_feature_symbol = pd.read_csv('./output/important_feature_symbol.csv')
    important_feature_app = pd.read_csv('./output/important_feature_app.csv')
    # 下面对数据进行降维-------------------------------------------------------
    # r_df_feature_18 = reduce_data(pd.merge(df_feature_18.dropna(), sample_train, on='id', how='left'),
    #                               'df_feature_18')  # 10分钟
    # 加了这个垃圾反而变差了
    all_id = pd.concat([sample_train[['id']], valid_id[['id']], test_id[['id']]], axis=0)

    # 对数值型数据进行“排序化”和“离散化”-----------------------------------------
    new_df_feature_18 = get_new_df_feature_18()

    # 把图聚类得到的结果提取出一部分---------------------------------------------
    # times_sum_0 = use_part_label(times_sum, 'times_sum')
    agg_2_0 = use_part_label(agg_2, 'agg_2')   # 这个是作为times_sum_0的备胎,备胎牛逼！

    symbol_final = get_symbol_final()

    apps_minor_major = get_apps_final(apps_dummy)
    # apps_minor_major.to_csv('./output/apps_minor_major.csv', index=False)
    # 下面是把所有的数据merge起来------------------------------------------------
    data_reduced_pre = reduce(lambda x, y: pd.merge(x, y, on='id', how='left'),
                              [all_id,
                              new_df_feature_18, agg_2_0,
                              sample_dat_risk,
                              symbol_final,
                              one_step_apps_dummy, apps_minor_major])

    data_reduced_pre.to_csv('./output/data_reduced_pre.csv', index=False)
    # 只用data_reduced_pre的AUC为0.651

    # 加上下面的AUC为0.673
    # 下面在添加一些从关联图上提取的特征
    data_reduced_pre = pd.read_csv('./output/data_reduced_pre.csv')
    graph_feature_big = pd.read_csv('./output/graph_feature_big.csv')
    one_step_label = pd.read_csv('./output/one_step_label.csv')
    two_step_label = pd.read_csv('./output/two_step_label.csv')
    one_spread_label = pd.read_csv('./output/one_spread_label.csv')
    black_ratio_feature = pd.read_csv('./output/black_ratio_feature.csv')
    one_step_feature_df = pd.read_csv('./output/one_step_feature_df.csv')

    graph_feature_rank = get_rank(graph_feature_big)
    one_step_label_rank = get_rank(one_step_label)
    two_step_label_rank = get_rank(two_step_label)
    one_spread_label_rank = get_rank(one_spread_label)
    black_ratio_feature_rank = get_rank(black_ratio_feature)
    one_step_feature_df_rank = get_rank(one_step_feature_df)  # 有0.5%的提升，继续加变量吧

    graph_feature_box = get_box(graph_feature_big)
    one_step_label_box = get_box(one_step_label)
    two_step_label_box = get_box(two_step_label)
    one_spread_label_box = get_box(one_spread_label)

    data_reduced = reduce(lambda x, y: pd.merge(x, y, on='id', how='left'),
                          [data_reduced_pre, graph_feature_rank, one_step_label_rank,
                           two_step_label_rank, one_spread_label_rank, black_ratio_feature_rank,
                           one_step_feature_df_rank, graph_feature_box,
                           one_step_label_box, two_step_label_box, one_spread_label_box])
    data_reduced.to_csv('./output/data_reduced.csv', index=False)

    # 得看看验证集和测试集上'one_spread_df_rank'这些特征的有无
    sample_data = pd.merge(sample_train[['id']], black_ratio_feature, on='id', how='left')
    valid_data = pd.merge(valid_id[['id']], black_ratio_feature, on='id', how='left')
    test_data = pd.merge(test_id[['id']], black_ratio_feature, on='id', how='left')

    sample_data.isnull().apply(sum, axis=0)/len(sample_data)
    valid_data.isnull().apply(sum, axis=0)/len(valid_data)
    test_data.isnull().apply(sum, axis=0) / len(test_data)