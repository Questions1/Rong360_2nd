
import os
import igraph
import pandas as pd


def summary_cluster(mode='WEAK'):
    """
    团伙发现，为半监督做准备
    """
    clusters = graph.clusters(mode=mode)  # 这个是用来做团伙发现的
    cluster_df = pd.DataFrame({'id': [int(x) for x in whole_id], 'membership': clusters.membership})

    # 下面整理一下各个团伙里有标签的人数和黑化率
    cluster_df_label = pd.merge(cluster_df, sample_train, on='id', how='inner')
    label_count = cluster_df_label.groupby(by='membership')['label'].count()
    label_sum = cluster_df_label.groupby(by='membership')['label'].sum()

    guys_df = pd.DataFrame({'membership': label_count.index,
                            'count': label_count.values,
                            'label_sum': label_sum.values})

    # 算出每一个id所在团伙的有标签的长度和平均黑度
    c1 = pd.merge(all_id_num, cluster_df, on='id', how='left')
    c2 = pd.merge(c1, guys_df, on='membership', how='left')
    c3 = pd.merge(c2, sample_train, on='id', how='left')
    c4 = c3[c3['count'] > 1].copy()

    tmp1 = c4['label'].copy()
    tmp2 = c4['count'].copy()
    tmp2[tmp1 >= 0] = tmp2[tmp1 >= 0] - 1
    tmp1[tmp1.isnull()] = 0

    c4['black_ratio'] = (c4['label_sum'] - tmp1)/tmp2
    c4['black_ratio'] = [round(x, 3) for x in c4['black_ratio']]

    # 下面算出所有团伙里的总人数
    cluster_df_count = cluster_df.groupby(by='membership')['membership'].count()
    cluster_df_count_df = pd.DataFrame({'membership': cluster_df_count.index,
                                        'count_all': cluster_df_count.values})
    # 注意：那些黑化率为0且团伙人数很少的可以统一归为好人了！！！！
    the_cluster_summary = pd.merge(guys_df, cluster_df_count_df, on='membership')
    # 从那些小数群体下手岂不美哉，毕竟作案的都是些小团伙才对啊！！！！
    return the_cluster_summary, cluster_df, c4[['id', 'membership', 'black_ratio', 'count']]


if __name__ == '__main__':
    input_path = './'
    sample_train = pd.read_table(os.path.join(input_path, "open_data/sample_train.txt"))  # 训练集约1.9万
    valid_id = pd.read_table(os.path.join(input_path, "open_data/valid_id.txt"))  # 验证集
    test_id = pd.read_table(os.path.join(input_path, "open_data/test_id.txt"))  # 测试集

    all_id_num = pd.concat([sample_train[['id']], valid_id[['id']], test_id[['id']]], axis=0)
    all_id_str = [str(x) for x in all_id_num['id']]

    data_louvain_output = pd.read_csv('./output/data_louvain_output.csv')
    data_louvain_output['from_id'] = [str(x) for x in data_louvain_output['from_id']]
    data_louvain_output['to_id'] = [str(x) for x in data_louvain_output['to_id']]

    whole_id = list(set(data_louvain_output['from_id']).union(set(data_louvain_output['to_id'])))

    inner_id = list(set(all_id_str).intersection(set(whole_id)))
    # 把边添加到图里边
    # 我们采取一步一步添加的方式，也可以在创建图的过程中一次性把点的属性、边的属性加上去
    graph = igraph.Graph(directed=True)
    graph.add_vertices(whole_id)
    graph.add_edges(data_louvain_output[['from_id', 'to_id']].values)
    graph.es['weight'] = data_louvain_output['weight_sum'].values + 0.01

    degree_in = graph.degree(vertices=whole_id, mode='IN')
    degree_out = graph.degree(vertices=whole_id, mode='OUT')
    degree_all = graph.degree(vertices=whole_id, mode='ALL')
    between_undirected = graph.betweenness(vertices=whole_id, directed=False, cutoff=9, weights='weight')
    between_directed = graph.betweenness(vertices=whole_id, directed=True, cutoff=9, weights='weight')
    eigenvector_centrality = graph.eigenvector_centrality(weights='weight')
    closeness_in = graph.closeness(vertices=whole_id, cutoff=9, mode='IN', weights='weight')
    closeness_out = graph.closeness(vertices=whole_id, cutoff=9, mode='OUT', weights='weight')
    closeness_all = graph.closeness(vertices=whole_id, cutoff=9, mode='ALL', weights='weight')
    page_rank_directed = graph.pagerank(vertices=whole_id, directed=True, weights='weight')
    page_rank_undirected = graph.pagerank(vertices=whole_id, directed=False, weights='weight')

    centrality_df = pd.DataFrame({'id': whole_id,
                                  'degree_in': degree_in,
                                  'degree_out': degree_out,
                                  'degree_all': degree_all,
                                  'between_undirected': between_undirected,
                                  'between_directed': between_directed,
                                  'closeness_in': closeness_in,
                                  'closeness_out': closeness_out,
                                  'closeness_all': closeness_all,
                                  'page_rank_directed': page_rank_directed,
                                  'page_rank_undirected': page_rank_undirected,
                                  'eigen': eigenvector_centrality})

    centrality_df.to_csv('./output/graph_feature_big.csv', index=False)

    cluster_summary, all_cluster_info, black_ratio_feature = summary_cluster(mode='STRONG')
    black_ratio_feature.to_csv('./output/black_ratio_feature.csv', index=False)


