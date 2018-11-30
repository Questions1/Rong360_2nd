
import community
import networkx as nx
import numpy as np
import pandas as pd
from data_preprocess import time_pass


@time_pass
def get_graph_data(path):
    """
    先选出那些‘转账次数’和‘转账金额’大于0的数据
    然后生成一个新的特征，来代表图上边的权重
    """
    data_pre = pd.read_csv(path)
    rule = (data_pre['times_sum'] > 0) & (data_pre['weight_sum'] > 0)
    the_data = data_pre[rule].copy()

    the_data['agg_2'] = np.log(the_data['weight_sum'] + 0.01)

    return the_data


@time_pass
def get_community(weight):
    """
    进行图聚类，发现社区
    weight: 选择哪个变量作为权重
    """
    FG = nx.Graph()
    FG.add_weighted_edges_from(graph_data[['from_id', 'to_id', weight]].values)

    result = pd.DataFrame({'id': list(FG.nodes)})
    print('node number: %s' % len(result))
    dendrogram = community.generate_dendrogram(FG)

    for level in range(len(dendrogram)):
        the_partition = community.partition_at_level(dendrogram, level)
        result['%s_label_%s' % (weight, level)] = list(the_partition.values())

    return result


if __name__ == '__main__':
    which_weight = 'agg_2'  # 可能的取值为'times_sum','agg_2'
    graph_data = get_graph_data('./output/data_louvain.csv')
    data_community = get_community(which_weight)
    data_community.to_csv('./output/Louvain_result/%s.csv' % which_weight, index=False)


