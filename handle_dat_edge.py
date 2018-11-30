
import os
import numpy as np
import pandas as pd
from data_preprocess import time_pass


@time_pass
def handle_dat_edge(data_all):
    """
    把dat_edge个每一条记录的info拆开，然后输出，方便后续的计算
    为了简化计算，忽略时间信息，把所有的月份的联系记录汇总起来
    """
    def cal_multi_3(string):
        s = string.split(',')
        month_times = len(s)
        df = list(map(lambda x: list(map(eval, x.split(':')[1].split('_'))), s))
        times_sum, weight_sum = pd.DataFrame(df).sum().values

        return month_times, times_sum, weight_sum

    def cal_single_3(string):
        times_sum, weight_sum = list(map(eval, string.split(':')[1].split('_')))
        return 1, times_sum, weight_sum

    length = list(map(len, map(lambda x: x.split(','), data_all['info'])))

    dat_edge_single = data_all[np.array(length) == 1]
    dat_edge_multi = data_all[np.array(length) > 1]

    multi_pre_df = map(cal_multi_3, dat_edge_multi['info'])
    multi_feature_3 = pd.DataFrame(list(multi_pre_df), columns=['month_times', 'times_sum', 'weight_sum'])
    id_part = dat_edge_multi[['from_id', 'to_id']].reset_index(drop=True)
    multi_result = pd.concat([id_part, multi_feature_3], axis=1)

    single_pre_df = map(cal_single_3, dat_edge_single['info'])
    single_feature_3 = pd.DataFrame(list(single_pre_df), columns=['month_times', 'times_sum', 'weight_sum'])
    id_part = dat_edge_single[['from_id', 'to_id']].reset_index(drop=True)
    single_result = pd.concat([id_part, single_feature_3], axis=1)

    both_result = pd.concat([multi_result, single_result], ignore_index=True)

    return both_result


if __name__ == '__main__':
    for i in range(1, 12):
        # 第一个dat_edge有文件名，其余的没有，所以需要加一个if语句来区别对待
        if i == 1:
            dat_edge = pd.read_table('./open_data/dat_edge/dat_edge_%s' % str(i))
        else:
            dat_edge = pd.read_table('./open_data/dat_edge/dat_edge_%s' % str(i), header=None)
            dat_edge.columns = ['from_id', 'to_id', 'info']

        dat_edge_feature = handle_dat_edge(dat_edge)
        if 'dat_edge_feature' not in os.listdir('./output/'):
            os.makedirs('./output/dat_edge_feature')
        dat_edge_feature.to_csv('./output/dat_edge_feature/dat_edge_feature_%s.csv' % str(i), index=False)
