
import os
import time
import pandas as pd
import numpy as np
import functools

from functools import reduce


def time_pass(func):
    @functools.wraps(func)
    def wrapper(*args, **kw):
        time_begin = time.time()
        result = func(*args, **kw)
        time_stop = time.time()
        time_passed = time_stop - time_begin
        minutes, seconds = divmod(time_passed, 60)
        hours, minutes = divmod(minutes, 60)
        print('%s: %s:%s:%s' % (func.__name__, int(hours), int(minutes), int(seconds)))
        return result

    return wrapper


@time_pass
def complete_data(the_dat_edge, the_dat_app, the_input_path):
    """
    把剩下的数据读取之后拼接到前面读取的数据后面
    """

    def read_big_table(path):
        reader = pd.read_table(path, header=None, chunksize=10000)
        data = pd.concat(reader, axis=0, ignore_index=True)

        return data

    def read_edge(filename):  # 定义一个读取数据的函数来批量读取那些被分开的数据集
        tmp = read_big_table(os.path.join(the_input_path, "open_data/dat_edge/%s" % filename))
        tmp.columns = ['from_id', 'to_id', 'info']

        return tmp

    dat_edge_names = ['dat_edge_%s' % str(x) for x in list(range(2, 12))]
    dat_edge_left = reduce(lambda x, y: x.append(y),
                           (read_edge(filename) for filename in dat_edge_names))

    def read_app(filename):  # 定义一个读取数据的函数来批量读取那些被分开的数据集
        tmp = read_big_table(os.path.join(the_input_path, "open_data/dat_app/%s" % filename))
        tmp.columns = ['id', 'apps']

        return tmp

    dat_app_names = ['dat_app_%s' % str(x) for x in list(range(2, 8))]
    dat_app_left = reduce(lambda x, y: x.append(y),
                          (read_app(filename) for filename in dat_app_names))

    dat_edge_1 = the_dat_edge.append(dat_edge_left)  # 把第一个数据和剩下的数据合并起来
    dat_app_1 = the_dat_app.append(dat_app_left)  # 把第一个数据和剩下的数据合并起来

    return dat_edge_1, dat_app_1


@time_pass
def dummy_symbol(the_dat_symbol):
    """
    1. 把dat_symbol的一级分类的所有可能取值all_first挑出来，
    2. 然后得到：每一个id的'symbol'列里的一级分类是否包含all_first,得到0-1向量
    3. 同样的处理一级分类和二级分类的组合，单独处理二级分类我觉得没这个必要了
    """

    def get_first(string):
        f_s = string.split(',')
        first = set(list(map(lambda x: x.split('_')[0], f_s)))

        return first

    def get_second(string):
        f_s = string.split(',')
        second = set(list(map(lambda x: x.split('_')[1], f_s)))

        return second

    def get_both(string):
        f_s = string.split(',')
        return set(f_s)

    def is_in_first(string):
        f_s = string.split(',')
        first = set(list(map(lambda x: x.split('_')[0], f_s)))
        is_in = list(map(lambda x: x in first, all_first))

        return is_in

    def is_in_second(string):
        f_s = string.split(',')
        second = set(list(map(lambda x: x.split('_')[1], f_s)))
        is_in = list(map(lambda x: x in second, all_second))

        return is_in

    def is_in_both(string):
        f_s = set(string.split(','))
        is_in = list(map(lambda x: x in f_s, all_both))

        return is_in

    tmp = the_dat_symbol['symbol'].unique()

    # 获取所有的一级分类和一二级分类
    all_first = reduce(lambda x, y: x.union(y),
                       map(get_first, tmp))
    all_second = reduce(lambda x, y: x.union(y),
                        map(get_second, tmp))
    all_both = reduce(lambda x, y: x.union(y),
                      map(get_both, tmp))

    # 得到每个id的0-1向量，存储成DataFrame
    in_first_0 = pd.DataFrame(list(map(is_in_first, the_dat_symbol['symbol'])),
                              columns=all_first)
    in_second_0 = pd.DataFrame(list(map(is_in_second, the_dat_symbol['symbol'])),
                               columns=all_second)
    in_both_0 = pd.DataFrame(list(map(is_in_both, the_dat_symbol['symbol'])),
                             columns=all_both)
    in_first_1 = pd.concat([the_dat_symbol[['id']], in_first_0], axis=1) + 0
    in_second_1 = pd.concat([the_dat_symbol[['id']], in_second_0], axis=1) + 0
    in_both_1 = pd.concat([the_dat_symbol[['id']], in_both_0], axis=1) + 0

    return in_first_1, in_second_1, in_both_1


@time_pass
def deal_dat_edge(data_all):
    """
    1. 把dat_edge处理好,运行dat_edge.head(15)，就会发现需要把第10行这类数据和其他数据分开，
    2. 分为dat_edge_single，dat_edge_multi
    3. 然后把dat_edge_multi处理成跟dat_edge_single一样的格式，叫做dat_edge_multi_new
    4. 然后把两者合并成为dat_edge_new
    5. 之后经由dat_edge_split_2把info分为三个部分：['date', 'times', 'weight']
    """
    length = list(map(len, map(lambda x: x.split(','), data_all['info'])))

    dat_edge_single = data_all[np.array(length) == 1]
    dat_edge_multi = data_all[np.array(length) > 1]

    def dat_edge_split(i):
        i_info = dat_edge_multi.iloc[i]
        string = i_info['info']
        s = string.split(',')
        result = pd.DataFrame({'info': s,
                               'from_id': [i_info['from_id']] * len(s),
                               'to_id': [i_info['to_id']] * len(s),
                               'id': [i_info['id']] * len(s)})
        return result[['id', 'from_id', 'to_id', 'info']]

    all_df = map(dat_edge_split, range(len(dat_edge_multi)))
    dat_edge_multi_new = pd.concat(all_df, axis=0, ignore_index=True)  # 比较慢

    dat_edge_new = pd.concat([dat_edge_single, dat_edge_multi_new], axis=0, ignore_index=True)
    # dat_edge_new = dat_edge_single.append(dat_edge_multi_new, ignore_index=True)

    @time_pass
    def dat_edge_split_2(data):
        def split(string):
            date, left = string.split(':')
            times, weight = left.split('_')

            return date, times, weight

        info_df = pd.DataFrame(list(map(split, data['info'])),
                               columns=['date', 'times', 'weight'])
        data_new_2 = pd.concat([data[['id', 'from_id', 'to_id']], info_df], axis=1)

        return data_new_2

    dat_edge_new_2 = dat_edge_split_2(dat_edge_new)

    return dat_edge_new_2


@time_pass
def deal_edge(the_sample_train, the_dat_edge):
    """
    提取出每一个用户的“流出”特征: 向量长度、times之和、times的中位数、最小值、最大值
    weight之和、weight的中位数、最小值、最大值，这样就用9个特征提取出了“流出”特征
    """
    col_names = (['length', 'unique_count', 'times_sum', 'weight_sum']
                 + ['dup_ratio_left', 'dup_ratio_1', 'dup_ratio_2', 'dup_ratio_3', 'dup_ratio_4', 'dup_ratio_5']
                 + ['times_left', 'times_1', 'times_2', 'times_3', 'times_4', 'times_5',
                    'times_6', 'times_7', 'times_8', 'times_9', 'times_10']
                 + ['times_min', 'times_25', 'times_median', 'times_75', 'times_max']
                 + ['weight_min', 'weight_25', 'weight_median', 'weight_75', 'weight_max']
                 + ['times_up_out_ratio', 'times_low_out_ratio']
                 + ['weight_up_out_ratio', 'weight_low_out_ratio']
                 + ['time_sign_trend', 'time_abs', 'weight_sign_trend', 'weight_abs']
                 + ['times_2017_11', 'times_2017_12', 'times_2017_13']
                 + ['weight_2017_11', 'weight_2017_12', 'weight_2017_13']
                 + ['date_unique_count', 'date_min', 'date_max', 'days_gap']
                 + ['latest_times', 'latest_peoples', 'latest_weights', 'multi_ratio'])

    sample_dat_edge_from = pd.merge(the_sample_train, the_dat_edge,
                                    left_on='id', right_on='from_id',
                                    how='inner')
    dat_edge_from = deal_dat_edge(sample_dat_edge_from)
    dat_edge_from['times'] = list(map(int, dat_edge_from['times']))
    dat_edge_from['weight'] = list(map(float, dat_edge_from['weight']))

    unique_id_from = np.unique(dat_edge_from['id'])
    feature_9_1 = list(map(lambda x: cal_9_feature(x, dat_edge_from, 'to_id'), unique_id_from))
    df_feature_9_1 = pd.DataFrame(feature_9_1, columns=['out_%s' % x for x in col_names])
    df_feature_9_1['id'] = unique_id_from

    # 提取出每一个用户的“流入”特征，类似上面的，可以提取出9个“流入”特征
    sample_dat_edge_to = pd.merge(the_sample_train, the_dat_edge,
                                  left_on='id', right_on='to_id',
                                  how='inner')
    dat_edge_to = deal_dat_edge(sample_dat_edge_to)
    dat_edge_to['times'] = list(map(int, dat_edge_to['times']))
    dat_edge_to['weight'] = list(map(float, dat_edge_to['weight']))

    unique_id_to = np.unique(dat_edge_to['id'])

    feature_9_2 = list(map(lambda x: cal_9_feature(x, dat_edge_to, 'from_id'), unique_id_to))
    df_feature_9_2 = pd.DataFrame(feature_9_2, columns=['in_%s' % x for x in col_names])
    df_feature_9_2['id'] = unique_id_to

    unique_id_both = list(set(unique_id_from).union(set(unique_id_to)))
    feature_9_3 = list(map(lambda x: cal_both(x, dat_edge_from, dat_edge_to), unique_id_both))
    df_feature_9_3 = pd.DataFrame(feature_9_3, columns=['both_%s' % x for x in col_names])
    df_feature_9_3['id'] = unique_id_both

    # 接下来需要把df_feature_9_1和df_feature_9_2, df_feature_9_3以并联方式merge起来，
    # 然后左连接到sample_train上
    the_df_feature_18 = reduce(lambda x, y: pd.merge(x, y, on='id', how='outer'),
                               [df_feature_9_1, df_feature_9_2, df_feature_9_3])
    the_df_feature_18['net_in'] = the_df_feature_18['in_weight_sum'] - the_df_feature_18['out_weight_sum']
    the_df_feature_18['out_unique_ratio'] = the_df_feature_18['out_unique_count']/the_df_feature_18['out_length']
    the_df_feature_18['in_unique_ratio'] = the_df_feature_18['in_unique_count'] / the_df_feature_18['in_length']
    the_df_feature_18['out_longer_5'] = (the_df_feature_18['out_length'] > 5) + 0
    the_df_feature_18['out_longer_10'] = (the_df_feature_18['out_length'] > 10) + 0
    the_df_feature_18['in_longer_5'] = (the_df_feature_18['in_length'] > 5) + 0
    the_df_feature_18['in_longer_10'] = (the_df_feature_18['in_length'] > 10) + 0
    the_df_feature_18['both_longer_10'] = (the_df_feature_18['both_length'] > 10) + 0
    the_df_feature_18['both_longer_20'] = (the_df_feature_18['both_length'] > 20) + 0

    # 下面时看了视频解析之后，确认数据为通话数据之后生成的特征
    # 先算出总的联系人数目和总的通话时间
    the_df_feature_18['sum_degree'] = the_df_feature_18['out_unique_count'] + the_df_feature_18['in_unique_count']
    the_df_feature_18['sum_weight'] = the_df_feature_18['out_weight_sum'] + the_df_feature_18['in_weight_sum']

    # 进入、出去、总的closeness
    the_df_feature_18['out_closeness'] = the_df_feature_18['out_weight_sum']/the_df_feature_18['out_unique_count']
    the_df_feature_18['in_closeness'] = the_df_feature_18['in_weight_sum'] / the_df_feature_18['in_unique_count']
    the_df_feature_18['sum_closeness'] = the_df_feature_18['sum_weight'] / the_df_feature_18['sum_degree']

    return the_df_feature_18


@time_pass
def get_apps_dummy(data):
    """
    把dat_app里用户装的app信息0-1化
    1. 先把所有的app得到：all_apps
    2. 然后得到长度为all_apps的0-1向量
    """
    all_apps = set()
    for string in data['apps']:
        apps = string.split(',')
        all_apps = all_apps.union(set(apps))
    all_apps = list(all_apps)

    def is_in_all_apps(x):
        xs = x.split(',')
        xs = set(xs)
        app_vec = list(map(lambda app: app in xs, all_apps))

        return app_vec

    apps_dummy_0 = list(map(is_in_all_apps, data['apps']))
    apps_dummy_1 = pd.DataFrame(apps_dummy_0, columns=all_apps)
    apps_dummy_2 = pd.concat([data[['id']], apps_dummy_1], axis=1)

    return apps_dummy_2


def outlier_ratio(the_series):
    """利用箱线图来检测异常率"""
    the_median = np.median(the_series)
    q1 = np.percentile(the_series, 20)
    q3 = np.percentile(the_series, 70)
    iqr = q3 - q1
    up_bound = the_median + 1.5*iqr
    low_bound = the_median - 1.5*iqr

    up_out_count = sum(the_series > up_bound)
    low_out_count = sum(the_series < low_bound)

    the_up_out_ratio = up_out_count/len(the_series)
    the_low_out_ratio = low_out_count/len(the_series)

    return the_up_out_ratio, the_low_out_ratio


def cal_dup_ratio(series, n):
    """计算一个人给另一个人不同月份转的频次"""
    the_dup_ratio = np.zeros(6)
    tmp = pd.Series(series.value_counts().values).value_counts()
    for j in tmp.index:
        if j > 5:
            continue
        else:
            the_dup_ratio[j] = tmp[j] / n

    the_dup_ratio[0] = 1 - np.sum(the_dup_ratio)

    return the_dup_ratio


def cal_both(the_id, data_from, data_to):
    """
    data_from: 取值为dat_edge_from
    data_to: 取值为dat_edge_to
    """
    the_id_dat_from = data_from[data_from['id'] == the_id].copy()
    the_id_dat_to = data_to[data_to['id'] == the_id].copy()

    the_id_dat_from['date'] = pd.to_datetime(the_id_dat_from['date'])
    the_id_dat_from.sort_values(by='date', inplace=True)
    if the_id_dat_from['times'].dtype == 'object':
        the_id_dat_from['times'] = list(map(eval, the_id_dat_from['times']))

    the_id_dat_to['date'] = pd.to_datetime(the_id_dat_to['date'])
    the_id_dat_to.sort_values(by='date', inplace=True)
    if the_id_dat_to['times'].dtype == 'object':
        the_id_dat_to['times'] = list(map(eval, the_id_dat_to['times']))

    the_id_dat = pd.concat([the_id_dat_from, the_id_dat_to], axis=0)

    agg_the_id_dat_from = the_id_dat_from.groupby('to_id')['times', 'weight'].sum()
    agg_the_id_dat_to = the_id_dat_to.groupby('from_id')['times', 'weight'].sum()

    agg_dat = pd.concat([agg_the_id_dat_from, agg_the_id_dat_to], axis=0)

    value_counts_pre = the_id_dat_from['to_id'].append(the_id_dat_to['from_id'])
    dup_ratio = cal_dup_ratio(value_counts_pre, len(agg_dat))

    # 开始计算特征
    length = len(value_counts_pre)
    unique_count = len(agg_dat)

    multi_ratio = (length - unique_count) / length

    times = agg_dat['times']
    times_sum = np.sum(times)

    # 计算一个人给另一个人转账次数的value_counts
    times_value_counts = np.zeros(11)
    tmp_2 = times.value_counts() / len(times)
    for i in tmp_2.index:
        if i > 10:
            continue
        else:
            times_value_counts[i] = tmp_2[i]
    times_value_counts[0] = 1 - np.sum(times_value_counts)

    times_min = np.min(times)
    times_25 = np.percentile(times, 25)
    times_median = np.median(times)
    times_75 = np.percentile(times, 75)
    times_max = np.max(times)

    times_up_out_ratio, times_low_out_ratio = outlier_ratio(times)

    weight = agg_dat['weight']
    weight_sum = np.sum(weight)
    weight_min = np.min(weight)
    weight_25 = np.percentile(weight, 25)
    weight_median = np.median(weight)
    weight_75 = np.percentile(weight, 75)
    weight_max = np.max(weight)

    weight_up_out_ratio, weight_low_out_ratio = outlier_ratio(weight)

    agg_date = the_id_dat.groupby('date')['times', 'weight'].sum()
    if len(agg_date) > 1:
        time_sign_trend = np.sign(times.diff()[1:]).sum()/(len(agg_date)-1)
        time_abs = np.abs(times.diff()[1:]).sum()/(len(agg_date-1))
        weight_sign_trend = np.sign(weight.diff()[1:]).sum()/(len(agg_date)-1)
        weight_abs = np.abs(weight.diff()[1:]).sum()/(len(agg_date-1))
    else:
        time_sign_trend, time_abs, weight_sign_trend, weight_abs = [np.nan]*4

    prior_date = pd.to_datetime(['2017-11', '2017-12', '2018-01'])
    exist_date = set(agg_date.index)

    prior_times = np.zeros(3)
    for i in range(3):
        if prior_date[i] in exist_date:
            prior_times[i] = agg_date['times'][prior_date[i]]

    prior_weight = np.zeros(3)
    for i in range(3):
        if prior_date[i] in exist_date:
            prior_weight[i] = agg_date['weight'][prior_date[i]]

    date = agg_date.index
    begin_date = pd.to_datetime('2017-11')
    date_unique_count = len(np.unique(date))
    date_min = (np.min(date) - begin_date).days
    date_max = (np.max(date) - begin_date).days
    days_gap = date_max - date_min
    if times_sum != 0:
        if np.max(date) == pd.to_datetime('2018-01'):
            latest_times_ratio = np.sum(agg_date['times'][agg_date.index == np.max(date)])/times_sum
        else:
            latest_times_ratio = 0
    else:
        latest_times_ratio = 1

    latest_peoples = len(value_counts_pre[the_id_dat['date'] == np.max(date)].unique())
    latest_peoples_ratio = latest_peoples/unique_count
    if weight_sum != 0:
        if np.max(date) == pd.to_datetime('2018-01'):
            latest_weights_ratio = np.sum(the_id_dat['weight'][the_id_dat['date'] == np.max(date)])/weight_sum
        else:
            latest_weights_ratio = 0
    else:
        latest_weights_ratio = 1

    result = ([length, unique_count, times_sum, weight_sum]
              + list(dup_ratio)
              + list(times_value_counts)
              + [times_min, times_25, times_median, times_75, times_max]
              + [weight_min, weight_25, weight_median, weight_75, weight_max]
              + [times_up_out_ratio, times_low_out_ratio, weight_up_out_ratio, weight_low_out_ratio]
              + [time_sign_trend, time_abs, weight_sign_trend, weight_abs]
              + list(prior_times)
              + list(prior_weight)
              + [date_unique_count, date_min, date_max, days_gap, latest_times_ratio, latest_peoples_ratio,
                 latest_weights_ratio, multi_ratio])

    return result


def cal_9_feature(the_id, the_dat_edge_from, string):
    """string: 取值为'from_id'或'to_id'"""
    the_id_dat = the_dat_edge_from[the_dat_edge_from['id'] == the_id].copy()
    the_id_dat['date'] = pd.to_datetime(the_id_dat['date'])
    the_id_dat.sort_values(by='date', inplace=True)
    if the_id_dat['times'].dtype == 'object':
        the_id_dat['times'] = list(map(eval, the_id_dat['times']))

    agg_the_id_dat = the_id_dat.groupby(string).sum()

    dup_ratio = cal_dup_ratio(the_id_dat[string], len(agg_the_id_dat))

    # 开始计算特征
    length = len(the_id_dat)
    unique_count = len(agg_the_id_dat)

    multi_ratio = (length - unique_count)/length

    times = agg_the_id_dat['times']
    times_sum = np.sum(times)

    # 计算一个人给另一个人转账次数的value_counts
    times_value_counts = np.zeros(11)
    tmp_2 = times.value_counts()/len(times)
    for i in tmp_2.index:
        if i > 10:
            continue
        else:
            times_value_counts[i] = tmp_2[i]
    times_value_counts[0] = 1 - np.sum(times_value_counts)

    times_min = np.min(times)
    times_25 = np.percentile(times, 25)
    times_median = np.median(times)
    times_75 = np.percentile(times, 75)
    times_max = np.max(times)

    times_up_out_ratio, times_low_out_ratio = outlier_ratio(times)

    weight = agg_the_id_dat['weight']
    weight_sum = np.sum(weight)
    weight_min = np.min(weight)
    weight_25 = np.percentile(weight, 25)
    weight_median = np.median(weight)
    weight_75 = np.percentile(weight, 75)
    weight_max = np.max(weight)

    weight_up_out_ratio, weight_low_out_ratio = outlier_ratio(weight)

    agg_date = the_id_dat.groupby('date').sum()
    if len(agg_date) > 1:
        time_sign_trend = np.sign(agg_date['times'].diff()[1:]).sum()/(len(agg_date)-1)
        time_abs = np.abs(agg_date['times'].diff()[1:]).sum()/(len(agg_date-1))
        weight_sign_trend = np.sign(agg_date['weight'].diff()[1:]).sum()/(len(agg_date)-1)
        weight_abs = np.abs(agg_date['weight'].diff()[1:]).sum()/(len(agg_date-1))
    else:
        time_sign_trend, time_abs, weight_sign_trend, weight_abs = [np.nan]*4

    prior_date = pd.to_datetime(['2017-11', '2017-12', '2018-01'])
    exist_date = set(agg_date.index)

    prior_times = np.zeros(3)
    for i in range(3):
        if prior_date[i] in exist_date:
            prior_times[i] = agg_date['times'][prior_date[i]]

    prior_weight = np.zeros(3)
    for i in range(3):
        if prior_date[i] in exist_date:
            prior_weight[i] = agg_date['weight'][prior_date[i]]

    date = agg_date.index
    begin_date = pd.to_datetime('2017-11')
    date_unique_count = len(np.unique(date))
    date_min = (np.min(date) - begin_date).days
    date_max = (np.max(date) - begin_date).days
    days_gap = date_max - date_min
    if times_sum != 0:
        if np.max(date) == pd.to_datetime('2018-01'):
            latest_times_ratio = np.sum(agg_date['times'][agg_date.index == np.max(date)])/times_sum
        else:
            latest_times_ratio = 0
    else:
        latest_times_ratio = 1

    latest_peoples = len(the_id_dat[string][the_id_dat['date'] == np.max(date)].unique())
    latest_peoples_ratio = latest_peoples/unique_count
    if weight_sum != 0:
        if np.max(date) == pd.to_datetime('2018-01'):
            latest_weights_ratio = np.sum(the_id_dat['weight'][the_id_dat['date'] == np.max(date)])/weight_sum
        else:
            latest_weights_ratio = 0
    else:
        latest_weights_ratio = 1

    result = ([length, unique_count, times_sum, weight_sum]
              + list(dup_ratio)
              + list(times_value_counts)
              + [times_min, times_25, times_median, times_75, times_max]
              + [weight_min, weight_25, weight_median, weight_75, weight_max]
              + [times_up_out_ratio, times_low_out_ratio, weight_up_out_ratio, weight_low_out_ratio]
              + [time_sign_trend, time_abs, weight_sign_trend, weight_abs]
              + list(prior_times)
              + list(prior_weight)
              + [date_unique_count, date_min, date_max, days_gap, latest_times_ratio, latest_peoples_ratio,
                 latest_weights_ratio, multi_ratio])

    col_names = (['length', 'unique_count', 'times_sum', 'weight_sum']
                 + ['dup_ratio_left', 'dup_ratio_1', 'dup_ratio_2', 'dup_ratio_3', 'dup_ratio_4', 'dup_ratio_5']
                 + ['times_left', 'times_1', 'times_2', 'times_3', 'times_4', 'times_5',
                    'times_6', 'times_7', 'times_8', 'times_9', 'times_10']
                 + ['times_min', 'times_25', 'times_median', 'times_75', 'times_max']
                 + ['weight_min', 'weight_25', 'weight_median', 'weight_75', 'weight_max']
                 + ['times_up_out_ratio', 'times_low_out_ratio']
                 + ['weight_up_out_ratio', 'weight_low_out_ratio']
                 + ['time_sign_trend', 'time_abs', 'weight_sign_trend', 'weight_abs']
                 + ['times_2017_11', 'times_2017_12', 'times_2017_13']
                 + ['weight_2017_11', 'weight_2017_12', 'weight_2017_13']
                 + ['date_unique_count', 'date_min', 'date_max', 'days_gap']
                 + ['latest_times', 'latest_peoples', 'latest_weights', 'multi_ratio'])
    return result


if __name__ == '__main__':
    input_path = './'
    sample_train = pd.read_table(os.path.join(input_path, "open_data/sample_train.txt"))  # 训练集约1.9万
    dat_risk = pd.read_table(os.path.join(input_path, "open_data/dat_risk.txt"))  # 用户疑似风险行为数据
    dat_symbol = pd.read_table(os.path.join(input_path, "open_data/dat_symbol.txt"))  # 用户分类数据
    valid_id = pd.read_table(os.path.join(input_path, "open_data/valid_id.txt"))  # 验证集
    test_id = pd.read_table(os.path.join(input_path, "open_data/test_id.txt"))  # 测试集
    dat_edge = pd.read_table(os.path.join(input_path, "open_data/dat_edge/dat_edge_1"))  # 用户关联数据
    dat_app = pd.read_table(os.path.join(input_path, "open_data/dat_app/dat_app_1"),
                            header=None,
                            names=['id', 'apps'])  # 用户app数据

    all_id = pd.concat([sample_train[['id']], valid_id, test_id], axis=0)

    if 3 > 2:  # 最后再运行这段代码, 测试的时候先在小数据集上进行测试
        dat_edge, dat_app = complete_data(dat_edge, dat_app, input_path)  # 9分9秒

    df_feature_18 = deal_edge(all_id, dat_edge)  # 累计29分钟左右

    # 下面只选出有标签的样本进行dummy
    sample_dat_app = pd.merge(all_id, dat_app, on='id', how="inner")  # 1分钟左右
    apps_dummy = get_apps_dummy(sample_dat_app)  # 36秒
    in_first, in_second, in_both = dummy_symbol(dat_symbol)  # 50秒

    sample_in_first = pd.merge(all_id, in_first, on='id', how="inner")
    sample_in_second = pd.merge(all_id, in_second, on='id', how="inner")
    sample_in_both = pd.merge(all_id, in_both, on='id', how="inner")
    sample_dat_risk = pd.merge(all_id[['id']], dat_risk.drop('a_cnt', axis=1), on='id', how="inner")

    # 把清洗后的数据输出，在下一个文件feature_engineer.py里进行特征工程
    all_id.to_csv('./output/sample_valid.csv', index=False)
    df_feature_18.to_csv('./output/df_feature_18.csv', index=False)
    sample_dat_risk.to_csv('./output/sample_dat_risk.csv', index=False)
    sample_in_first.to_csv('./output/sample_in_first.csv', index=False)
    sample_in_second.to_csv('./output/sample_in_second.csv', index=False)
    sample_in_both.to_csv('./output/sample_in_both.csv', index=False)
    apps_dummy.to_csv('./output/apps_dummy.csv', index=False)

