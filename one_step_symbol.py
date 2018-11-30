
import pandas as pd
from functools import reduce


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


if __name__ == '__main__':
    dat_symbol = pd.read_table('./open_data/dat_symbol.txt')  # 用户分类数据
    important_feature_symbol = pd.read_csv('./output/important_feature_symbol.csv')

    son = pd.read_csv('./output/son.csv')
    father = pd.read_csv('./output/father.csv')

    one_step_id = pd.DataFrame({'id': list(set(son['to_id']).union(set(father['from_id'])))})

    one_step_symbol = pd.merge(one_step_id, dat_symbol, on='id')
    in_first, in_second, in_both = dummy_symbol(one_step_symbol)

    one_step_dummy_symbol = reduce(lambda x, y: pd.merge(x, y, on='id', how='left'),
                                   [in_first, in_second, in_both])

    one_step_dummy_symbol = one_step_dummy_symbol[['id'] + list(important_feature_symbol['feature'])]
    one_step_dummy_symbol.to_csv('./output/one_step_dummy_symbol.csv', index=False)

