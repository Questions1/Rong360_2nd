
import pandas as pd

from functools import reduce


if __name__ == '__main__':
    son = pd.read_csv('./output/son.csv')
    father = pd.read_csv('./output/father.csv')

    one_step_id = pd.DataFrame({'id': list(set(son['to_id']).union(set(father['from_id'])))})

    graph_feature_big = pd.read_csv('./output/graph_feature_big.csv')
    feature_7_df = pd.read_csv('./output/feature_7_df.csv')
    one_step_apps_dummy = pd.read_csv('./output/one_step_apps_dummy.csv')
    one_step_dummy_symbol = pd.read_csv('./output/one_step_dummy_symbol.csv')
    dat_risk = pd.read_table('./open_data/dat_risk.txt')

    one_step_id_feature_agg = reduce(lambda x, y: pd.merge(x, y, on='id', how='left'),
                                     [one_step_id, graph_feature_big, feature_7_df,
                                      one_step_apps_dummy, dat_risk, one_step_dummy_symbol])
    one_step_id_feature_agg.to_csv('./output/one_step_id_feature_agg.csv', index=False)
