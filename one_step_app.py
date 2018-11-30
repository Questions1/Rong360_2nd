
import numpy as np
import pandas as pd


def read_app(file_path):
    reader = pd.read_table(file_path, header=None, chunksize=10000)
    data = pd.concat(reader, axis=0, ignore_index=True)
    data.columns = ['id', 'apps']

    return data


def get_apps_dummy(data):
    """
    把dat_app里用户装的app信息0-1化
    1. 读取需要的104个app：app_104_list
    2. 然后得到长度为‘len(app_104_list)’的0-1向量
    """
    def is_in_all_apps(x):
        xs = x.split(',')
        xs = set(xs)
        app_vec = list(map(lambda app: int(app in xs), app_66))

        return app_vec

    apps_dummy_0 = list(map(is_in_all_apps, data['apps']))
    apps_dummy_1 = pd.DataFrame(apps_dummy_0, columns=app_66)
    apps_dummy_2 = pd.concat([data[['id']], apps_dummy_1], axis=1)

    return apps_dummy_2


if __name__ == '__main__':
    input_path = './'
    sample_train = pd.read_table('./open_data/sample_train.txt')  # 训练集约1.9万
    valid_id = pd.read_table('./open_data/valid_id.txt')  # 验证集
    test_id = pd.read_table('./open_data/test_id.txt')  # 测试集

    dat_app = pd.concat([read_app('./open_data/dat_app/dat_app_%s' % x) for x in range(1, 8)],
                        axis=0, ignore_index=True)
    important_feature_app = pd.read_csv('./output/important_feature_app.csv')
    app_66 = [str(x) for x in important_feature_app['feature']]

    son = pd.read_csv('./output/son.csv')
    father = pd.read_csv('./output/father.csv')

    all_id = pd.concat([sample_train[['id']], valid_id[['id']], test_id[['id']]], axis=0)
    whole_id = list(set(son['to_id']).union(set(father['from_id'])).union(all_id['id']))
    one_step_id = pd.DataFrame({'id': whole_id})

    one_step_app = pd.merge(one_step_id, dat_app, on='id')
    one_step_apps_dummy = get_apps_dummy(one_step_app)
    one_step_apps_dummy.to_csv('./output/one_step_apps_dummy.csv', index=False)
