import pandas as pd


def formatEnv_df(env, filters=[], POSTFIX='', REMOVE_POSTFIX=True):
    ''' input a df, return a df with scaled and filtered data'''
    result_df = pd.DataFrame()
    post_len = len(POSTFIX)
    R_POST = "" if REMOVE_POSTFIX else POSTFIX
    # first copy the data with new name
    for col in env.columns:
        feature = col
        if post_len > 0:
            feature = col[:-post_len]
        # check if the feature is needed
        if feature not in filters and not filters == []:
            continue
        r_col = feature if REMOVE_POSTFIX else col
        result_df[r_col] = env[col]
    # then fix the data
    if 'MEM' + POSTFIX not in env.columns:
        result_df['MEM' +
                  R_POST] = env['READ' + POSTFIX] + env['WRITE' + POSTFIX]
        result_df['INST' +
                  R_POST] = env['ACYC' + POSTFIX] / env['INST' + POSTFIX]
        result_df['INSTnom%' + R_POST] = env['INSTnom%' + POSTFIX] / 100.0
        result_df['PhysIPC%' + R_POST] = env['PhysIPC%' + POSTFIX] / 100.0
    return result_df


def formatEnv(env, features, POSTFIX=''):
    result = []
    for feature in features:
        if feature == 'MEM':
            result.append(env['READ' + POSTFIX] + env['WRITE' + POSTFIX])
        elif feature == 'INST':
            result.append(env['ACYC' + POSTFIX] / env['INST' + POSTFIX])
        elif feature == 'INSTnom%' or feature == 'PhysIPC%':
            result.append(env[feature + POSTFIX] / 100.0)
        else:
            result.append(env[feature + POSTFIX])
    return list(map(lambda x: float(x), result))


def env_to_frame(env, features):
    result = OrderedDict()
    i = 0
    for feature in features:
        result[feature] = [env[i]]
        i += 1
    return pd.DataFrame(data=result)
