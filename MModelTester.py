from Classes.MModel import *

DATA = '/home/liuliu/Research/rapid_m_backend_server/testData/assemble/assemble.csv'
MACHINE_FILE = '/home/liuliu/Research/rapid_m_backend_server/examples/example_machine_empty.json'

m_model = MModel(MACHINE_FILE)
features = list(map(lambda x: x[:-2], m_model.features))

df = pd.read_csv(DATA)
columns = list(df.columns.values)
length = len(columns)

column1 = columns[0:int(length / 3)]
column2 = columns[int(length / 3):int(length * 2 / 3)]
column3 = columns[int(length * 2 / 3):]


# test the first 5 rows


def formatEnv(env, features, postfix):
    result = []
    for feature in features:
        if feature == 'MEM':
            result.append(env['READ' + postfix] + env['WRITE' + postfix])
        elif feature == 'INST':
            result.append(env['ACYC' + postfix] / env['INST' + postfix])
        elif feature == 'INSTnom' or feature == 'PhysIPC%':
            result.append(env[feature + postfix] / 100.0)
        else:
            result.append(env[feature + postfix])
    return list(map(lambda x: float(x), result))


i = 0
for index, row in df.iterrows():
    input1 = row[column1]
    input2 = row[column2]
    output = formatEnv(row[column3], features, '-C')
    predict = m_model.predict(
        formatEnv(input1, features, '-1'),
        formatEnv(input2, features, '-2'))
    print('predict:')
    print(predict)
    print('real:')
    print(output)
    i += 1
    if i == 1:
        break
