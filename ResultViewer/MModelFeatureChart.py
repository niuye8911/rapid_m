import json


def getSel(feature, selected):
    first = feature + '-1'
    second = feature + '-2'
    if first in selected and second in selected:
        return 'A'
    if first in selected:
        return '1'
    if second in selected:
        return '2'
    else:
        return '-'


def getHeader(features):
    line = [' '] + features
    line += ['Model', 'isPoly']
    return ','.join(line) + '\n'


def genChart(m_file, output_file):
    with open(m_file, 'r') as file:
        model = json.load(file)
        features = model['features']
        metas = model['model_params']['Meta']
        output = open(output_file, 'w')
        # write header
        output.write(getHeader(features))
        for feature in features:
            line = [feature]
            meta = metas[feature]
            feature_sel = list(
                map(lambda f: getSel(f, meta['features']), features))
            line = line + feature_sel
            line.append(meta['name'])
            line.append(str(meta['isPoly']))
            # write the line
            output.write(','.join(line) + '\n')
        output.close()


genChart('../examples/example_machine_empty.json', '../feature_selection.csv')
