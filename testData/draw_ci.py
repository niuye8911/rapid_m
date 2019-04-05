import matplotlib.pyplot as plt

names = []
ci_lows = []
ci_upps = []
means = []

with open('./ci.csv') as f:
    for line in f:
        items = line.split()
        names.append(items[0])
        means.append(float(items[1]))
        ci_lows.append(float(items[1])-float(items[2]))
        ci_upps.append(float(items[3])-float(items[1]))

plt.errorbar(xrange(len(names)), means, yerr=[ci_lows,ci_upps],fmt='o')
plt.xticks(xrange(len(names)), names,fontsize='15')
plt.ylabel('Prediction MRE')
plt.ylim(0.0,0.10)
plt.savefig('./CI.png')
