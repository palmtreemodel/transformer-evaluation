import json
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt

results = {}
for model in ['Bert', 'stateformer', "Trex", "jTrans", "Bert_JTP", 'Bert_CWP', 'Bert_DUP', "Bert_GSM"]:
    results[model] = []
    for i in range(10):
        with open("results/{}/{}_score.log".format(model, i), "r") as f:
            map = f.readline()
            map = map[10:16].replace('}', '0')
            results[model].append(float(map))
# print(results.values())
            
for model in ['stateformer', "Trex", "jTrans", "Bert_JTP", 'Bert_CWP', 'Bert_DUP', "Bert_GSM"]:
    print(results['Bert'],)
    res = ttest_ind(results['Bert'], results[model])
    print('BERT', model, res)


plt.figure(figsize=(5, 3))
plt.boxplot(list(results.values()), labels=['BERT', 'StateFormer', "Trex", "jTrans", "BERT-JTP", 'BERT-CWP', 'BERT-DUP', "BERT-GSM"])
plt.xticks(rotation=25)
plt.tight_layout()
plt.savefig('wft.pdf')

