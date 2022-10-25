import evaluate
import jsonlines
from datasets import load_dataset
from evaluator.bleu import _bleu_without_file

dataset = load_dataset("neulab/conala")
with open('conala_train.txt', 'r') as f:
    lines = f.readlines()
    list_of_lines = []
    tmp_lines = []
    for idx, line in enumerate(lines):
        tmp_lines.append(line)
        if idx > 0 and idx % 16 == 15:
            list_of_lines.append(tmp_lines)
            tmp_lines = []   

sacrebleu = evaluate.load("sacrebleu")

list_of_candidates = []
for idx, data in enumerate(dataset['train']):
    target = data['snippet'].replace('\n', ' ')
    for i in range(len(list_of_lines[idx])):
        pred = list_of_lines[idx][i].replace('\n', ' ')
        results = sacrebleu.compute(predictions=[pred], references=[[target]])
        score = results['score']
        list_of_lines[idx][i] = (list_of_lines[idx][i], score)

brio_dataset = []
for idx, data in enumerate(dataset['train']):
    data_dict = {}
    data_dict['intent'] = data['intent']
    data_dict['rewritten_intent'] = data['rewritten_intent']
    data_dict['snippet'] = data['snippet']
    data_dict['candidates'] = []
    for line in list_of_lines[idx]:
        data_dict['candidates'].append(line)
    brio_dataset.append(data_dict)

with jsonlines.open('conala_train_brio.jsonl', 'w') as writer:
    writer.write_all(brio_dataset)
