from src.data import parse_episodes_from_index
import json

train_episodes_1 = parse_episodes_from_index("data/train.json", "data/train_1_doc_indices.json", tokenizer=None, markers=False, cache=None, no_processing=True)

train_episodes_3 = parse_episodes_from_index("data/train.json", "data/train_3_doc_indices.json", tokenizer=None, markers=False, cache=None, no_processing=True)

dev_episodes_1 = parse_episodes_from_index("data/dev.json", "data/dev_1_doc_indices.json", tokenizer=None, markers=False, cache=None, no_processing=True)

dev_episodes_3 = parse_episodes_from_index("data/dev.json", "data/dev_1_doc_indices.json", tokenizer=None, markers=False, cache=None, no_processing=True)

with open('train_1_doc.json', 'w') as fout:
    json.dump(train_episodes_1, fout)
with open('train_3_doc.json', 'w') as fout:
    json.dump(train_episodes_3, fout)

with open('dev_1_doc.json', 'w') as fout:
    json.dump(dev_episodes_1, fout)
with open('dev_3_doc.json', 'w') as fout:
    json.dump(dev_episodes_3, fout)
