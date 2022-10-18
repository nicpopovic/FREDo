from src.data import parse_episodes_from_index
import json


indomain_test_episodes_1 = parse_episodes_from_index("data/test_docred.json", "data/test_in_domain_1_doc_indices.json", tokenizer=None, markers=False, cache=None, no_processing=True)

indomain_test_episodes_3 = parse_episodes_from_index("data/test_docred.json", "data/test_in_domain_3_doc_indices.json", tokenizer=None, markers=False, cache=None, no_processing=True)

crossdomain_test_episodes_1 = parse_episodes_from_index("data/test_scierc.json", "data/test_cross_domain_1_doc_indices.json", tokenizer=None, markers=False, cache=None, no_processing=True)

crossdomain_test_episodes_3 = parse_episodes_from_index("data/test_scierc.json", "data/test_cross_domain_3_doc_indices.json", tokenizer=None, markers=False, cache=None, no_processing=True)

with open('test_in_domain_1_doc.json', 'w') as fout:
    json.dump(indomain_test_episodes_1, fout)
with open('test_in_domain_3_doc.json', 'w') as fout:
    json.dump(indomain_test_episodes_3, fout)

with open('test_cross_domain_1_doc.json', 'w') as fout:
    json.dump(crossdomain_test_episodes_1, fout)
with open('test_cross_domain_3_doc.json', 'w') as fout:
    json.dump(crossdomain_test_episodes_3, fout)
