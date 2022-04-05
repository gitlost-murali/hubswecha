import spacy
from spacy.gold import GoldParse
from spacy.scorer import Scorer
import json

def evaluate(ner_model, examples):
    scorer = Scorer()
    for input_, annot in examples:
        doc_gold_text = ner_model.make_doc(input_)
        gold = GoldParse(doc_gold_text, entities=annot)
        pred_value = ner_model(input_)
        scorer.score(pred_value, gold)
    return scorer.scores

# example run

examples = [
    ('Who is the PM of India Maps?',
     [(7, 17, 'PERSON')]),
    ('I like London and Berlin.',
     [(7, 13, 'LOC'), (18, 24, 'LOC')])
]
examples = [
            # ["Uber blew through $1 million a week", [[0, 4, "ORG"]]],
            # ["Android Pay expands to Canada", [[0, 11, "PRODUCT"], [23, 29, "GPE"]]],
            ('I like London and Berlin.', [(7, 13, 'GPE'), (18, 25, 'GPE')])
        ]
ner_model = spacy.load("model") # for spaCy's pretrained use 'en_core_web_sm'
results = evaluate(ner_model, examples)

print(results["ents_per_type"])

scores_file = "scores.json"

with open(scores_file, 'w') as fd:
    json.dump(results["ents_per_type"], fd)
