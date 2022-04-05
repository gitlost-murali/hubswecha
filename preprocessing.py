import os
import re
import json


print()
import spacy
from spacy.gold import biluo_tags_from_offsets
from spacy.tokenizer import Tokenizer

nlp = spacy.load("en_core_web_sm")
nlp.tokenizer = Tokenizer(nlp.vocab, token_match=re.compile(r'\S+').match)

with open("data/data.json","r") as fh: 
    data = json.load(fh)

TRAIN_DATA = data["spacy_data"]

iob_data = []

for text, annot in TRAIN_DATA:
    doc = nlp(text)
    tags = biluo_tags_from_offsets(doc, annot)
    tokens = [tok.text for tok in doc]
    # tags = [tag.replace("L-","I-").replace("U-","B-") for tag in tags]
    # then convert L->I and U->B to have IOB tags for the tokens in the doc
    iob_data.append((tokens,tags))

features_data_folder = "features/"
if not os.path.exists(features_data_folder): os.makedirs(features_data_folder,exist_ok=True)

with open(features_data_folder+"/iobdata.json","w") as fh:
    json.dump({"iob_data":iob_data},fh,indent=4)