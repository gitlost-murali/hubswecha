import re
import spacy
import random
import json
from spacy.tokenizer import Tokenizer
from spacy.gold import offsets_from_biluo_tags
from tqdm import tqdm

import yaml
params = yaml.safe_load(open('params.yaml'))["model"]
epochs = params["epochs"]

nlp = spacy.load("en_core_web_sm")
nlp.tokenizer = Tokenizer(nlp.vocab, token_match=re.compile(r'\S+').match)

with open("features/iobdata.json","r") as fh: 
    data = json.load(fh)

data = data["iob_data"]

train_data = []
for snts,tgts in data:
    train_data.append((" ".join(snts),{"entities":offsets_from_biluo_tags(nlp(" ".join(snts)), tgts)}))



## GET ACTUAL & BETTER SCRIPT at https://v2.spacy.io/usage/training

# create the built-in pipeline components and add them to the pipeline
# nlp.create_pipe works for built-ins that are registered with spaCy
if "ner" not in nlp.pipe_names:
    ner = nlp.create_pipe("ner")
    nlp.add_pipe(ner, last=True)
# otherwise, get it so we can add labels
else:
    ner = nlp.get_pipe("ner")

# add labels
for _, annotations in train_data:
    for ent in annotations.get("entities"):
        ner.add_label(ent[2])

other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
with nlp.disable_pipes(*other_pipes):

    optimizer = nlp.begin_training()
    for i in tqdm(range(epochs),desc="epochs"):
        random.shuffle(train_data)
        for text, annotations in train_data:
            nlp.update([text], [annotations], sgd=optimizer)

nlp.to_disk("./model")