import json
from learn_phonemes.data_phonemes import Data

dictionnaire_filepath = "./lexique/dico.json"

with open(dictionnaire_filepath, 'r', encoding='utf-8') as f:
    dictionnaire = json.load(f)

data = Data()
data.analyze(dictionnaire)
data.load()

