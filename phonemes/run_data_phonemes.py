import json
from learn_phonemes.data_phonemes import Data

dictionnaire_filepath = "./lexique/dico.json"

with open(dictionnaire_filepath, 'r', encoding='utf-8') as f:
    dictionnaire = json.load(f)

CAR_LIMIT_MAX = 10
CAR_LIMIT_MIN = 5

limited_dictionnaire = {cle: valeur for cle, valeur in dictionnaire.items() if len(cle) <= CAR_LIMIT_MAX and len(cle) >= CAR_LIMIT_MIN}

data = Data()
data.analyze(limited_dictionnaire)
data.load()

