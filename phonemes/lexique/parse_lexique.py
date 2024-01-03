import pandas as pd
import json

lexique_filepath = "Lexique383.tsv"
dictionnaire_filepath = "dic.json"

# Replace 'file_path.tsv' with the path to your TSV file
df = pd.read_csv(lexique_filepath, sep='\t')

# Créer un dictionnaire vide
dictionnaire = {}

# Parcourir chaque ligne et remplir le dictionnaire
for index, row in df.iterrows():
    dictionnaire[row['ortho']] = {
        'phon': row['phon'],
        'nbsyll': row['nbsyll']
    }

with open(dictionnaire_filepath, 'w', encoding='utf-8') as f:
    json.dump(dictionnaire, f, ensure_ascii=False, indent=4)

print("Dictionnaire enregistré")
