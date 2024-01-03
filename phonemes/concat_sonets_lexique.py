import json, re

dictionnaire_filepath = "./lexique/dic.json"
sonnets_filepath = "./sonets/sonnets.json"

with open(dictionnaire_filepath, 'r', encoding='utf-8') as f:
    dictionnaire = json.load(f)

with open(sonnets_filepath, 'r', encoding='utf-8') as f:
    sonnets = json.load(f)

PUNCTUATION = ['.', ',', ';', '!', '?', '&', ':']

problems = set()
sonnets_clean = []
for sonnet in sonnets:
    problem = False
    for line in sonnet['lines']:
        text = line['text']
        splited_text = re.split('[ ,-.,\'.]', text)
        for word in splited_text:
            if word.lower() not in dictionnaire and word.lower() not in PUNCTUATION:
                #print(word)
                problems.add(word)
                problem = True
    if not problem:
        sonnets_clean.append(sonnet)

with open('erreurs.txt', 'w') as f:
    f.write('\n'.join(str(e) for e in problems))

print("")