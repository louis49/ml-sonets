import json, re

sonnets_filepath = "sonnets.json"

TEXT_PATH1 = "dataset/rhymes_1.json"
TEXT_PATH2 = "dataset/rhymes_2.json"
TEXT_PATH3 = "dataset/rhymes_3.json"
TITLE_PATH = "dataset/bd_meta.json"

CHARS_TO_REMOVE = ['', '■', '¡', '\x8a', '—', '�', '…', '«', '°', 'µ', '•', '–', '»', '„', '©', '[', ']', '\"', '(',
                   ')', '^', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '<', '>', '|', '/', '_', '~', '\\', '*']
QUOTES_TO_REPLACE = ['‘', '’', '\'\'', '`']
PHONS_TO_REPLACE = {
    'ɑ̃': 'â',
    'ɔ̃': 'ç',
    'ɛ̃': 'ê',
    'œ̃': 'ô'
}

WORDS_TO_REPLACE = {
    "repo/e": "repose",
    ".ôle": "môle",
    ".éperduvole": "éperdu vole",
    ".suprême": "suprême",
    "'/": "",
    "&/ouffrance": "& souffrance",
    "moi&": "mois",
    "revëche": "revêche",
    "cigales'": "cigales",
    "sa-détresse": "sa détresse",
    "Votr . e aimable": "Votre aimable",
    "repo e ": "repose ",
    "rcvenir": "revenir",
    " d ": " d'",
    " s ": " s'",
    "So u s": "Sous",
    "Mais j'usurpe le pain qui dans mes blés frissonne, v": "Mais j'usurpe le pain qui dans mes blés frissonne,",
    "O misérable , hélas ! toute I' humaine race": "O misérable , hélas ! toute l' humaine race",
    "Que veuxrtu ?": "Que veux-tu ?",
    "De ses forfaits au ciel ont monte' les clameurs": "De ses forfaits au ciel ont monté les clameurs",
    "|’aile": "l’aile",
    "œ": "oe",
    "Œ": "Oe",
    "!'": "!",
    "!I": "!",
    "!'": "!",
    "''": "'",
    "--": ""
}

def add_data_from_file(filepath):
    with open(filepath, "r", encoding='utf-8') as file:
        data = json.load(file)
        for _, value_list in data.items():
            initial_data.extend(value_list)

def convert_to_line_index(strophe_number, line_number):
    if strophe_number in [1, 2]:
        total_lines_before = (strophe_number - 1) * 4
    elif strophe_number == 3:
        total_lines_before = 2 * 4
    else:
        total_lines_before = 2 * 4 + 3
    line_index = total_lines_before + (line_number - 1)

    return line_index


def add_spaces(chaine):  # !
    punctuation_chars = r"[?:.!,;]"
    chaine = re.sub(fr'(?<=[^\s])({punctuation_chars})', r' \1', chaine)
    chaine = re.sub(fr'({punctuation_chars})(?=[^\s])', r'\1 ', chaine)
    return re.sub(r'(?<=[^\s])([\'’])', r'\1 ', chaine)


def remove_spaces(chaine):
    chaine = re.sub(r"\s+(['’])", r"\1 ", chaine)
    return re.sub(r'^\s+|\s+', ' ', chaine).strip()

def clean(line):
    clean_line = line
    for char in CHARS_TO_REMOVE:
        clean_line = clean_line.replace(char, ' ')
    for char in QUOTES_TO_REPLACE:
        clean_line = clean_line.replace(char, '\'')
    for word in WORDS_TO_REPLACE:
        clean_line = clean_line.replace(word, WORDS_TO_REPLACE[word])
    return clean_line

print("Reading data from initial DataSet")
initial_data = []

add_data_from_file(TEXT_PATH1)
add_data_from_file(TEXT_PATH2)
add_data_from_file(TEXT_PATH3)

initial_data = sorted(initial_data, key=lambda x: x['id'])

with open(TITLE_PATH, "r", encoding='utf-8') as file:
    meta_data = json.load(file)

print("Reconstructing sonnets")
sonnets_id = {}
for item in initial_data:
    # Extraction des informations d'identification
    parts = item['id'].split('-')
    sonnet_id = parts[0] + "-" + parts[1]
    strophe_number = parts[2]
    line_number = parts[3]
    text = item['text']

    if sonnet_id not in sonnets_id:
        title = remove_spaces(add_spaces(clean(meta_data[item['id_sonnet']]['titre sonnet'])))
        theme = meta_data[item['id_sonnet']]['thème']
        sonnets_id[sonnet_id] = {
            'lines': [{} for _ in range(14)],
            'title': title,
            'theme': theme
        }

    line = convert_to_line_index(int(strophe_number), int(line_number))

    sonnets_id[sonnet_id]['lines'][line]['text'] = remove_spaces(add_spaces(clean(text)))

sonnets = []
for key in sonnets_id:
    sonnet = sonnets_id[key]
    sonnet['id'] = key
    sonnets.append(sonnet)

with open(sonnets_filepath, 'w', encoding='utf-8') as f:
    json.dump(sonnets, f, ensure_ascii=False, indent=4)

print("Sonnets enregistrés")