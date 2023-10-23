import json, os, re, random

import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.src.preprocessing.text import tokenizer_from_json

NB_EXAMPLES_MODEL_1_WHITE = 100000
NB_EXAMPLES_MODEL_1_GREY = 100000
SIZE_PATH = "data/sizes.json"

TOKENIZER_TITLE_PATH = "data/tokenizer_title.json"
TOKENIZER_TEXT_PATH = "data/tokenizer_text.json"
TOKENIZER_PHON_PATH = "data/tokenizer_phon.json"

MODEL_1_SEQ_BLACK_PATH = "data/seq_model1_black.tfrecord"
MODEL_1_SEQ_GREY_PATH = "data/seq_model1_grey.tfrecord"
MODEL_1_SEQ_WHITE_PATH = "data/seq_model1_white.tfrecord"
MODEL_1_SEQ_BLACK_TEST_PATH = "data/seq_model1_black_test.tfrecord"
MODEL_1_SEQ_GREY_TEST_PATH = "data/seq_model1_grey_test.tfrecord"
MODEL_1_SEQ_WHITE_TEST_PATH = "data/seq_model1_white_test.tfrecord"

MODEL_2_SEQ_BLACK_PATH = "data/seq_model2_black.tfrecord"
MODEL_2_SEQ_GREY_PATH = "data/seq_model2_grey.tfrecord"
MODEL_2_SEQ_WHITE_PATH = "data/seq_model2_white.tfrecord"
MODEL_2_SEQ_BLACK_TEST_PATH = "data/seq_model2_black_test.tfrecord"
MODEL_2_SEQ_GREY_TEST_PATH = "data/seq_model2_grey_test.tfrecord"
MODEL_2_SEQ_WHITE_TEST_PATH = "data/seq_model2_white_test.tfrecord"

COMPRESSION_TYPE = None  # "GZIP"

TEXT_PATH1 = "dataset/rhymes_1.json"
TEXT_PATH2 = "dataset/rhymes_2.json"
TEXT_PATH3 = "dataset/rhymes_3.json"
TITLE_PATH = "dataset/bd_meta.json"
rhymes = []
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
    "!'": "!",
    "!I": "!",
    "!'": "!",
    "''": "'",
    "--": ""
}
SCHEMAS_SONNETS = {
    "sonnet_sicilien1": "ABABABABCDECDE",
    "sonnet_sicilien2": "ABABABABCDCCDC",
    "sonnet_petrarquien1": "ABBAABBACDECDE",
    "sonnet_petrarquien2": "ABBAABBACDCDCD",
    "sonnet_petrarquien3": "ABBAABBACDEDCE",
    "ABBAABBACCDCDC"
    "sonnet_marotique": "ABBAABBACCDEED",
    "sonnet_francais": "ABBAABBACCDEDE",
    "sonnet_queneau": "ABABABABCCDEDE",
    "sonnet_shakespearien": "ABABCDCDEFEFGG",
    "sonnet_spencerien": "ABABBCBCCDCDEE",
    "sonnet_irrationnel": "AABCBAABCCDCCD",

    "sonnet_miroir_quatrains_tercets": "ABBABAABCDCDCD",
    "sonnet_miroir_transition": "ABBABAABCCDEED",
    "sonnet_miroir_mixte": "ABBACDDCEFFEGG",
    "sonnet_miroir_couple_final": "AABBAABBCCDDAA",

    "sonnet_monorime": "AAAAAAAAAAAAAA",
    "sonnet_terza_rima": "ABAABBCBCCDCDD",
    "sonnet_envelope": "ABBAABBAABBAAB",
    "sonnet_rime_royal": "ABABBCCDDEEFFG",
    "sonnet_keats_ode": "ABABCDECDEFGFG",
    "sonnet_miltonic": "ABBAACCADEDEFF",
    "sonnet_retourné": "AABBCCDDEEFFGG",
    "sonnet_tailed": "AABAABAABAABAA",

    "sonnet_custom": "ABBAABBACDDCEE"
}


class Data():
    def __init__(self):
        self.text_tokenizer = None
        self.title_tokenizer = None
        self.phon_tokenizer = None

        self.title_words = 0
        self.text_words = 0
        self.phon_words = 0

        self.title_max_size = 0
        self.text_max_size = 0
        self.phon_max_size = 0

        self.model_1_white_seq = None
        self.model_1_grey_seq = None
        self.model_1_black_seq = None

        print("Init ok")

    def convertir_rimes_en_lettres(self, sequence_rimes):
        lettres = []
        map_rimes = {}
        courant_lettre = 65

        for rime in sequence_rimes:
            if rime not in map_rimes:
                map_rimes[rime] = chr(courant_lettre)
                courant_lettre += 1
            lettres.append(map_rimes[rime])

        return ''.join(lettres)

    def analyze(self):
        print("Start Analysing")

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

        def clean_phon(line):
            clean_line = line
            for char in PHONS_TO_REPLACE:
                clean_line = clean_line.replace(char, PHONS_TO_REPLACE[char])
            return clean_line

        def sonnet_is_valid(lettres):
            for schema in SCHEMAS_SONNETS:
                if lettres == SCHEMAS_SONNETS[schema]:
                    # print(schema)
                    return True
            return False

        def sequence_and_pad_model1(model_1_data):
            sequences = []
            for data in model_1_data:
                lines = [s.rjust(3, '.') for s in data['phons']]
                phons = "$" + ''.join(lines) + "€"
                sequences.append({
                    'title': pad_sequences([self.title_tokenizer.texts_to_sequences([data['title']])[0]],
                                           maxlen=self.title_max_size, padding='post')[0],
                    'phons': pad_sequences(self.phon_tokenizer.texts_to_sequences([phons]),
                                           maxlen=self.phon_max_size * 14 + 2, padding='post')[0]
                })
            return sequences

        def sequence_and_pad_model2(model_2_data):
            sequences = []
            for data in model_2_data:
                lines = [s.rjust(3, '.') for s in data['phons']]
                phons = "$" + ''.join(lines) + "€"
                sequences.append({
                    'title': pad_sequences([self.title_tokenizer.texts_to_sequences([data['title']])[0]],
                                           maxlen=self.title_max_size, padding='post')[0],
                    'phons': pad_sequences([self.phon_tokenizer.texts_to_sequences([phons])],
                                           maxlen=self.phon_max_size * 14 + 2, padding='post')[0],
                    'text': pad_sequences([self.text_tokenizer.texts_to_sequences([data['text']])[0]],
                                          maxlen=self.text_max_size, padding='post')[0]
                })
            return sequences

        initial_data = []

        add_data_from_file(TEXT_PATH1)
        add_data_from_file(TEXT_PATH2)
        add_data_from_file(TEXT_PATH3)

        initial_data = sorted(initial_data, key=lambda x: x['id'])

        with open(TITLE_PATH, "r", encoding='utf-8') as file:
            meta_data = json.load(file)

        sonnets_id = {}
        for item in initial_data:
            # Extraction des informations d'identification
            parts = item['id'].split('-')
            sonnet_id = parts[0] + "-" + parts[1]
            strophe_number = parts[2]
            line_number = parts[3]
            # escaped_word = 'ɛ̃'
            # if re.search(rf'\b{escaped_word}\b', item['phon']):
            #    print("")
            phon = clean_phon(item['phon'])  # .replace(' ̃', '~')
            text = item['text']

            if "Un myflère d'amour dans le métal" in text:
                if len(phon) == 1:
                    phon = 'p'
                if len(phon) == 2:
                    phon = 'po'
                if len(phon) == 3:
                    phon = 'poz'

            if text == "L'eau du canal s'irrite, et la lagune au .ôle":
                if len(phon) == 1:
                    phon = 'l'
                if len(phon) == 2:
                    phon = 'ol'
                if len(phon) == 3:
                    phon = 'mol'

            if sonnet_id not in sonnets_id:
                title = remove_spaces(add_spaces(clean(meta_data[item['id_sonnet']]['titre sonnet'])))
                theme = meta_data[item['id_sonnet']]['thème']
                sonnets_id[sonnet_id] = {
                    'lines': [{'phon': []} for _ in range(14)],
                    'title': title,
                    'theme': theme
                }

            line = convert_to_line_index(int(strophe_number), int(line_number))

            # if '!\'' in text:
            #    print("")
            sonnets_id[sonnet_id]['lines'][line]['text'] = "<start> " + remove_spaces(
                add_spaces(clean(text))) + " <end>"
            sonnets_id[sonnet_id]['lines'][line]['phon'].append(phon)

        sonnets = []
        for key in sonnets_id:
            sonnet = sonnets_id[key]
            sonnet['id'] = key
            sonnets.append(sonnet)

        # Array des titres
        titles = [sonnet['title'] for sonnet in sonnets]

        # Array des lines
        lines = [line['text'] for sonnet in sonnets for line in sonnet['lines']]
        with open("debug/lines.txt", "w") as f:
            for word in lines:
                f.write(word + '\n')

        # Set des phons
        set_phons = {phon for sonnet in sonnets for line in sonnet['lines'] for phon in line['phon']}
        phons_list = list(set_phons)
        phons_list.sort(key=len, reverse=True)

        # Dictionaire des themes
        themes = {}
        for sonnet in sonnets:
            theme = sonnet['theme']
            if theme in themes:
                themes[theme] += 1
            else:
                themes[theme] = 1

        # Set des themes avec phons
        themes_phons = {theme: {} for theme in themes}
        set_phons = set()
        for sonnet in sonnets:
            theme = sonnet['theme']
            for line in sonnet['lines']:
                for phon in line['phon']:
                    set_phons.add(phon)
                    if phon not in themes_phons[theme]:
                        themes_phons[theme][phon] = 0
                    themes_phons[theme][phon] += 1

        # On créé un dictionnaire des mots utilisés (2 max) en fonction des thèmes et des phons associés
        themes_phons_words = {}
        for theme in themes_phons:
            themes_phons_words[theme] = {}
            for phon in themes_phons[theme]:
                themes_phons_words[theme][phon] = set()

        for sonnet in sonnets:
            theme = sonnet['theme']
            for line in sonnet['lines']:
                phons = line['phon']
                words = line['text'].split(' ')[1:-1]  # On enlève les tags de début et fin pour l'analyse
                index_word1 = len(words) - 1
                index_word2 = len(words) - 1
                for index, word in enumerate(reversed(words)):  # Parcourir la liste de mots en sens inverse
                    if len(word) > 1 and index_word1 != len(words) - 1:
                        index_word2 = index
                        break
                    elif len(word) > 1:
                        index_word1 = index
                for phon in phons:
                    if index_word2 != index_word1:
                        themes_phons_words[theme][phon].add(
                            words[len(words) - index_word2 - 1] + " " + words[len(words) - index_word1 - 1])
                    else:
                        themes_phons_words[theme][phon].add(words[len(words) - index_word1 - 1])

        # On récupère le schéma du sonnet depuis les rimes pauvres
        # On l'utilisera plus tard pour définir ou non si on conserve les rimes + riches
        themes_schema = {theme: {} for theme in themes}
        for sonnet in sonnets:
            sonnet_seq_array = []
            for i, line in enumerate(sonnet['lines']):
                sonnet_seq_array.append(sonnet['lines'][i]['phon'][0])
            sonnet['schema'] = self.convertir_rimes_en_lettres(sonnet_seq_array)
            if sonnet['schema'] not in themes_schema[sonnet['theme']]:
                themes_schema[sonnet['theme']][sonnet['schema']] = 1
            else:
                themes_schema[sonnet['theme']][sonnet['schema']] += 1

        # Liste de tous les schemas
        set_schemas = {sonnet['schema'] for sonnet in sonnets}

        # Tokenisation
        self.text_tokenizer = Tokenizer(filters='"#$%&()*+/:=?@[\\]^_`{|}~\t\n')
        lines.append("<start>")
        lines.append("<end>")
        self.text_tokenizer.fit_on_texts(lines)
        words_text = list(self.text_tokenizer.word_index.keys())
        with open("debug/words_text.txt", "w") as f:
            for word in words_text:
                f.write(word + '\n')

        self.title_tokenizer = Tokenizer()
        self.title_tokenizer.fit_on_texts(titles)
        words_title = list(self.title_tokenizer.word_index.keys())
        with open("debug/words_title.txt", "w") as f:
            for word in words_title:
                f.write(word + '\n')

        self.phon_tokenizer = Tokenizer(filters="", char_level=True)
        list_phons = list(set_phons)
        list_phons.append("$")
        list_phons.append("€")
        list_phons.append(".")
        self.phon_tokenizer.fit_on_texts(list_phons)
        words_phon = list(self.phon_tokenizer.word_index.keys())
        with open("debug/words_phon.txt", "w") as f:
            for word in words_phon:
                f.write(word + '\n')

        self.title_max_size = max([len(line) for line in titles])
        self.title_words = len(words_title)

        self.text_max_size = max([len(line) for line in lines])
        self.text_words = len(words_text)

        self.phon_max_size = max([len(phons) for phons in list(set_phons)])
        self.phon_words = len(words_phon)

        # On génère le premier Dataset du modèle 1
        # Model 1 - Black Data
        # Sur la base des données réelles
        # Title seq => 14 x Phon seq
        print("Generating - Model 1 - Black Data")
        model_1_data_black = []

        for i, sonnet in enumerate(sonnets):
            progress_percent = (i / len(sonnets)) * 100
            print(f"\rGenerating progress: {progress_percent:.2f}%", end="")
            sonnet_seq1 = {}
            sonnet_seq1['title'] = sonnet['title']
            sonnet_seq1['phons'] = [[] for _ in range(14)]
            sonnet_seq2 = {}
            sonnet_seq2['title'] = sonnet['title']
            sonnet_seq2['phons'] = [[] for _ in range(14)]
            sonnet_seq3 = {}
            sonnet_seq3['title'] = sonnet['title']
            sonnet_seq3['phons'] = [[] for _ in range(14)]
            sonnet_seq2_array = []
            sonnet_seq3_array = []
            for i, line in enumerate(sonnet['lines']):
                sonnet_seq2_array.append(sonnet['lines'][i]['phon'][1])
                sonnet_seq3_array.append(sonnet['lines'][i]['phon'][2])
                sonnet_seq1['phons'][i] = sonnet['lines'][i]['phon'][0]
                sonnet_seq2['phons'][i] = sonnet['lines'][i]['phon'][1]
                sonnet_seq3['phons'][i] = sonnet['lines'][i]['phon'][2]
                # sonnet_seq1[f'phon_{i + 1}'] = sonnet['lines'][i]['phon'][0]
                # sonnet_seq2[f'phon_{i + 1}'] = sonnet['lines'][i]['phon'][1]
                # sonnet_seq3[f'phon_{i + 1}'] = sonnet['lines'][i]['phon'][2]

            model_1_data_black.append(sonnet_seq1)
            if self.convertir_rimes_en_lettres(sonnet_seq2_array) == sonnet['schema']:
                model_1_data_black.append(sonnet_seq2)
            if self.convertir_rimes_en_lettres(sonnet_seq3_array) == sonnet['schema']:
                model_1_data_black.append(sonnet_seq3)

        self.model_1_black_seq = sequence_and_pad_model1(model_1_data_black)
        print("\rGenerated - Model 1 - {} Black Data".format(str(len(self.model_1_black_seq))))

        # Model 1 - Grey Data
        # Génération de données sur la base des schémas et des rimes distribués par thème
        # Pour X données, on réparti selon :
        # On choisi un thème avec sa distribution
        # On choisi un schéma avec sa distribution au sein du thème
        # On choisi si on veut générer des rimes riches (3), normales (2), pauvres (1)
        # On choisi des rimes avec sa distribution au sein du thème,
        # => si la rime choisie a déjà été prise on en prend une autre
        # On met un titre à vide
        # Title seq => 14 x Phon seq

        print("Generating - Model 1 - Grey Data")
        model_1_data_grey = []
        for i in range(NB_EXAMPLES_MODEL_1_GREY):
            progress_percent = (i / NB_EXAMPLES_MODEL_1_GREY) * 100
            print(f"\rGenerating progress: {progress_percent:.2f}%", end="")
            selected_theme = random.choices(list(themes.keys()), weights=themes.values(), k=1)[0]
            selected_schema = \
            random.choices(list(themes_schema[selected_theme].keys()), weights=themes_schema[selected_theme].values(),
                           k=1)[0]
            taille_rimes = random.randint(1, 3)
            all_rimes = {phon: themes_phons[selected_theme][phon] for phon in themes_phons[selected_theme] if
                         len(phon) == taille_rimes}

            rimes_choisies = set()
            while len(rimes_choisies) < len(set(selected_schema)):
                selection = random.choices(list(all_rimes.keys()), weights=all_rimes.values(), k=1)[0]
                rimes_choisies.add(selection)

            rimes_choisies_list = list(rimes_choisies)
            map_schema_to_rimes = dict(zip(set(selected_schema), rimes_choisies_list))
            rimes = [map_schema_to_rimes[char] if char in map_schema_to_rimes else char for char in
                     selected_schema]
            model_1_data_grey.append({
                'title': '',
                'phons': rimes
            })
        self.model_1_grey_seq = sequence_and_pad_model1(model_1_data_grey)
        print("\rGenerated - Model 1 - {} Grey Data".format(str(len(self.model_1_grey_seq))))

        # Model 1 - White Data
        # Génération de données sur la base des schémas et des rimes
        # Pour X données, on réparti selon :
        # On choisi un schéma avec sa distribution
        # On choisi si on veut générer des rimes riches (3), normales (2), pauvres (1)
        # On choisi des rimes
        # => si la rime choisie a déjà été prise on en prend une autre
        # On met un titre à vide
        # Title seq => 14 x Phon seq

        print("Generating - Model 1 - White Data")
        model_1_data_white = []
        for i in range(NB_EXAMPLES_MODEL_1_WHITE):
            progress_percent = (i / NB_EXAMPLES_MODEL_1_WHITE) * 100
            print(f"\rGenerating progress: {progress_percent:.2f}%", end="")
            selected_schema = random.choices(list(set_schemas))[0]
            taille_rimes = random.randint(1, 3)
            all_rimes = {phon for phon in set_phons if len(phon) == taille_rimes}

            rimes_choisies = set()
            while len(rimes_choisies) < len(set(selected_schema)):
                selection = random.choices(list(all_rimes))[0]
                rimes_choisies.add(selection)

            rimes_choisies_list = list(rimes_choisies)
            map_schema_to_rimes = dict(zip(set(selected_schema), rimes_choisies_list))
            rimes = [map_schema_to_rimes[char] if char in map_schema_to_rimes else char for char in
                     selected_schema]
            model_1_data_white.append({
                'title': '',
                'phons': rimes
            })
        self.model_1_white_seq = sequence_and_pad_model1(model_1_data_white)
        print("\rGenerated - Model 1 - {} White Data".format(str(len(self.model_1_white_seq))))

        # On génère le premier Dataset du modèle 2
        # Model 2 - Black Data
        # Sur la base des données réelles
        # Title seq + Phon seq => Verset
        print("Generating - Model 2 - Black Data")
        model_2_data_black = []
        for i, sonnet in enumerate(sonnets):
            progress_percent = (i / len(sonnets)) * 100
            print(f"\rGenerating progress: {progress_percent:.2f}%", end="")
            title = sonnet['title']
            for line in sonnet['lines']:
                for phon in line['phon']:
                    model_2_data_black.append({
                        'title': title,
                        'phons': phon,
                        'text': line['text']
                    })
        self.model_2_black_seq = sequence_and_pad_model2(model_2_data_black)
        print("\rGenerated - Model 2 - {} Black Data".format(str(len(self.model_2_black_seq))))

        # Model 2 - Grey Data
        # Sur la base des données réelles mais on fait varier les derniers mots sur la bases des autres mots pour le même thème et le même phon
        # Title seq + Phon seq => Verset
        print("Generating - Model 2 - Grey Data")
        model_2_data_grey = []
        for i, sonnet in enumerate(sonnets):
            progress_percent = (i / len(sonnets)) * 100
            print(f"\rGenerating progress: {progress_percent:.2f}%", end="")
            title = sonnet['title']
            theme = sonnet['theme']
            for line in sonnet['lines']:
                for phon in line['phon']:
                    words_dictionnary_phon = list(themes_phons_words[theme][phon])
                    num_samples = min(len(words_dictionnary_phon), 10)
                    selected_words = random.sample(words_dictionnary_phon, num_samples)
                    for word in selected_words:
                        new_line_array = line['text'].split(' ')[1:-1]

                        # On gère la ponctuation (On la supprime pour la remettre à la fin)
                        if len(new_line_array[-1]) == 1:
                            last_car = ' ' + new_line_array[-1] + ' '
                            new_line_array = new_line_array[:-1]
                        else:
                            last_car = ''
                        words_list = word.split(' ')

                        if len(phon) == 1:
                            limit = 3  # 3 lettres maximum pour un phoneme (eau dans beau, ien dans chien)
                        elif len(phon) == 2:
                            limit = 5  # On considère qu'il est peu probable que ça arrive 2 fois de suite qu'il y ait 3 lettres pour un phoneme
                        else:
                            limit = 6  # Idem
                        # Si le dernier mot est plus petit que les limites définies alors on ajoute un second mot (le phoneme peut représenter deux mots distincts dont le dernier très court)
                        if len(words_list) == 2 and len(new_line_array) >= 2 and len(words_list[-1]) <= limit:
                            new_line_array[-2] = words_list[0]
                        if (len(words_list) == 1):
                            new_line_array[-1] = words_list[0]
                        else:
                            new_line_array[-1] = words_list[1]

                        new_line = ' '.join(new_line_array)

                        model_2_data_grey.append({
                            'title': title,
                            'phons': phon,
                            'text': '<start> ' + new_line + last_car + ' <end>'
                        })
        self.model_2_grey_seq = sequence_and_pad_model2(model_2_data_grey)
        print("\rGenerated - Model 2 - {} Grey Data".format(str(len(self.model_2_grey_seq))))

        print("Generating - Model 1 - White Data")
        print("End analyze")

    def save(self):
        print("Start Saving")

        def save_rimes_to_tfrecord(sequences, filename):
            options = tf.io.TFRecordOptions(compression_type=COMPRESSION_TYPE)
            with tf.io.TFRecordWriter(filename, options=options) as writer:
                for sequence in sequences:
                    feature = {
                        'title': tf.train.Feature(int64_list=tf.train.Int64List(value=list(sequence['title']))),
                        'phons': tf.train.Feature(int64_list=tf.train.Int64List(value=list(sequence['phons'])))
                    }
                    example = tf.train.Example(features=tf.train.Features(feature=feature))
                    writer.write(example.SerializeToString())

        def save_versets_to_tfrecord(sequences, filename):
            options = tf.io.TFRecordOptions(compression_type=COMPRESSION_TYPE)
            with tf.io.TFRecordWriter(filename, options=options) as writer:
                for sequence in sequences:
                    feature = {
                        'title': tf.train.Feature(int64_list=tf.train.Int64List(value=list(sequence['title']))),
                        'phons': tf.train.Feature(int64_list=tf.train.Int64List(value=list(sequence['phons']))),
                        'text': tf.train.Feature(int64_list=tf.train.Int64List(value=list(sequence['phons'])))
                    }
                    example = tf.train.Example(features=tf.train.Features(feature=feature))
                    writer.write(example.SerializeToString())

        def save_tokenizer(tokenizer, filename):
            tokenizer_json = tokenizer.to_json()
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(tokenizer_json)

        with open(SIZE_PATH, 'w', encoding='utf-8') as f:
            json.dump({
                'title_max_size': self.title_max_size,
                'text_max_size': self.text_max_size,
                'phon_max_size': self.phon_max_size,
                'title_words': self.title_words,
                'text_words': self.text_words,
                'phon_words': self.phon_words
            }, f)

        save_tokenizer(self.title_tokenizer, TOKENIZER_TITLE_PATH)
        save_tokenizer(self.text_tokenizer, TOKENIZER_TEXT_PATH)
        save_tokenizer(self.phon_tokenizer, TOKENIZER_PHON_PATH)

        size_black_rimes = int(len(self.model_1_black_seq) / 100)
        save_rimes_to_tfrecord(self.model_1_black_seq[:-size_black_rimes], MODEL_1_SEQ_BLACK_PATH)
        save_rimes_to_tfrecord(self.model_1_black_seq[-size_black_rimes:], MODEL_1_SEQ_BLACK_TEST_PATH)

        size_grey_rimes = int(len(self.model_1_grey_seq) / 100)
        save_rimes_to_tfrecord(self.model_1_grey_seq[:-size_grey_rimes], MODEL_1_SEQ_GREY_PATH)
        save_rimes_to_tfrecord(self.model_1_grey_seq[-size_grey_rimes:], MODEL_1_SEQ_GREY_TEST_PATH)

        size_white_rimes = int(len(self.model_1_white_seq) / 100)
        save_rimes_to_tfrecord(self.model_1_white_seq[:-size_white_rimes], MODEL_1_SEQ_WHITE_PATH)
        save_rimes_to_tfrecord(self.model_1_white_seq[-size_white_rimes:], MODEL_1_SEQ_WHITE_TEST_PATH)

        size_black_versets = int(len(self.model_2_black_seq) / 100)
        save_versets_to_tfrecord(self.model_2_black_seq[:-size_black_versets], MODEL_2_SEQ_BLACK_PATH)
        save_versets_to_tfrecord(self.model_2_black_seq[-size_black_versets:], MODEL_2_SEQ_BLACK_TEST_PATH)

        size_grey_versets = int(len(self.model_2_grey_seq) / 100)
        save_versets_to_tfrecord(self.model_2_grey_seq[:-size_grey_versets], MODEL_2_SEQ_GREY_PATH)
        save_versets_to_tfrecord(self.model_2_grey_seq[-size_grey_versets:], MODEL_2_SEQ_GREY_TEST_PATH)

        # size_white_versets = int(len(self.model_1_white_seq) / 100)
        # save_versets_to_tfrecord(self.model_2_white_seq[:-size_white_versets], MODEL_2_SEQ_WHITE_PATH)
        # save_versets_to_tfrecord(self.model_2_white_seq[-size_white_versets:], MODEL_2_SEQ_WHITE_TEST_PATH)

        print("End Saving")

    def load(self):
        print("Start Loading")

        def load_tokenizer(filename):
            with open(filename, 'r', encoding='utf-8') as f:
                tokenizer_json = f.read()
            tokenizer = tokenizer_from_json(tokenizer_json)
            return tokenizer

        with open(SIZE_PATH, 'r', encoding='utf-8') as f:
            sizes = json.load(f)
            self.title_max_size = sizes['title_max_size']
            self.text_max_size = sizes['text_max_size']
            self.phon_max_size = sizes['phon_max_size']
            self.title_words = sizes['title_words']
            self.text_words = sizes['text_words']
            self.phon_words = sizes['phon_words']

        self.title_tokenizer = load_tokenizer(TOKENIZER_TITLE_PATH)
        self.text_tokenizer = load_tokenizer(TOKENIZER_TEXT_PATH)
        self.phon_tokenizer = load_tokenizer(TOKENIZER_PHON_PATH)

        print("End Loading")

    def load_rimes_from_tfrecord(self, filename, batch_size, title_max_size, phon_max_size):
        feature = {
            'title': tf.io.FixedLenFeature([title_max_size], tf.int64),
            'phons': tf.io.FixedLenFeature([phon_max_size * 14 + 2], tf.int64)
        }

        def _parse_function(example_proto):
            return tf.io.parse_single_example(example_proto, feature)

        raw_dataset = tf.data.TFRecordDataset(filename, compression_type=COMPRESSION_TYPE)
        parsed_dataset = raw_dataset.map(_parse_function, num_parallel_calls=tf.data.AUTOTUNE).take(
            batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
        return parsed_dataset

    def load_versets_from_tfrecord(self, filename, batch_size, title_max_size, phon_max_size, text_max_size):
        feature = {
            'title': tf.io.FixedLenFeature([title_max_size], tf.int64),
            'phons': tf.io.FixedLenFeature([phon_max_size * 14 + 2], tf.int64),
            'text': tf.io.FixedLenFeature([text_max_size], tf.int64)
        }

        def _parse_function(example_proto):
            return tf.io.parse_single_example(example_proto, feature)

        raw_dataset = tf.data.TFRecordDataset(filename, compression_type=COMPRESSION_TYPE)
        parsed_dataset = raw_dataset.map(_parse_function, num_parallel_calls=tf.data.AUTOTUNE).take(
            batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
        return parsed_dataset

    def count_tfrecord_samples(self, filename):
        count = 0
        for _ in tf.data.TFRecordDataset(filename, compression_type=COMPRESSION_TYPE):
            count += 1
        return count

    def generate_data_phon(self, batch_size, epoch_size, epoch, train):
        # print("\nStart generate_data_phon epoch :{}".format(epoch))

        data_black = 0.0
        data_grey = 0.0
        data_white = 1.0

        initial_stable_epochs = 10
        phase_limit = 10
        change_per_epoch = 1.0 / phase_limit

        if epoch < initial_stable_epochs:
            # Pas de changement, rester à 100% white
            pass
        else:
            # Calcul de l'epoch ajustée après les epochs initiales stables
            adjusted_epoch = epoch - initial_stable_epochs

            if adjusted_epoch < phase_limit:
                # Première phase : diminuer white, augmenter grey
                data_white -= adjusted_epoch * change_per_epoch
                data_grey += adjusted_epoch * change_per_epoch
            elif adjusted_epoch < (2 * phase_limit):
                # Deuxième phase : white est à 0, diminuer grey, augmenter black
                data_white = 0.0
                remaining_epochs = adjusted_epoch - phase_limit
                data_grey = 1.0 - remaining_epochs * change_per_epoch
                data_black = remaining_epochs * change_per_epoch
            else:
                # Troisième phase : seulement des données black
                data_white = 0.0
                data_grey = 0.0
                data_black = 1.0

        if train == True:
            print("\nUsing {}% White, {}% Grey, {}% Black".format(data_white * 100, data_grey * 100, data_black * 100))

        batch_white = int(data_white * batch_size)
        batch_grey = int(data_grey * batch_size)
        batch_black = batch_size - (batch_white + batch_grey)

        for i in range(epoch_size):
            # print("Start batch {} {}".format(i+1, epoch_size))
            if train == True:
                white_data = self.load_rimes_from_tfrecord(MODEL_1_SEQ_WHITE_PATH, batch_white, self.title_max_size,
                                                           self.phon_max_size)
                grey_data = self.load_rimes_from_tfrecord(MODEL_1_SEQ_GREY_PATH, batch_grey, self.title_max_size,
                                                          self.phon_max_size)
                black_data = self.load_rimes_from_tfrecord(MODEL_1_SEQ_BLACK_PATH, batch_black, self.title_max_size,
                                                           self.phon_max_size)
            else:
                white_data = self.load_rimes_from_tfrecord(MODEL_1_SEQ_WHITE_TEST_PATH, batch_white,
                                                           self.title_max_size, self.phon_max_size)
                grey_data = self.load_rimes_from_tfrecord(MODEL_1_SEQ_GREY_TEST_PATH, batch_grey, self.title_max_size,
                                                          self.phon_max_size)
                black_data = self.load_rimes_from_tfrecord(MODEL_1_SEQ_BLACK_TEST_PATH, batch_black,
                                                           self.title_max_size, self.phon_max_size)

            merged_data = white_data.concatenate(grey_data).concatenate(black_data)

            # Vérifier si le dataset est vide
            # if merged_data.cardinality().numpy() == 0:
            #    print("Le dataset fusionné est vide. Veuillez vérifier les datasets sources.")
            #    continue  # Passez à la prochaine itération si le dataset est vide

            for batch in merged_data.batch(batch_size):
                # print("Start generate_data batch")
                input_title = batch['title']
                target_phon = batch['phons']

                # input_title = tf.convert_to_tensor(input_title, dtype=tf.int64)
                # target_phon = tf.convert_to_tensor(target_phon, dtype=tf.int64)

                decoder_input = target_phon[:, :-1]
                decoder_output = target_phon[:, 1:]

                decoder_output_onehot = tf.keras.utils.to_categorical(decoder_output, num_classes=self.phon_words + 1)
                # print("End generate_data batch")
                yield (input_title, decoder_input), decoder_output_onehot
        #print("\nEnd generate_data")

    def generate_data_verset(self, batch_size, epoch_size, epoch, train):
        # print("\nStart generate_data_verset epoch :{}".format(epoch))

        data_black = 0.0
        data_grey = 0.0
        data_white = 1.0

        initial_stable_epochs = 10
        phase_limit = 10
        change_per_epoch = 1.0 / phase_limit

        if epoch < initial_stable_epochs:
            # Pas de changement, rester à 100% white
            pass
        else:
            # Calcul de l'epoch ajustée après les epochs initiales stables
            adjusted_epoch = epoch - initial_stable_epochs

            if adjusted_epoch < phase_limit:
                # Première phase : diminuer white, augmenter grey
                data_white -= adjusted_epoch * change_per_epoch
                data_grey += adjusted_epoch * change_per_epoch
            elif adjusted_epoch < (2 * phase_limit):
                # Deuxième phase : white est à 0, diminuer grey, augmenter black
                data_white = 0.0
                remaining_epochs = adjusted_epoch - phase_limit
                data_grey = 1.0 - remaining_epochs * change_per_epoch
                data_black = remaining_epochs * change_per_epoch
            else:
                # Troisième phase : seulement des données black
                data_white = 0.0
                data_grey = 0.0
                data_black = 1.0

        if train == True:
            print("\nUsing {}% White, {}% Grey, {}% Black".format(data_white * 100, data_grey * 100, data_black * 100))

        batch_white = int(data_white * batch_size)
        batch_grey = int(data_grey * batch_size)
        batch_black = batch_size - (batch_white + batch_grey)

        for i in range(epoch_size):
            # print("Start batch {} {}".format(i+1, epoch_size))
            if train == True:
                white_data = self.load_versets_from_tfrecord(MODEL_2_SEQ_WHITE_PATH, batch_white, self.title_max_size,
                                                           self.phon_max_size)
                grey_data = self.load_versets_from_tfrecord(MODEL_2_SEQ_GREY_PATH, batch_grey, self.title_max_size,
                                                          self.phon_max_size)
                black_data = self.load_versets_from_tfrecord(MODEL_2_SEQ_BLACK_PATH, batch_black, self.title_max_size,
                                                           self.phon_max_size)
            else:
                white_data = self.load_versets_from_tfrecord(MODEL_2_SEQ_WHITE_TEST_PATH, batch_white,
                                                           self.title_max_size, self.phon_max_size)
                grey_data = self.load_versets_from_tfrecord(MODEL_2_SEQ_GREY_TEST_PATH, batch_grey, self.title_max_size,
                                                          self.phon_max_size)
                black_data = self.load_versets_from_tfrecord(MODEL_2_SEQ_BLACK_TEST_PATH, batch_black,
                                                           self.title_max_size, self.phon_max_size)

            merged_data = white_data.concatenate(grey_data).concatenate(black_data)

            # Vérifier si le dataset est vide
            # if merged_data.cardinality().numpy() == 0:
            #    print("Le dataset fusionné est vide. Veuillez vérifier les datasets sources.")
            #    continue  # Passez à la prochaine itération si le dataset est vide

            for batch in merged_data.batch(batch_size):
                # print("Start generate_data batch")
                input_title = batch['title']
                target_phon = batch['phons']

                # input_title = tf.convert_to_tensor(input_title, dtype=tf.int64)
                # target_phon = tf.convert_to_tensor(target_phon, dtype=tf.int64)

                decoder_input = target_phon[:, :-1]
                decoder_output = target_phon[:, 1:]

                decoder_output_onehot = tf.keras.utils.to_categorical(decoder_output, num_classes=self.phon_words + 1)
                # print("End generate_data batch")
                yield (input_title, decoder_input), decoder_output_onehot
        #print("\nEnd generate_data")
