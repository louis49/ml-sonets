import copy
import json, os, re, random

import numpy as np
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.src.preprocessing.text import tokenizer_from_json

NB_EXAMPLES_MODEL_1_WHITE = 25000
NB_EXAMPLES_MODEL_1_GREY = 25000
NB_EXAMPLES_MODEL_2_WHITE = 25000
SIZE_PATH = "data/sizes.json"
RATIO_TEST = 0.002

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

MODEL_3_SEQ_BLACK_PATH = "data/seq_model3_black.tfrecord"
MODEL_3_SEQ_BLACK_TEST_PATH = "data/seq_model3_black_test.tfrecord"

COMPRESSION_TYPE = "GZIP"  # None

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

        def save_phons(sequences, writer_train, writer_test):
            for seq in sequences:
                if random.random() > RATIO_TEST:
                    save_phons_to_tfrecord(seq, writer_train)
                else:
                    save_phons_to_tfrecord(seq, writer_test)

        def save_phons_to_tfrecord(sequence, writer):
            feature = {
                'title': tf.train.Feature(int64_list=tf.train.Int64List(value=list(sequence['title']))),
                'phons': tf.train.Feature(int64_list=tf.train.Int64List(value=list(sequence['phons'])))
            }
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())

        def save_versets(sequences, writer_train, writer_test):
            for seq in sequences:
                if random.random() > RATIO_TEST:
                    save_versets_to_tfrecord(seq, writer_train)
                else:
                    save_versets_to_tfrecord(seq, writer_test)
        def save_versets_to_tfrecord(sequence, writer):
            feature = {
                'title': tf.train.Feature(int64_list=tf.train.Int64List(value=list(sequence['title']))),
                'phons': tf.train.Feature(int64_list=tf.train.Int64List(value=list(sequence['phons']))),
                'text': tf.train.Feature(int64_list=tf.train.Int64List(value=list(sequence['text']))),
                'line': tf.train.Feature(int64_list=tf.train.Int64List(value=[sequence['line']]))
            }
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())

        def save_sonnets(sequences, writer_train, writer_test):
            for seq in sequences:
                if random.random() > RATIO_TEST:
                    save_sonnets_to_writer(seq, writer_train)
                else:
                    save_sonnets_to_writer(seq, writer_test)
        def save_sonnets_to_writer(sequence, writer):
            feature = {
                'title': tf.train.Feature(int64_list=tf.train.Int64List(value=list(sequence['title'].tolist()))),
                'phons': tf.train.Feature(int64_list=tf.train.Int64List(value=list(sequence['phons'].tolist()))),
                'texts': tf.train.Feature(int64_list=tf.train.Int64List(value=list(sequence['texts'])))
            }
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())

        def generate_subsequences_tokenized(phon_sequence, start=2):
            subsequences = []
            for i in range(start, len(phon_sequence) + 1):
                subsequences.append(phon_sequence[:i])
            return subsequences

        def sequence_and_pad_model1(model_1_data):
            sequences = []
            for data in model_1_data:
                lines = [s.rjust(3, '.') for s in data['phons']]
                inversed_lines = [string[::-1] for string in lines]
                phons = "$" + ''.join(inversed_lines) + "€"

                phons_tokens = self.phon_tokenizer.texts_to_sequences([phons])[0]
                title_tokens = self.title_tokenizer.texts_to_sequences([data['title']])

                phon_subsequences = generate_subsequences_tokenized(phons_tokens)

                for subseq in phon_subsequences:
                    sequences.append({
                        'title': pad_sequences(title_tokens,
                                               maxlen=self.title_max_size, padding='post')[0],
                        'phons': pad_sequences([subseq],
                                               maxlen=self.phon_max_size * 14 + 2, padding='post')[0]
                    })
            return sequences

        def sequence_and_pad_model2(model_2_data, limit=False):
            sequences = []
            for data in model_2_data:
                line = data['phons'].rjust(3, '.')[::-1]

                title_tokens = self.title_tokenizer.texts_to_sequences([data['title']])[0]
                phons_tokens = self.phon_tokenizer.texts_to_sequences([line])[0]
                texts_tokens = self.text_tokenizer.texts_to_sequences([data['text']])[0]

                if limit and len(texts_tokens) >= 3:
                    text_subsequences = generate_subsequences_tokenized(texts_tokens, start=len(texts_tokens)-3)
                else:
                    text_subsequences = generate_subsequences_tokenized(texts_tokens)

                for subseq in text_subsequences:
                    sequences.append({
                        'title': pad_sequences([title_tokens],
                                               maxlen=self.title_max_size, padding='post')[0],
                        'phons': pad_sequences([phons_tokens],
                                               maxlen=self.phon_max_size, padding='post')[0],
                        'text': pad_sequences([subseq],
                                              maxlen=self.text_max_size, padding='post')[0],
                        'line': data['line']
                    })
            return sequences

        def sequence_and_pad_model3(model_3_data):
            sequences = []
            for data in model_3_data:
                lines = [phon.rjust(3, '.') for phon in data['phons']]
                inversed_lines = [string[::-1] for string in lines]
                phons = "$" + ''.join(inversed_lines) + "€"

                title_tokens = self.title_tokenizer.texts_to_sequences([data['title']])[0]
                phons_tokens = self.phon_tokenizer.texts_to_sequences([phons])[0]
                texts_tokens = self.text_tokenizer.texts_to_sequences(data['texts'])

                phons_subsequences = generate_subsequences_tokenized(phons_tokens)
                texts = list(np.array([pad_sequences([text_tokens], maxlen=self.text_max_size, padding='post')[0] for text_tokens in texts_tokens]).flatten())
                texts_subsequences = generate_subsequences_tokenized(texts)

                for i, text_subsequence in enumerate(texts_subsequences):
                    j = i//self.text_max_size
                    phon_sub = phons_subsequences[j * 3+1:(j + 1) * 3 + 1]
                    for phon in phon_sub:
                        sequences.append({
                            'title': pad_sequences([title_tokens], maxlen=self.title_max_size, padding='post')[0],
                            'phons': pad_sequences([phon], maxlen=self.phon_max_size * 14 + 2, padding='post')[0],
                            'texts': pad_sequences([text_subsequence],  maxlen=self.text_max_size * 14, padding='post')[0]
                        })
            return sequences

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

        print("Creating Titles")
        # Array des titres
        titles = [sonnet['title'] for sonnet in sonnets]

        print("Creating Lines")
        # Array des lines
        lines = [line['text'] for sonnet in sonnets for line in sonnet['lines']]
        with open("debug/lines.txt", "w") as f:
            for word in lines:
                f.write(word + '\n')

        print("Creating Phons")
        # Set des phons
        set_phons = {phon for sonnet in sonnets for line in sonnet['lines'] for phon in line['phon']}
        phons_list = list(set_phons)
        phons_list.sort(key=len, reverse=True)

        print("Creating Themes")
        # Dictionaire des themes
        themes = {}
        for sonnet in sonnets:
            theme = sonnet['theme']
            if theme in themes:
                themes[theme] += 1
            else:
                themes[theme] = 1

        print("Creating Phons with themes")
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

        print("Creating Word with Phons with themes")
        # On créé un dictionnaire des mots utilisés (2 max) en fonction des thèmes et des phons associés
        themes_phons_words = {}
        for theme in themes_phons:
            themes_phons_words[theme] = {}
            for phon in themes_phons[theme]:
                themes_phons_words[theme][phon] = set()

        phons_words = {}
        for phon in set_phons:
            phons_words[phon] = set()

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
                        phons_words[phon].add(
                            words[len(words) - index_word2 - 1] + " " + words[len(words) - index_word1 - 1])
                    else:
                        themes_phons_words[theme][phon].add(words[len(words) - index_word1 - 1])
                        phons_words[phon].add(words[len(words) - index_word1 - 1])

        print("Creating Schemas with themes")
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

        print("Tokenizing Text")
        # Tokenisation
        self.text_tokenizer = Tokenizer(filters='"#$%&()*+/:=?@[\\]^_`{|}~\t\n')
        self.text_tokenizer.fit_on_texts(lines)
        words_text = list(self.text_tokenizer.word_index.keys())
        with open("debug/words_text.txt", "w") as f:
            for word in words_text:
                f.write(word + '\n')

        print("Tokenizing Titles")
        self.title_tokenizer = Tokenizer()
        self.title_tokenizer.fit_on_texts(titles)
        words_title = list(self.title_tokenizer.word_index.keys())
        with open("debug/words_title.txt", "w") as f:
            for word in words_title:
                f.write(word + '\n')

        print("Tokenizing Phons")
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

        self.title_max_size = max([len(line.split()) for line in titles])
        self.title_words = len(words_title)

        self.text_max_size = max([len(line.split()) for line in lines])
        self.text_words = len(words_text)

        self.phon_max_size = max([len(phons) for phons in list(set_phons)])
        self.phon_words = len(words_phon)


        # On génère le premier Dataset du modèle 1
        # Model 1 - Black Data
        # Sur la base des données réelles
        # Title seq => 14 x Phon seq
        print("Generating - Model 1 - Black Data")
        options = tf.io.TFRecordOptions(compression_type=COMPRESSION_TYPE)
        with tf.io.TFRecordWriter(MODEL_1_SEQ_BLACK_PATH, options=options) as writer_train:
            with tf.io.TFRecordWriter(MODEL_1_SEQ_BLACK_TEST_PATH, options=options) as writer_test:
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

                    seq1 = sequence_and_pad_model1([sonnet_seq1])
                    save_phons(seq1, writer_train, writer_test)

                    if self.convertir_rimes_en_lettres(sonnet_seq2_array) == sonnet['schema']:
                        seq2 = sequence_and_pad_model1([sonnet_seq2])
                        save_phons(seq2, writer_train, writer_test)
                    if self.convertir_rimes_en_lettres(sonnet_seq3_array) == sonnet['schema']:
                        seq3 = sequence_and_pad_model1([sonnet_seq3])
                        save_phons(seq3, writer_train, writer_test)
        print("\rCounting data...")
        count_train = self.count_tfrecord_samples(MODEL_1_SEQ_BLACK_PATH)
        count_test = self.count_tfrecord_samples(MODEL_1_SEQ_BLACK_TEST_PATH)
        print("\rGenerated - Model 1 - {} Black Data - {} Black Test Data".format(count_train, count_test))

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
        options = tf.io.TFRecordOptions(compression_type=COMPRESSION_TYPE)
        with tf.io.TFRecordWriter(MODEL_1_SEQ_GREY_PATH, options=options) as writer_train:
            with tf.io.TFRecordWriter(MODEL_1_SEQ_GREY_TEST_PATH, options=options) as writer_test:
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
                    sonnet_seq = {
                        'title': '',
                        'phons': rimes
                    }
                    seq = sequence_and_pad_model1([sonnet_seq])
                    save_phons(seq, writer_train, writer_test)
                    #if len(model_1_data_grey) > 5000:
                    #    break
        print("\rCounting data...")
        count_train = self.count_tfrecord_samples(MODEL_1_SEQ_GREY_PATH)
        count_test = self.count_tfrecord_samples(MODEL_1_SEQ_GREY_TEST_PATH)
        print("\rGenerated - Model 1 - {} Grey Data - {} Grey Test Data".format(count_train, count_test))
        

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
        options = tf.io.TFRecordOptions(compression_type=COMPRESSION_TYPE)
        with tf.io.TFRecordWriter(MODEL_1_SEQ_WHITE_PATH, options=options) as writer_train:
            with tf.io.TFRecordWriter(MODEL_1_SEQ_WHITE_TEST_PATH, options=options) as writer_test:
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

                    sonnet_seq = {
                        'title': '',
                        'phons': rimes
                    }
                    seq = sequence_and_pad_model1([sonnet_seq])
                    save_phons(seq, writer_train, writer_test)
        print("\rCounting data...")
        count_train = self.count_tfrecord_samples(MODEL_1_SEQ_WHITE_PATH)
        count_test = self.count_tfrecord_samples(MODEL_1_SEQ_WHITE_TEST_PATH)
        print("\rGenerated - Model 1 - {} White Data - {} White Test Data".format(count_train, count_test))

        '''
        
        # On génère le premier Dataset du modèle 2
        # Model 2 - Black Data
        # Sur la base des données réelles
        # Title seq + Phon seq => Verset
        print("Generating - Model 2 - Black Data")
        options = tf.io.TFRecordOptions(compression_type=COMPRESSION_TYPE)
        with tf.io.TFRecordWriter(MODEL_2_SEQ_BLACK_PATH, options=options) as writer_train:
            with tf.io.TFRecordWriter(MODEL_2_SEQ_BLACK_TEST_PATH, options=options) as writer_test:
                for i, sonnet in enumerate(sonnets):
                    progress_percent = (i / len(sonnets)) * 100
                    print(f"\rGenerating progress: {progress_percent:.2f}%", end="")
                    title = sonnet['title']
                    for j, line in enumerate(sonnet['lines']):
                        for phon in line['phon']:
                            sonnet_seq = {
                                'title': title,
                                'phons': phon,
                                'text': line['text'],
                                'line': j+1
                            }
                            seq = sequence_and_pad_model2([sonnet_seq])
                            save_versets(seq, writer_train, writer_test)
        print("\rCounting data...")
        count_train = self.count_tfrecord_samples(MODEL_2_SEQ_BLACK_PATH)
        count_test = self.count_tfrecord_samples(MODEL_2_SEQ_BLACK_TEST_PATH)
        print("\rGenerated - Model 2 - {} Black Data - {} Black Test Data".format(count_train,count_test))

        # Model 2 - Grey Data
        # Sur la base des données réelles mais on fait varier les derniers mots sur la bases des autres mots pour le même thème et le même phon
        # Title seq + Phon seq => Verset
        print("Generating - Model 2 - Grey Data")

        options = tf.io.TFRecordOptions(compression_type=COMPRESSION_TYPE)
        with tf.io.TFRecordWriter(MODEL_2_SEQ_GREY_PATH, options=options) as writer_train:
            with tf.io.TFRecordWriter(MODEL_2_SEQ_GREY_TEST_PATH, options=options) as writer_test:
                for i, sonnet in enumerate(sonnets):
                    progress_percent = (i / len(sonnets)) * 100
                    print(f"\rGenerating progress: {progress_percent:.2f}%", end="")
                    title = sonnet['title']
                    theme = sonnet['theme']
                    for i, line in enumerate(sonnet['lines']):
                        for phon in line['phon']: # Il y a trois phons : un pour chaque type de rime (pauvre, riche, etc...)
                            words_dictionnary_phon = list(themes_phons_words[theme][phon])
                            num_samples = min(len(words_dictionnary_phon), 2)
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

                                sonnet_seq = {
                                    'title': title,
                                    'phons': phon,
                                    'text': '<start> ' + new_line + last_car + ' <end>',
                                    'line': i + 1
                                }
                                seq = sequence_and_pad_model2([sonnet_seq], limit=True) # On limite la génération à la fin de la séquence
                                save_versets(seq, writer_train, writer_test)
        print("\rCounting data...")
        count_train = self.count_tfrecord_samples(MODEL_2_SEQ_GREY_PATH)
        count_test = self.count_tfrecord_samples(MODEL_2_SEQ_GREY_TEST_PATH)
        print("\rGenerated - Model 2 - {} Grey Data - {} Grey Test Data".format(count_train, count_test))


        # Model 2 - White Data
        # Sur la base de données factices on fait varier les derniers mots
        # On choisi la longeur de la rime
        # On récupére autant de mots du dictionnaire en fonction de leur distribution
        # On construit une phrase qui se termine par le mot associé au phonème
        # Title seq + Phon seq => Verset
        print("Generating - Model 2 - White Data")
        dictionary = copy.deepcopy(self.text_tokenizer.word_counts)

        del dictionary['<start>']
        del dictionary['<end>']
        options = tf.io.TFRecordOptions(compression_type=COMPRESSION_TYPE)
        with tf.io.TFRecordWriter(MODEL_2_SEQ_WHITE_PATH, options=options) as writer_train:
            with tf.io.TFRecordWriter(MODEL_2_SEQ_WHITE_TEST_PATH, options=options) as writer_test:
                for i, phon in enumerate(phons_words):
                    progress_percent = (i / len(phons_words)) * 100
                    print(f"\rGenerating progress: {progress_percent:.2f}%", end="")
                    for word in phons_words[phon]:
                        for _ in range(3):
                            taille_verset = random.randint(0, self.text_max_size-2)

                            selected_words = random.choices(list(dictionary.keys()), weights=dictionary.values(), k=taille_verset)
                            line = ' '.join(selected_words) + ' ' + word

                            j = random.randint(0, 13)

                            sonnet_seq = {
                                'title': '',
                                'phons': phon,
                                'text': '<start> ' + line + ' <end>',
                                'line': j + 1
                            }
                            seq = sequence_and_pad_model2([sonnet_seq], limit=True)
                            save_versets(seq, writer_train, writer_test)
        print("\rCounting data...")
        count_train = self.count_tfrecord_samples(MODEL_2_SEQ_WHITE_PATH)
        count_test = self.count_tfrecord_samples(MODEL_2_SEQ_WHITE_TEST_PATH)
        print("\rGenerated - Model 2 - {} White Data - {} Grey Test Data".format(count_train, count_test))
        
        # On génère le Dataset du modèle 3
        # Model 3 - Black Data
        # Sur la base des données réelles
        # Title seq + Phon seq => 14 Versets + 14 Phons
        print("Generating - Model 3 - Black Data")
        options = tf.io.TFRecordOptions(compression_type=COMPRESSION_TYPE)
        with tf.io.TFRecordWriter(MODEL_3_SEQ_BLACK_PATH, options=options) as writer_train:
            with tf.io.TFRecordWriter(MODEL_3_SEQ_BLACK_TEST_PATH, options=options) as writer_test:
                for i, sonnet in enumerate(sonnets):
                    progress_percent = (i / len(sonnets)) * 100
                    print(f"\rGenerating progress: {progress_percent:.2f}%", end="")
                    phons = [line['phon'] for line in sonnet['lines']]
                    zipped_phons = list(zip(*phons))
                    texts = [line['text'] for line in sonnet['lines']]

                    sonnet_seq1 = {
                        'title': sonnet['title'],
                        'phons': zipped_phons[0],
                        'texts': texts
                    }
                    seq1 = sequence_and_pad_model3([sonnet_seq1])
                    save_sonnets(seq1, writer_train, writer_test)

                    sonnet_seq2 = {
                        'title': sonnet['title'],
                        'phons': zipped_phons[1],
                        'texts': texts
                    }
                    seq2 = sequence_and_pad_model3([sonnet_seq2])
                    save_sonnets(seq2, writer_train, writer_test)

                    sonnet_seq3 = {
                        'title': sonnet['title'],
                        'phons': zipped_phons[2],
                        'texts': texts
                    }
                    seq3 = sequence_and_pad_model3([sonnet_seq3])
                    save_sonnets(seq3, writer_train, writer_test)

        print("\rCounting data...")
        count_train = self.count_tfrecord_samples(MODEL_3_SEQ_BLACK_PATH)
        count_test = self.count_tfrecord_samples(MODEL_3_SEQ_BLACK_TEST_PATH)
        print("\rGenerated - Model 3 - {} Black Data - {} Black Test Data".format(count_train, count_test))
        '''
        print("End analyze")

    def save(self):
        print("Start Saving")

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
            'phons': tf.io.FixedLenFeature([phon_max_size], tf.int64),
            'text': tf.io.FixedLenFeature([text_max_size], tf.int64),
            'line': tf.io.FixedLenFeature([1], tf.int64),
        }

        def _parse_function(example_proto):
            return tf.io.parse_single_example(example_proto, feature)

        raw_dataset = tf.data.TFRecordDataset(filename, compression_type=COMPRESSION_TYPE)
        parsed_dataset = raw_dataset.map(_parse_function, num_parallel_calls=tf.data.AUTOTUNE).take(
            batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
        return parsed_dataset

    def load_sonnets_from_tfrecord(self, filename, batch_size, title_max_size, phon_max_size, text_max_size):
        feature = {
            'title': tf.io.FixedLenFeature([title_max_size], tf.int64),
            'phons': tf.io.FixedLenFeature([phon_max_size*14+2], tf.int64),
            'texts': tf.io.FixedLenFeature([text_max_size*14+2], tf.int64),
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

        if epoch <= initial_stable_epochs:
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

        #data_black = 1.0
        #data_grey = 0.0
        #data_white = 0.0

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
                merged_data = white_data.concatenate(grey_data).concatenate(black_data)
            else:
                #white_data = self.load_rimes_from_tfrecord(MODEL_1_SEQ_WHITE_TEST_PATH, batch_white, self.title_max_size, self.phon_max_size)
                #grey_data = self.load_rimes_from_tfrecord(MODEL_1_SEQ_GREY_TEST_PATH, batch_grey, self.title_max_size, self.phon_max_size)
                black_data = self.load_rimes_from_tfrecord(MODEL_1_SEQ_BLACK_TEST_PATH, batch_size,
                                                           self.title_max_size, self.phon_max_size)

                merged_data = black_data


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
                                                           self.phon_max_size, self.text_max_size)
                grey_data = self.load_versets_from_tfrecord(MODEL_2_SEQ_GREY_PATH, batch_grey, self.title_max_size,
                                                          self.phon_max_size, self.text_max_size)
                black_data = self.load_versets_from_tfrecord(MODEL_2_SEQ_BLACK_PATH, batch_black, self.title_max_size,
                                                           self.phon_max_size, self.text_max_size)
                merged_data = white_data.concatenate(grey_data).concatenate(black_data)
            else:
                #white_data = self.load_versets_from_tfrecord(MODEL_2_SEQ_WHITE_TEST_PATH, batch_white, self.title_max_size, self.phon_max_size, self.text_max_size)
                #grey_data = self.load_versets_from_tfrecord(MODEL_2_SEQ_GREY_TEST_PATH, batch_grey, self.title_max_size, self.phon_max_size, self.text_max_size)
                black_data = self.load_versets_from_tfrecord(MODEL_2_SEQ_BLACK_TEST_PATH, batch_size, self.title_max_size, self.phon_max_size, self.text_max_size)
                merged_data = black_data


            for batch in merged_data.batch(batch_size):
                # print("Start generate_data batch")
                input_title = batch['title']
                input_phons = batch['phons']
                input_line = batch['line']
                target_text = batch['text']

                input_phons_onehot = tf.keras.utils.to_categorical(input_phons, num_classes=self.phon_words+1)

                decoder_input = target_text[:, :-1]
                decoder_output = target_text[:, 1:]

                decoder_output_onehot = tf.keras.utils.to_categorical(decoder_output, num_classes=self.text_words+1)
                yield (input_title, input_phons_onehot, input_line, decoder_input), decoder_output_onehot

    def generate_data_sonnet(self, batch_size, epoch_size, epoch, train):
        # print("\nStart generate_data_verset epoch :{}".format(epoch))
        batch_black = 1

        for i in range(epoch_size):
            # print("Start batch {} {}".format(i+1, epoch_size))
            if train == True:
                black_data = self.load_sonnets_from_tfrecord(MODEL_3_SEQ_BLACK_PATH, batch_black, self.title_max_size,
                                                           self.phon_max_size, self.text_max_size)
            else:
                black_data = self.load_sonnets_from_tfrecord(MODEL_3_SEQ_BLACK_TEST_PATH, batch_black,
                                                           self.title_max_size, self.phon_max_size, self.text_max_size)


            for batch in black_data.batch(batch_size):
                # print("Start generate_data batch")
                input_title = batch['title']
                input_phons = batch['phons']
                target_text = batch['texts']

                input_phons_onehot = tf.keras.utils.to_categorical(input_phons[:, :-1], num_classes=self.phon_words + 1)
                output_phons_onehot = tf.keras.utils.to_categorical(input_phons[:, 1:], num_classes=self.phon_words + 1)

                input_text_split = tf.split(target_text, num_or_size_splits=14, axis=1)
                input_texts = []
                output_texts = []
                for i in range(14):
                    input_texts.append(input_text_split[i][:, :-1])
                    output_texts.append(input_text_split[i][:, 1:])

                output_text = tf.concat(output_texts, axis=1)
                decoder_input_text = tf.concat(input_texts, axis=1)
                #decoder_input_text = target_text[:, :-1]
                #output_text = target_text[:, 1:]

                output_text_onehot = tf.keras.utils.to_categorical(output_text, num_classes=self.text_words+1)
                #print((input_title.shape, input_phons_onehot.shape, decoder_input_text.shape), (output_phons_onehot.shape, output_text_onehot.shape))
                yield (input_title, input_phons_onehot, decoder_input_text), (output_phons_onehot, output_text_onehot)
