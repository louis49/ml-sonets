import os, json, re
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
from keras.src.preprocessing.text import tokenizer_from_json

TEXT_PATH1 = "dataset/rhymes_1.json"
TEXT_PATH2 = "dataset/rhymes_2.json"
TEXT_PATH3 = "dataset/rhymes_3.json"
TITLE_PATH = "dataset/bd_meta.json"
rhymes = []
CHARS_TO_REMOVE = ['', '■', '¡', '\x8a', '—', '�', '…', '«', '°', 'µ', '•', '–', '»', '„', '©']
QUOTES_TO_REPLACE = ['‘', '’']

#'’', '‘'
def space_out_phon(phon):
    # Utilise une expression régulière pour insérer un espace entre chaque caractère
    spaced_phon = re.sub(r"(.)", r"\1 ", phon)
    return spaced_phon.strip()

def add_spaces(chaine):
    chaine = re.sub(r'(?<=[^\s])([?:.!,;])', r' \1 ', chaine)
    return re.sub(r'(?<=[^\s])([\'’])', r'\1 ', chaine)

def clean(line):
    clean_line = line
    for char in CHARS_TO_REMOVE:
        clean_line = clean_line.replace(char, ' ')
    for char in QUOTES_TO_REPLACE:
        clean_line = clean_line.replace(char, '\'')
    return clean_line

def remove_spaces(chaine):
    return re.sub(r'^\s+|\s+', ' ', chaine).strip()

def consolidate_data():

    consolidated_data = []

    def convert_to_line_index(strophe_number, line_number):
        # Vérifiez vos numéros de strophes et de lignes, ils peuvent commencer à 1 ou à une autre valeur selon votre cas.
        # S'ils commencent à 1, vous pouvez soustraire 1 de chaque pour travailler avec des index basés sur 0.

        # Le nombre total de lignes avant la strophe actuelle. Nous supposons ici qu'il y a 4 strophes,
        # les trois premières ayant 4 lignes et la dernière ayant 3 lignes.
        if strophe_number in [1, 2]:
            total_lines_before = (strophe_number - 1) * 4
        elif strophe_number == 3:
            total_lines_before = 2 * 4
        else:
            total_lines_before = 2 * 4 + 3# Les 2 premières strophes ont 4 lignes chacune.

        # Calcul de l'index de la ligne globale
        line_index = total_lines_before + (line_number - 1)

        return line_index

    # Fonction pour lire et ajouter les données d'un fichier JSON
    def add_data_from_file(filepath):
        with open(filepath, "r") as file:
            data = json.load(file)
            for _, value_list in data.items():
                consolidated_data.extend(value_list)

    # Lecture et ajout des données des trois fichiers
    add_data_from_file(TEXT_PATH1)
    add_data_from_file(TEXT_PATH2)
    add_data_from_file(TEXT_PATH3)

    consolidated_data = sorted(consolidated_data, key=lambda x: x['id'])#[:10000]

    with open(TITLE_PATH, "r") as file:
        meta = json.load(file)

    versets = []
    for data in consolidated_data:
        versets.append({
            'text': '<start> ' + remove_spaces(add_spaces(clean(data['text']))) + ' <end>',
            'phon': space_out_phon(data['phon']),
            'title': remove_spaces(add_spaces(clean(meta[data['id_sonnet']]['titre sonnet'])))
        })

    # TEXT

    # 1. Tokenisation
    text_tokenizer = Tokenizer(filters='"#$%&()*+/:=?@[\\]^_`{|}~\t\n')
    text_tokenizer.fit_on_texts([d['text'] for d in versets])

    # 2. Déterminer la longueur maximale
    max_text_length = max([len(d['text'].split()) for d in versets])

    # 3. Créer des séquences
    text_sequences = text_tokenizer.texts_to_sequences([d['text'] for d in versets])

    # 4. Padding
    text_padded = pad_sequences(text_sequences, maxlen=max_text_length, padding='post')

    # Récupérer les mots du tokenizer
    words_text = list(text_tokenizer.word_index.keys())

    # Enregistrer les mots dans un fichier texte
    with open("debug/words_text.txt", "w") as f:
        for word in words_text:
            f.write(word + '\n')

    # TITLE

    # 1. Tokenisation
    title_tokenizer = Tokenizer()
    title_tokenizer.fit_on_texts([d['title'] for d in versets])

    # 2. Déterminer la longueur maximale
    max_title_length = max([len(d['title'].split()) for d in versets])

    # 3. Créer des séquences
    title_sequences = title_tokenizer.texts_to_sequences([d['title'] for d in versets])

    # 4. Padding
    title_padded = pad_sequences(title_sequences, maxlen=max_title_length, padding='post')

    # Récupérer les mots du tokenizer
    words_title = list(title_tokenizer.word_index.keys())

    # Enregistrer les mots dans un fichier texte
    with open("debug/words_title.txt", "w") as f:
        for word in words_title:
            f.write(word + '\n')

    # PHON

    # 1. Tokenisation
    phon_tokenizer = Tokenizer(filters=" ")
    phon_tokenizer.fit_on_texts([d['phon'] for d in versets])

    # 2. Déterminer la longueur maximale
    max_phon_length = max([len(d['phon'].split()) for d in versets])

    # 3. Créer des séquences
    phon_sequences = phon_tokenizer.texts_to_sequences([d['phon'] for d in versets])

    # 4. Padding
    phon_padded = pad_sequences(phon_sequences, maxlen=max_phon_length, padding='post')

    # Récupérer les mots du tokenizer
    phon_words = list(phon_tokenizer.word_index.keys())

    # Enregistrer les mots dans un fichier texte
    with open("debug/words_phon.txt", "w") as f:
        for word in phon_words:
            f.write(word + '\n')

    size = int(len(title_padded) / 100)
    save_versets_to_tfrecord(title_padded[:-size], text_padded[:-size], phon_padded[:-size], "data/data_verset.tfrecord")
    save_versets_to_tfrecord(title_padded[-size:], text_padded[-size:], phon_padded[-size:], "data/data_val_verset.tfrecord")

    # Enregistrons les tailles maximum de chaque séquence
    save_max_len(title_max_len=max_title_length,
                 text_max_len=max_text_length,
                 phon_max_len=max_phon_length,
                 title_words=len(title_tokenizer.word_index),
                 text_words=len(text_tokenizer.word_index),
                 phon_words=len(phon_tokenizer.word_index),
                 filename="data/verset_max_len.json")

    save_tokenizer(phon_tokenizer, "data/phon_tokenizer.json")
    save_tokenizer(text_tokenizer, "data/text_tokenizer.json")
    save_tokenizer(title_tokenizer, "data/title_tokenizer.json")

    # Sonnets & rimes

    sonnets = {}
    for item in consolidated_data:
        # Extraction des informations d'identification
        parts = item['id'].split('-')
        sonnet_id = parts[0] + "-" + parts[1]
        strophe_number = parts[2]
        line_number = parts[3]
        phon = item['phon']
        text = item['text']

        if sonnet_id not in sonnets:
            titre = remove_spaces(add_spaces(clean(meta[item['id_sonnet']]['titre sonnet'])))
            sonnets[sonnet_id] = {'lines': [{'phon': []} for _ in range(14)],
                                  'titre': titre,
                                  }

        line = convert_to_line_index(int(strophe_number), int(line_number))
        sonnets[sonnet_id]['lines'][line]['text'] = '<start> ' + remove_spaces(add_spaces(clean(text))) + ' <end>',
        sonnets[sonnet_id]['lines'][line]['phon'].append(space_out_phon(phon))

    sonnets_global = [sonnets[item] for item in sonnets]
    sonnets_seq = []
    for sonnet in sonnets_global:
        title_seq = title_tokenizer.texts_to_sequences([sonnet['titre']])[0]
        title_seq_pad = pad_sequences([title_seq], maxlen=max_title_length, padding='post')[0]
        lines_1 = []
        lines_2 = []
        lines_3 = []
        for line in sonnet['lines']:
            phon1seq = phon_tokenizer.texts_to_sequences([line['phon'][0]])[0]
            phon2seq = phon_tokenizer.texts_to_sequences([line['phon'][1]])[0]
            phon3seq = phon_tokenizer.texts_to_sequences([line['phon'][2]])[0]
            textseq = text_tokenizer.texts_to_sequences(line['text'])[0]
            padtextseq = pad_sequences([textseq], maxlen=max_text_length, padding='post')[0]
            lines_1.append({
                'phon': pad_sequences([phon1seq], maxlen=max_phon_length, padding='post')[0],
                'text': padtextseq,
            })
            lines_2.append({
                'phon': pad_sequences([phon2seq], maxlen=max_phon_length, padding='post')[0],
                'text': padtextseq,
            })
            lines_3.append({
                'phon': pad_sequences([phon3seq], maxlen=max_phon_length, padding='post')[0],
                'text': padtextseq,
            })

        sonnets_seq.append({
            'title': title_seq_pad,
            'lines': lines_1
        })
        sonnets_seq.append({
            'title': title_seq_pad,
            'lines': lines_2
        })
        sonnets_seq.append({
            'title': title_seq_pad,
            'lines': lines_3
        })

    size = int(len(sonnets_seq) / 100)
    save_sonnets_to_tfrecord(sonnets_seq[:-size], "data/data_sonnet.tfrecord")
    save_sonnets_to_tfrecord(sonnets_seq[-size:], "data/data_val_sonnet.tfrecord")

    print("")

def save_max_len(title_max_len, text_max_len, phon_max_len, title_words, text_words, phon_words,filename):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump({
            'title_max_len': title_max_len,
            'text_max_len': text_max_len,
            'phon_max_len': phon_max_len,
            'title_words': title_words,
            'text_words': text_words,
            'phon_words': phon_words
        }, f)

def load_max_len(filename):
    max_len = {
        'title_max_len': 0,
        'text_max_len': 0,
        'phon_max_len': 0,
        'title_words': 0,
        'text_words': 0,
        'phon_words': 0
    }
    with open(filename, 'r', encoding='utf-8') as f:
        max_len = json.load(f)
    return max_len

def save_sonnets_to_tfrecord(sonnets_sequences, filename):
    options = tf.io.TFRecordOptions(compression_type="GZIP")
    with tf.io.TFRecordWriter(filename, options=options) as writer:
        for sonnet_sequences in sonnets_sequences:
            feature = {
                'title': tf.train.Feature(int64_list=tf.train.Int64List(value=sonnet_sequences['title'])),
            }
            for i, line in enumerate(sonnet_sequences['lines']):
                feature[f'phon_{i + 1}'] = tf.train.Feature(int64_list=tf.train.Int64List(value=line['phon']))
                feature[f'text_{i + 1}'] = tf.train.Feature(int64_list=tf.train.Int64List(value=line['text']))

            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())

def load_sonnets_from_tfrecord(filename, batch_size, max_len_title, max_len_text, max_len_phon ):
    def _parse_function(example_proto):
        # Définir le dictionnaire de votre schéma ici
        keys_to_features = {
            'title': tf.io.FixedLenFeature([max_len_title], tf.int64),
        }

        # Pour chaque ligne, vous avez stocké 'phon' et 'text' avec des noms de clé spécifiques
        for i in range(1, 15):  # si vous avez 14 lignes
            keys_to_features[f'phon_{i}'] = tf.io.FixedLenFeature([max_len_phon], tf.int64)
            keys_to_features[f'text_{i}'] = tf.io.FixedLenFeature([max_len_text], tf.int64)

        # Charger un des exemples
        parsed_features = tf.io.parse_single_example(example_proto, keys_to_features)

        # Vous pouvez effectuer tout traitement supplémentaire et renvoyer les données nécessaires ici
        return parsed_features

    raw_dataset = tf.data.TFRecordDataset(filename, compression_type="GZIP")
    parsed_dataset = raw_dataset.map(_parse_function).shuffle(100000).batch(batch_size)
    return parsed_dataset

def save_versets_to_tfrecord(title_sequences, text_sequences, phon_sequences, filename):
    options = tf.io.TFRecordOptions(compression_type="GZIP")
    with tf.io.TFRecordWriter(filename, options=options) as writer:
        for title_seq, text_seq, phon_seq in zip(title_sequences, text_sequences, phon_sequences):
            feature = {
                'title': tf.train.Feature(int64_list=tf.train.Int64List(value=title_seq)),
                'text': tf.train.Feature(int64_list=tf.train.Int64List(value=text_seq)),
                'phon': tf.train.Feature(int64_list=tf.train.Int64List(value=phon_seq))
            }
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())

def load_versets_from_tfrecord(filename, batch_size, max_len_title, max_len_text, max_len_phon):
    feature = {
        'title': tf.io.FixedLenFeature([max_len_title], tf.int64),
        'text': tf.io.FixedLenFeature([max_len_text], tf.int64),
        'phon': tf.io.FixedLenFeature([max_len_phon], tf.int64)
    }

    def _parse_function(example_proto):
        return tf.io.parse_single_example(example_proto, feature)

    raw_dataset = tf.data.TFRecordDataset(filename, compression_type="GZIP")
    parsed_dataset = raw_dataset.map(_parse_function).shuffle(100000).batch(batch_size)
    return parsed_dataset

def count_tfrecord_samples(filename):
    count = 0
    for _ in tf.data.TFRecordDataset(filename, compression_type="GZIP"):
        count += 1
    return count

def save_tokenizer(tokenizer, filename):
    tokenizer_json = tokenizer.to_json()
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(tokenizer_json)

def load_tokenizer(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        tokenizer_json = f.read()
    tokenizer = tokenizer_from_json(tokenizer_json)
    return tokenizer