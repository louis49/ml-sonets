import random, json
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.src.preprocessing.text import tokenizer_from_json

RATIO_TEST = 0.005
TOKENIZER_WORD_PATH = "./learn_phonemes/data/tokenizer_word_NEW2.json"
TOKENIZER_PHONS_PATH = "./learn_phonemes/data/tokenizer_phons_NEW2.json"
SIZE_PATH = "./learn_phonemes/data/sizes_NEW2.json"
DATA_LEARN_PATH = "./learn_phonemes/data/word_phon_learn_NEW2.tfrecord"
DATA_TEST_PATH = "./learn_phonemes/data/word_phon_test_NEW2.tfrecord"
DATA_PATH = "./learn_phonemes/data/word_phon_NEW2.tfrecord"
COMPRESSION_TYPE = "GZIP"

nasal_mapping = {
    'ɑ̃': '1',
    'ɛ̃': '2',
    'œ̃': '3',
    'ɔ̃': '4'
}
class Data():
    def __init__(self):
        print("Init Data")
        self.word_tokenizer = None
        self.phons_tokenizer = None
        self.word_max_size = 0
        self.phons_max_size = 0
        self.word_index_size = 0
        self.phons_index_size = 0
        self.records_learn = 0
        self.records_test = 0

    def analyze(self, dictionnaire):
        def write_tfrecord(sequence, writer):
            feature = {
                'word': tf.train.Feature(int64_list=tf.train.Int64List(value=list(sequence['word']))),
                'phons': tf.train.Feature(int64_list=tf.train.Int64List(value=list(sequence['phons'])))
            }
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())

        def save_tokenizer(tokenizer, filename):
            tokenizer_json = tokenizer.to_json()
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(tokenizer_json)

        def count_tfrecord_samples(filename):
            count = 0
            for _ in tf.data.TFRecordDataset(filename, compression_type=COMPRESSION_TYPE):
                count += 1
            return count

        def replace_nasals(text):
            for nasal, symbol in nasal_mapping.items():
                text = text.replace(nasal, symbol)
            return text

        words = []
        for word in dictionnaire.keys():
            words.append(word)
        self.word_tokenizer = Tokenizer(filters='"#$%&()*+/:=?@[\\]^_`{|}~\t\n.\'', char_level=True)
        self.word_tokenizer.fit_on_texts(words)

        with open("lettre.txt", 'w', encoding='utf-8') as f:
            for key in self.word_tokenizer.index_word:
                f.write(f'{self.word_tokenizer.index_word[key]}\n')

        phons = []
        with open("prononciation.txt", 'w', encoding='utf-8') as f:
            for word in dictionnaire:
                phon = replace_nasals(dictionnaire[word]['phon'])
                phons.append('$' + phon + '#')
                f.write(phon + '\n')
        self.phons_tokenizer = Tokenizer(filters='', char_level=True, lower=False)
        self.phons_tokenizer.fit_on_texts(phons)

        with open("phons.txt", 'w', encoding='utf-8') as f:
            for key in self.phons_tokenizer.index_word:
                f.write(f'{self.phons_tokenizer.index_word[key]}\n')


        save_tokenizer(self.word_tokenizer, TOKENIZER_WORD_PATH)
        save_tokenizer(self.phons_tokenizer, TOKENIZER_PHONS_PATH)

        sequences_word = self.word_tokenizer.texts_to_sequences(words)
        sequences_phons = self.phons_tokenizer.texts_to_sequences(phons)

        self.word_max_size = max([len(word) for word in sequences_word])
        self.phons_max_size = max([len(phons) for phons in sequences_phons])
        self.word_index_size = len(list(self.word_tokenizer.word_index.keys()))
        self.phons_index_size = len(list(self.phons_tokenizer.word_index.keys()))

        def generate_subsequences_tokenized(phon_sequence, start=2):
            subsequences = []
            for i in range(start, len(phon_sequence) + 1):
                subsequences.append(phon_sequence[:i])
            return subsequences

        def write_data(writer, sequences):
            mapping = []
            for sequence in sequences:
                subsequences_phons = generate_subsequences_tokenized(sequence['phons'])
                sequence_word_padded = pad_sequences([sequence['word']], maxlen=self.word_max_size, padding='post')[0]
                for subsequence_phons in subsequences_phons:
                    sequence_phons_padded = pad_sequences([subsequence_phons], maxlen=self.phons_max_size, padding='post')[0]
                    mapping.append({'word': sequence_word_padded, 'phons': sequence_phons_padded})
            for sequence in mapping:
                write_tfrecord(sequence=sequence, writer=writer)

        sequences = []
        for sequence_word, sequence_phons in zip(sequences_word, sequences_phons):
            sequences.append({'word': sequence_word, 'phons': sequence_phons})
        random.shuffle(sequences)
        sequences = sequences[:100000]

        index_part = int(len(sequences) * RATIO_TEST)
        data_learn = sequences[index_part:]
        data_test = sequences[:index_part]

        word = ''.join(self.word_tokenizer.sequences_to_texts([data_learn[0]['word']])[0].split(' '))
        phon = ''.join(self.phons_tokenizer.sequences_to_texts([data_learn[0]['phons']])[0].replace('$','').replace('#','').split(' '))
        print("Le mot " + word + " a été ajouté avec la phonétique " + phon)

        options = tf.io.TFRecordOptions(compression_type=COMPRESSION_TYPE)
        with tf.io.TFRecordWriter(DATA_LEARN_PATH, options=options) as writer_learn:
            write_data(writer_learn, data_learn)
        with tf.io.TFRecordWriter(DATA_TEST_PATH, options=options) as writer_test:
            write_data(writer_test, data_test)
        with tf.io.TFRecordWriter(DATA_PATH, options=options) as writer:
            write_data(writer, sequences)

        self.records_learn = count_tfrecord_samples(DATA_LEARN_PATH)
        self.records_test = count_tfrecord_samples(DATA_TEST_PATH)


        with open(SIZE_PATH, 'w', encoding='utf-8') as file:
            json.dump({
                'word_max_size': self.word_max_size,
                'phons_max_size': self.phons_max_size,
                'word_index_size': self.word_index_size + 1,  # + 1 pour le 0
                'phons_index_size': self.phons_index_size + 1,  # + 1 pour le 0
                'records_learn': self.records_learn,
                'records_test': self.records_test
            }, file)


    def load(self,):
        def load_tokenizer(filename):
            with open(filename, 'r', encoding='utf-8') as f:
                tokenizer_json = f.read()
            tokenizer = tokenizer_from_json(tokenizer_json)
            return tokenizer

        self.word_tokenizer = load_tokenizer(TOKENIZER_WORD_PATH)
        self.phons_tokenizer = load_tokenizer(TOKENIZER_PHONS_PATH)

        with open(SIZE_PATH, 'r', encoding='utf-8') as f:
            sizes = json.load(f)
            self.word_max_size = sizes['word_max_size']
            self.phons_max_size = sizes['phons_max_size']
            self.word_index_size = sizes['word_index_size']
            self.phons_index_size = sizes['phons_index_size']
            self.records_learn = sizes['records_learn']
            self.records_test = sizes['records_test']

    def load_from_tfrecord(self, filename, batch_size, word_max_size, phons_max_size):
        feature = {
            'word': tf.io.FixedLenFeature([word_max_size], tf.int64),
            'phons': tf.io.FixedLenFeature([phons_max_size], tf.int64)
        }

        def _parse_function(example_proto):
            return tf.io.parse_single_example(example_proto, feature)

        raw_dataset = tf.data.TFRecordDataset(filename, compression_type=COMPRESSION_TYPE)
        parsed_dataset = raw_dataset.map(_parse_function, num_parallel_calls=tf.data.AUTOTUNE).take(
            batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
        return parsed_dataset

    def generate_data(self, batch_size, train):
        if train:
            data = self.load_from_tfrecord(DATA_LEARN_PATH, batch_size, self.word_max_size, self.phons_max_size)
        else:
            data = self.load_from_tfrecord(DATA_TEST_PATH, batch_size, self.word_max_size, self.phons_max_size)

        for batch in data.batch(batch_size):
            input = batch['word']
            target_text = batch['phons']
            decoder_input = target_text[:, :-1]
            decoder_output = target_text[:, 1:]

            decoder_output_one_hot = tf.keras.utils.to_categorical(decoder_output, num_classes=self.phons_index_size)
            yield (input, decoder_input), decoder_output_one_hot