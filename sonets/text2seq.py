import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json

# Tokenize et pad sequences
def tokenize_and_pad(texts, max_len=None):
    tokenizer = Tokenizer(filters='\'…‘~"#$%&()*+-=@[\]`{|}~�■µ\\–»«•1234567890')
    tokenizer.fit_on_texts(texts)
    total_words = len(tokenizer.word_index) + 1
    sequences = tokenizer.texts_to_sequences(texts)
    if not max_len:
        max_len = max([len(seq) for seq in sequences])
    padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post')
    return tokenizer, padded_sequences, total_words, max_len

# Sauvegardez le tokenizer (dictionnaire) pour une utilisation future
def save_tokenizer(tokenizer, filename):
    tokenizer_json = tokenizer.to_json()
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(tokenizer_json)

def load_tokenizer(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        tokenizer_json = f.read()
    tokenizer = tokenizer_from_json(tokenizer_json)
    return tokenizer

def save_max_len(max_len, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump({'max_len':max_len}, f)

def load_max_len(filename):
    max_len = {'max_len': 0}
    with open(filename, 'r', encoding='utf-8') as f:
        max_len = json.load(f)
    return max_len['max_len']

# Enregistrer les séquences dans TFRecords
def save_to_tfrecord(sequences, filename):
    options = tf.io.TFRecordOptions(compression_type="GZIP")
    with tf.io.TFRecordWriter(filename, options=options) as writer:
        for seq in sequences:
            feature = {
                'sequence': tf.train.Feature(int64_list=tf.train.Int64List(value=seq))
            }
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())

# Charger les séquences depuis TFRecords
def load_from_tfrecord(filename, max_len, batch_size):
    feature_description = {
        'sequence': tf.io.FixedLenFeature([max_len], tf.int64)
    }

    def _parse_function(example_proto):
        return tf.io.parse_single_example(example_proto, feature_description)

    raw_dataset = tf.data.TFRecordDataset(filename, compression_type="GZIP")
    parsed_dataset = raw_dataset.map(_parse_function).shuffle(10000).batch(batch_size)
    return parsed_dataset

def count_tfrecord_samples(filename):
    count = 0
    for _ in tf.data.TFRecordDataset(filename, compression_type="GZIP"):
        count += 1
    return count

def text2seq(sonnets):
    tokenizer, padded_sequences, total_words, max_len = tokenize_and_pad(sonnets)
    save_tokenizer(tokenizer, "tokenizer.json")
    save_max_len(max_len, "max_len.json")
    size = int(len(padded_sequences) / 100)
    save_to_tfrecord(padded_sequences[:-size], "data.tfrecord")
    save_to_tfrecord(padded_sequences[-size:], "data_val.tfrecord")
