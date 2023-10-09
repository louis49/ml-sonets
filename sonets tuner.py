import xml.etree.ElementTree as ET

import numpy as np
import tensorflow as tf
from keras import Model, Input
from keras.src.layers import TimeDistributed
from keras_tuner import BayesianOptimization
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
from tensorflow.keras import layers
from tensorflow import keras
from sklearn.model_selection import train_test_split

def sonnet_to_dict(sonnet):
    # Obtenir le titre du sonnet
    namespace = {'tei': 'http://www.tei-c.org/ns/1.0'}

    sonnet_str = ET.tostring(sonnet, encoding='utf-8').decode('utf-8')
    title = sonnet.find('tei:head', namespaces=namespace).text.strip()

    # Obtenir les strophes et les lignes du sonnet
    strophes = []
    for lg in sonnet.findall('tei:lg', namespaces=namespace):
        lines = [line.text.strip() for line in lg.findall('tei:l', namespaces=namespace)]
        strophes.append(lines)

    return {'title': title, 'lines': strophes}


# Enlever les espaces de noms
ET.register_namespace('', 'http://www.tei-c.org/ns/1.0')

# Charger le XML depuis le fichier
tree = ET.parse('sonnets_oupoco_tei.xml')
root = tree.getroot()
namespace = {'tei': 'http://www.tei-c.org/ns/1.0'}
sonnets = root.findall(".//tei:div[@type='sonnet']", namespace)

# Convertir chaque élément sonnet en dictionnaire
sonnets_dicts = [sonnet_to_dict(sonnet) for sonnet in sonnets][:2000]

# Prétraitement des données
titles = [sonnet["title"] for sonnet in sonnets_dicts]
sonnets_blocks = [' '.join([' '.join(strophe) for strophe in sonnet["lines"]]) for sonnet in sonnets_dicts]

# Tokenisation
tokenizer = Tokenizer(filters='"#$%&()*+-/:;<=>@[\]^_`{|}~')
tokenizer.fit_on_texts(titles + sonnets_blocks)
total_words = len(tokenizer.word_index) + 1 #55268

input_sequences = tokenizer.texts_to_sequences(titles)
max_input_len = max([len(seq) for seq in input_sequences]) #33
input_sequences = pad_sequences(input_sequences, maxlen=max_input_len)

output_sequences = tokenizer.texts_to_sequences(sonnets_blocks)
max_output_len = max([len(seq) for seq in output_sequences]) #152
output_sequences = pad_sequences(output_sequences, maxlen=max_output_len)

# Pour le décodeur (entrée), nous excluons la dernière partie de chaque séquence
decoder_input_data = output_sequences[:, :-1]
decoder_input_data = pad_sequences(decoder_input_data, maxlen=max_output_len, padding='post')

# Pour le décodeur (sortie), nous excluons la première partie de chaque séquence
decoder_output_data_shifted = output_sequences[:, 1:]
decoder_output_data_shifted = pad_sequences(decoder_output_data_shifted, maxlen=max_output_len, padding='post')
decoder_output_data = tf.keras.utils.to_categorical(decoder_output_data_shifted, num_classes=total_words)

# définir la taille de votre batch
BATCH_SIZE = 5
# Hyperparamètres initiaux et configuration du modèle
def build_model(hp):
    # Encodeur
    LSTM_MAX = 128
    DIM_MAX = 128

    DROP_OUT = 0.5
    REG = 0.001

    lstm_units = hp.Int("lstm_units", min_value=128, max_value=LSTM_MAX, step=8, default=LSTM_MAX) #8-128
    embedding_dim = hp.Int("embedding_dim", min_value=128, max_value=DIM_MAX, step=8, default=DIM_MAX) #8-128

    encoder_input = keras.Input(shape=(max_input_len,))
    encoder_embedding = layers.Embedding(
        total_words,
        embedding_dim
    )(encoder_input)
    encoder_embedding = layers.LayerNormalization()(encoder_embedding)
    encoder_output, state_h, state_c = layers.LSTM(
        lstm_units,
        return_state=True,
        #dropout=DROP_OUT,
        #recurrent_dropout=DROP_OUT,
        kernel_regularizer=keras.regularizers.l2(REG),
        recurrent_regularizer=keras.regularizers.l2(REG),
        bias_regularizer=keras.regularizers.l2(REG)
    )(encoder_embedding)
    encoder_states = [state_h, state_c]

    # Décodeur
    decoder_input = keras.Input(shape=(max_output_len,))
    decoder_embedding = layers.Embedding(total_words, embedding_dim)(decoder_input)
    decoder_embedding = layers.LayerNormalization()(decoder_embedding)
    decoder_lstm = layers.LSTM(
        lstm_units,
        return_sequences=True,
        return_state=True,
        #dropout=DROP_OUT,
        #recurrent_dropout=DROP_OUT,
        kernel_regularizer=keras.regularizers.l2(REG),
        recurrent_regularizer=keras.regularizers.l2(REG),
        bias_regularizer=keras.regularizers.l2(REG)
    )
    decoder_output, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
    decoder_dense = layers.TimeDistributed(layers.Dense(total_words, activation='softmax'))
    decoder_output = decoder_dense(decoder_output)

    model = keras.Model([encoder_input, decoder_input], decoder_output)

    model.compile(
        optimizer=keras.optimizers.legacy.Adam(hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='LOG', default=1e-3)),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

# Instancier le tuner et rechercher les meilleurs hyperparamètres
tuner = BayesianOptimization(
    build_model,
    objective='val_accuracy',
    max_trials=100,
    executions_per_trial=1,
    directory='my_dir',
    project_name='helloworld',
    num_initial_points=10
)

indices = np.arange(len(input_sequences))
np.random.shuffle(indices)

train_indices = indices[:int(0.8 * len(indices))]
val_indices = indices[int(0.8 * len(indices)):]

def data_generator(input_data, decoder_input, decoder_output, batch_size, indices):
    data_len = len(indices)
    num_batches = data_len // batch_size
    while True:
        for i in range(num_batches):
            start = i * batch_size
            end = (i + 1) * batch_size
            batch_indices = indices[start:end]
            yield [input_data[batch_indices], decoder_input[batch_indices]], decoder_output[batch_indices]

train_generator = data_generator(input_sequences, decoder_input_data, decoder_output_data, BATCH_SIZE, train_indices)
val_generator = data_generator(input_sequences, decoder_input_data, decoder_output_data, BATCH_SIZE, val_indices)

train_steps_per_epoch = len(train_indices) // BATCH_SIZE
val_steps_per_epoch = len(val_indices) // BATCH_SIZE

tuner.search(train_generator,
             steps_per_epoch=train_steps_per_epoch,
             validation_data=val_generator,
             validation_steps=val_steps_per_epoch,
             epochs=10)