import xml.etree.ElementTree as ET
import lirecouleur.word

import numpy as np
import tensorflow as tf
from keras.src.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import layers
from tensorflow import keras
from tensorflow.keras.callbacks import Callback
from keras.callbacks import ReduceLROnPlateau
from keras.models import load_model

class DisplayOutput(Callback):
    def __init__(self, input_data, tokenizer, max_input_len, max_output_len, interval=1):
        super(DisplayOutput, self).__init__()
        self.input_data = input_data
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_output_len = max_output_len
        self.interval = interval

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.interval == 0:  # Afficher tous les `interval` epochs
            predicted_sonnet = self.generate_sonnet(self.model, self.input_data, self.tokenizer, self.max_input_len, self.max_output_len)
            print("\nEpoch {}: \n{}".format(epoch + 1, predicted_sonnet))

    def generate_sonnet(self, model, title, tokenizer, max_input_len, max_output_len):
        title_seq = tokenizer.texts_to_sequences([title])
        title_seq = pad_sequences(title_seq, maxlen=max_input_len)

        # Commencer la séquence de sortie avec le premier mot du titre
        curr_seq = [tokenizer.word_index['<start>']]

        output_sonnet = []

        for i in range(max_output_len):
            padded_seq = pad_sequences([curr_seq], maxlen=max_output_len, padding='post')
            predictions = model.predict([title_seq, padded_seq], verbose=0)

            # Obtenir le mot suivant (le mot le plus probable)
            next_word_idx = np.argmax(predictions[0],axis=-1)

            if next_word_idx[i] == 0:  # Si c'est un padding, on continue
                continue

            curr_seq.append(next_word_idx[i])

            # Convertir l'index du mot en mot
            next_word = tokenizer.index_word[next_word_idx[i]]
            output_sonnet.append(next_word)

            # Si le token 'end' est atteint, arrêter la prédiction
            if next_word == '<end>':
                break

        return ' '.join(output_sonnet)

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
sonnets_dicts = [sonnet_to_dict(sonnet) for sonnet in sonnets][:2500] #2000

# Prétraitement des données
titles = [sonnet["title"].replace(".", "") for sonnet in sonnets_dicts]

lengths = []
sonnets_blocks = []
for sonnet in sonnets_dicts:
    sonnets_block = " <start> "
    for strophe in sonnet["lines"]:
        sonnets_block += " <strophe> "
        for line in strophe:
            sonnets_block += " <ligne> "

            words = line.split()
            chars_to_replace = ['�', '■', 'µ', '[', ']', '(', ')', '\\', '/', '_', '!', ';', '-', '—', '–', ',', '.', '\'', '?', ':', '»', '«', '"', '>', '<', '•', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0', '    ', '   ', '  ']
            clean_line = line
            for char in chars_to_replace:
                clean_line = clean_line.replace(char, ' ')
            clean_line.rstrip()
            #Hermès
            #faïencier

            if(clean_line.split()[-1] == "faïencier"):
                rime = 's_c j_e_comp'
            else:
                phen3 = lirecouleur.word.phonemes(clean_line.split()[-1])
                filtered_phen3 = [x[0] for x in phen3 if x[0] != "#"]
                last_two = filtered_phen3[-2:]
                rime = ' '.join(last_two)
            #print(clean_line.split()[-1], ' '.join(last_two))

            if len(words) > 2:
                if(len(words[len(words)-1]) == 1):
                    words.insert(-2, "<rime> " + rime + " </rime>")
                else:
                    words.insert(-1, "<rime> " + rime + " </rime>")

            # Reconstruct the line with the tag
            line = ' '.join(words)

            sonnets_block += line.replace(".", " . ").replace(",", " , ").replace("!", " ! ").replace(":", " : ")
    sonnets_block += " <end> "
    sonnets_blocks.append(sonnets_block)
    lengths.append(len(sonnets_block))


# Tokenisation
#
tokenizer = Tokenizer(filters='"#$%&()*+-/:;=@[\]`{|}~�■µ\\–»«•1234567890') #»«
tokenizer.fit_on_texts(titles + sonnets_blocks)

assert '<start>' in tokenizer.word_index
assert '<end>' in tokenizer.word_index

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
BATCH_SIZE = 10
# Hyperparamètres initiaux et configuration du modèle
def build_model():
    # Encodeur

    DROP_OUT = 0.25
    REG = 0.0001
    lstm_units = 128
    embedding_dim = 128 #128

    encoder_input = keras.Input(shape=(max_input_len,))
    encoder_embedding = layers.Embedding(
        total_words,
        embedding_dim
    )(encoder_input)
    encoder_embedding = layers.LayerNormalization()(encoder_embedding)
    encoder_embedding = layers.Dropout(DROP_OUT)(encoder_embedding)
    encoder_output, forward_h, forward_c, backward_h, backward_c = layers.Bidirectional(layers.LSTM(
        lstm_units,
        return_state=True,
        return_sequences=True,
        #dropout=DROP_OUT,
        #recurrent_dropout=DROP_OUT,
        kernel_regularizer=keras.regularizers.l2(REG),
        recurrent_regularizer=keras.regularizers.l2(REG),
        bias_regularizer=keras.regularizers.l2(REG)
    ))(encoder_embedding)
    state_h = layers.Concatenate()([forward_h, backward_h])
    state_c = layers.Concatenate()([forward_c, backward_c])
    encoder_states = [state_h, state_c]

    encoder_attention = layers.MultiHeadAttention(num_heads=8, key_dim=embedding_dim)
    encoder_attention_output = encoder_attention(query=encoder_output, key=encoder_output, value=encoder_output)
    encoder_output = layers.Concatenate(axis=-1)([encoder_output, encoder_attention_output])

    # Décodeur
    decoder_input = keras.Input(shape=(max_output_len,))
    decoder_embedding = layers.Embedding(total_words, embedding_dim)(decoder_input)
    decoder_embedding = layers.LayerNormalization()(decoder_embedding)
    decoder_embedding = layers.Dropout(DROP_OUT)(decoder_embedding)
    decoder_output, _, _ = layers.LSTM(
        2*lstm_units,
        return_sequences=True,
        return_state=True,
        #dropout=DROP_OUT,
        #recurrent_dropout=DROP_OUT,
        kernel_regularizer=keras.regularizers.l2(REG),
        recurrent_regularizer=keras.regularizers.l2(REG),
        bias_regularizer=keras.regularizers.l2(REG)
    )(decoder_embedding, initial_state=encoder_states)

    attention = layers.MultiHeadAttention(num_heads=8, key_dim=embedding_dim)
    attention_output = attention(query=decoder_output, key=encoder_output, value=encoder_output)

    decoder_concat = layers.Concatenate(axis=-1)([decoder_output, attention_output])

    decoder_dense = layers.TimeDistributed(layers.Dense(total_words, activation='softmax'))
    decoder_output = decoder_dense(decoder_concat)

    model = keras.Model([encoder_input, decoder_input], decoder_output)

    model.compile(
        optimizer=keras.optimizers.legacy.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

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

LOAD = False
if LOAD == True :
    model = load_model('modele_epoch_05.h5')
else:
    model = build_model()
checkpoint = ModelCheckpoint('modele_epoch_{epoch:02d}.h5', save_freq=1000)
title_sample = "Amour fou"
display_callback = DisplayOutput(title_sample, tokenizer, max_input_len, max_output_len, interval=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)

model.fit(train_generator,
          steps_per_epoch=train_steps_per_epoch,
          validation_data=val_generator,
          validation_steps=val_steps_per_epoch,
          epochs=10000,
          callbacks=[checkpoint, display_callback, reduce_lr])