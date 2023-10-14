import json
import os
import tensorflow as tf
import numpy as np
from keras_tuner import BayesianOptimization, HyperParameters
from tensorflow.python.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.models import load_model
from tensorflow.saved_model import SaveOptions

import xml2json
import json2text
import text2seq
import sample_generator
import model_class

JSON_PATH = "sonnets.json"
TEXT_PATH = "sonnets.txt"
MODEL_PATH = "model.h5"
USE_TUNER = False
INPUT_TEXT = "<sonnet> <title> Pâques Fleuries </title>"
BATCH_SIZE = 10

dictionary = {}
if os.path.exists(JSON_PATH):
    with open(JSON_PATH, "r") as fichier:
        dictionary = json.load(fichier)
else:
    dictionary = xml2json.xml_to_dic()
    with open(JSON_PATH, "w", encoding='utf-8') as fichier:
        json.dump(dictionary, fichier, indent=4, ensure_ascii=False)

sonnets = []
if os.path.exists(TEXT_PATH):
    with open(TEXT_PATH, "r") as fichier:
        sonnets = json.load(fichier)
else:
    sonnets = json2text.dic2text(dictionary)
    with open(TEXT_PATH, "w", encoding='utf-8') as fichier:
        json.dump(sonnets, fichier, indent=4, ensure_ascii=False)

text2seq.text2seq(sonnets)

tokenizer = text2seq.load_tokenizer("tokenizer.json")
max_len = text2seq.load_max_len("max_len.json")
total_words = len(tokenizer.word_index) + 1


def generate_data(dataset, tokenizer, total_words, max_len, batch_size):
    encoder_inputs = []
    decoder_inputs = []
    decoder_outputs = []

    end_title_id = tokenizer.word_index['</title>']

    for item in dataset:
        sequences = item['sequence'].numpy()
        for full_sequence in sequences:
            title_end_idx = np.where(full_sequence == end_title_id)[0][0]
            title_sequence = full_sequence[:title_end_idx + 1]  # Include the </title> token
            sonnet_sequence = full_sequence[title_end_idx + 1:]  # Start after the </title> token

            title_sequence = np.pad(title_sequence, (0, max_len - len(title_sequence)))
            encoder_inputs.append(title_sequence)

            decoder_input = np.zeros_like(sonnet_sequence)
            decoder_input[1:] = sonnet_sequence[:-1]
            decoder_input = np.pad(decoder_input, (0, max_len - len(decoder_input)))
            decoder_inputs.append(decoder_input)

            sonnet_sequence = np.pad(sonnet_sequence, (0, max_len - len(sonnet_sequence)))
            decoder_output = tf.keras.utils.to_categorical(sonnet_sequence, num_classes=total_words)
            decoder_outputs.append(decoder_output)

            if len(encoder_inputs) == batch_size:
                yield (tuple([tf.convert_to_tensor(np.array(encoder_inputs), dtype=tf.int32),
                              tf.convert_to_tensor(np.array(decoder_inputs), dtype=tf.int32)]),
                       tf.convert_to_tensor(np.array(decoder_outputs), dtype=tf.int32))
                encoder_inputs = []
                decoder_inputs = []
                decoder_outputs = []


train_samples = text2seq.count_tfrecord_samples("data.tfrecord")
val_samples = text2seq.count_tfrecord_samples("data_val.tfrecord")

train_steps_per_epoch = train_samples // BATCH_SIZE
val_steps_per_epoch = val_samples // BATCH_SIZE

train_dataset = text2seq.load_from_tfrecord("data.tfrecord", max_len, BATCH_SIZE)
#k = generate_data(train_dataset, tokenizer, total_words, max_len, BATCH_SIZE)
train_data = tf.data.Dataset.from_generator(lambda: generate_data(train_dataset, tokenizer, total_words, max_len, BATCH_SIZE),
                                            output_signature=(
                                                (tf.TensorSpec(shape=(BATCH_SIZE, None), dtype=tf.int32),
                                                 tf.TensorSpec(shape=(BATCH_SIZE, None), dtype=tf.int32)),
                                                tf.TensorSpec(shape=(BATCH_SIZE, None, total_words), dtype=tf.int32)
                                            )).repeat()
test_dataset = text2seq.load_from_tfrecord("data_val.tfrecord", max_len, BATCH_SIZE)
test_data = tf.data.Dataset.from_generator(lambda: generate_data(test_dataset, tokenizer,total_words, max_len, BATCH_SIZE),
                                           output_signature=(
                                               (tf.TensorSpec(shape=(BATCH_SIZE, None), dtype=tf.int32),
                                                tf.TensorSpec(shape=(BATCH_SIZE, None), dtype=tf.int32)),
                                               tf.TensorSpec(shape=(BATCH_SIZE, None, total_words), dtype=tf.int32)
                                           )).repeat()



sonnetModel = model_class.SonnetModel(max_len, max_len, total_words)

best_hyperparameters = HyperParameters()
best_hyperparameters.Fixed('lstm_units', value=168)#168
best_hyperparameters.Fixed('embedding_dim', value=184)#184
best_hyperparameters.Fixed('learning_rate', value=0.001)#0.001
best_hyperparameters.Fixed('drop_out', value=0.38573)#0.38573
best_hyperparameters.Fixed('regularizer', value=0.0001)#0.0001
best_hyperparameters.Fixed('num_heads', value=4)#10

if USE_TUNER:
    tuner = BayesianOptimization(
        sonnetModel.build_model,
        objective='val_accuracy',
        max_trials=100,
        executions_per_trial=1,
        directory='tuner',
        project_name='sonnets',
        num_initial_points=10
    )
    tuner.search(train_data,
                 validation_data=test_data,
                 epochs=10,
                 steps_per_epoch=train_steps_per_epoch,
                 validation_steps=val_steps_per_epoch
                 )
    best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0]

model = sonnetModel.build_model(best_hyperparameters)
if os.path.exists(MODEL_PATH):
    model.load_weights(MODEL_PATH)

sample_generator = sample_generator.GenerateSample(input_text=INPUT_TEXT, tokenizer=tokenizer, total_words=total_words,
                                                   max_decoder_seq_length=max_len, max_encoder_seq_length=max_len, every_epoch=1)
save_options = tf.saved_model.SaveOptions()
checkpoint = ModelCheckpoint(MODEL_PATH,
                             save_freq='epoch',
                             monitor='val_accuracy',
                             save_best_only=True,
                             mode="max",
                             save_weights_only=False,
                             #options=save_options,
                             )
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, min_lr=0.00001)

model.fit(train_data,
          validation_data=test_data,
          epochs=100,
          steps_per_epoch=train_steps_per_epoch,
          validation_steps=val_steps_per_epoch,
          callbacks=[checkpoint, reduce_lr, sample_generator])
