import json
import os
import tensorflow as tf
import numpy as np
from keras_tuner import BayesianOptimization

import xml2json
import json2text
import text2seq
import sample_generator
import model_class

JSON_PATH = "sonnets.json"
TEXT_PATH = "sonnets.txt"

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


def generate_data(dataset, total_words, batch_size):
    encoder_inputs = []
    decoder_inputs = []
    decoder_outputs = []

    total = 0
    call = 0

    for item in dataset:
        sequences = item['sequence'].numpy()
        for seq in sequences:
            encoder_input = seq
            decoder_input = np.zeros_like(seq)
            decoder_input[1:] = seq[:-1]
            decoder_output = tf.keras.utils.to_categorical(seq, num_classes=total_words)

            encoder_inputs.append(encoder_input)
            decoder_inputs.append(decoder_input)
            decoder_outputs.append(decoder_output)

            if len(encoder_inputs) == batch_size:
                total += batch_size
                call += 1
                #print("total : " + str(total) + " - call : " + str(call))
                #print("generating " + str(len(encoder_inputs)) + "samples - " + str(batch_size) + " batch_size")
                #print(encoder_input.shape, decoder_input.shape, decoder_output.shape)
                yield (tuple([tf.convert_to_tensor(np.array(encoder_inputs), dtype=tf.int32),
                              tf.convert_to_tensor(np.array(decoder_inputs), dtype=tf.int32)]),
                       tf.convert_to_tensor(np.array(decoder_outputs), dtype=tf.int32))
                encoder_inputs = []
                decoder_inputs = []
                decoder_outputs = []
    #print("Fin DATASET - " + "total : " + str(total) + " - call : " + str(call))

batch_size = 10

train_samples = text2seq.count_tfrecord_samples("data.tfrecord")#4820
val_samples = text2seq.count_tfrecord_samples("data_val.tfrecord")

train_steps_per_epoch = train_samples // batch_size
val_steps_per_epoch = val_samples // batch_size

#print("steps_per_epoch : " + str(train_steps_per_epoch) + " - batch_size : " + str(batch_size))
#print("val_steps_per_epoch : " + str(val_steps_per_epoch) + " - batch_size : " + str(batch_size))

train_dataset = text2seq.load_from_tfrecord("data.tfrecord", max_len, batch_size)
#k = generate_data(train_dataset, total_words, batch_size)

train_data = tf.data.Dataset.from_generator(lambda: generate_data(train_dataset, total_words, batch_size),
                                            output_signature=(
                                                (tf.TensorSpec(shape=(batch_size, None), dtype=tf.int32),
                                                 tf.TensorSpec(shape=(batch_size, None), dtype=tf.int32)),
                                                tf.TensorSpec(shape=(batch_size, None, total_words), dtype=tf.int32)
                                            )).repeat()
test_dataset = text2seq.load_from_tfrecord("data_val.tfrecord", max_len, batch_size)
test_data = tf.data.Dataset.from_generator(lambda: generate_data(test_dataset, total_words, batch_size),
                                            output_signature=(
                                                (tf.TensorSpec(shape=(batch_size, None), dtype=tf.int32),
                                                 tf.TensorSpec(shape=(batch_size, None), dtype=tf.int32)),
                                                tf.TensorSpec(shape=(batch_size, None, total_words), dtype=tf.int32)
                                            )).repeat()

input_text = "<sonnet> <title> Amour fou </title>  <stropheA>  <lineA>"
sample_generator = sample_generator.GenerateSample(input_text=input_text, tokenizer=tokenizer, total_words=total_words, max_decoder_seq_length=max_len, max_encoder_seq_length=max_len)

#model = sonnet_model.get_model(max_len, total_words)

sonnetModel = model_class.SonnetModel(max_len, max_len, total_words)

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
             epochs=1,
             steps_per_epoch=train_steps_per_epoch,
             validation_steps=val_steps_per_epoch
             )

best_model = tuner.get_best_models(num_models=1)[0]
tuner.results_summary()

#model.fit(train_data,
#          validation_data=test_data,
#          epochs=10,
#          steps_per_epoch=train_steps_per_epoch,
#          validation_steps=val_steps_per_epoch,
#          callbacks=[sample_generator])

print("")
