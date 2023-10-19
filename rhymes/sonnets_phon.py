from keras_tuner import BayesianOptimization, HyperParameters
import tensorflow as tf
from tensorflow.python.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
import os

import dataset_generator
import sonnets_phon_model
import verset_generator

MODEL_PATH = "model/sonnets_phon_model.h5"
BATCH_SIZE = 100

def generate_data(dataset, total_words):
    for batch in dataset:
        input_title = batch['title']
        input_phon = batch['phon']
        target_text = batch['text']

        # Pour decoder_input, nous voulons toutes les séquences jusqu'à, mais sans inclure, '<end>'
        # Nous supposons ici que '<end>' est toujours le dernier token et qu'il n'est pas nécessaire dans decoder_input
        decoder_input = target_text[:, :-1]  # Tous les tokens sauf '<end>'

        # decoder_output sera utilisé pour la comparaison pendant l'entraînement, il commence donc par le premier mot réel et se termine par '<end>'
        decoder_output = target_text[:, 1:]  # Tous les tokens sauf '<start>'

        # Convertir decoder_output en encodage one-hot pour la formation
        decoder_output_onehot = tf.keras.utils.to_categorical(decoder_output, num_classes=total_words+1)

        # Votre générateur doit renvoyer un tuple de ((input1, input2, decoder_input), decoder_output_onehot)
        return tuple([input_title, input_phon, decoder_input]), decoder_output_onehot

def learn(use_tuner=False):
    train_samples = dataset_generator.count_tfrecord_samples("data/data_sonnet.tfrecord")
    val_samples = dataset_generator.count_tfrecord_samples("data/data_val_sonnet.tfrecord")

    train_steps_per_epoch = train_samples // BATCH_SIZE
    val_steps_per_epoch = val_samples // BATCH_SIZE

    max_len = dataset_generator.load_max_len("data/verset_max_len.json")

    max_len_title = max_len['title_max_len']
    max_len_text = max_len['text_max_len']
    max_len_phon = max_len['phon_max_len']

    title_words = max_len['title_words']
    text_words = max_len['text_words']
    phon_words = max_len['phon_words']

    phon_tokenizer= dataset_generator.load_tokenizer("data/phon_tokenizer.json")
    text_tokenizer= dataset_generator.load_tokenizer("data/text_tokenizer.json")
    title_tokenizer = dataset_generator.load_tokenizer("data/title_tokenizer.json")

    v_model = sonnets_phon_model.SonnetsPhonModel(
        title_words=title_words,
        text_words=text_words,
        phon_words=phon_words,
        max_len_title=max_len_title,
        max_len_text=max_len_text,
        max_len_phon=max_len_phon)

    train_dataset = dataset_generator.load_sonnets_from_tfrecord("data/data_sonnet.tfrecord", BATCH_SIZE, max_len_title, max_len_text, max_len_phon)
    test_dataset = dataset_generator.load_sonnets_from_tfrecord("data/data_val_sonnet.tfrecord", BATCH_SIZE, max_len_title, max_len_text, max_len_phon)
    k = generate_data(train_dataset, text_words)
    train_data = tf.data.Dataset.from_generator(
        lambda: generate_data(train_dataset, text_words),
        output_signature=(
            (tf.TensorSpec(shape=(None, max_len_title), dtype=tf.int32),
             tf.TensorSpec(shape=(None, max_len_phon), dtype=tf.int32),
             tf.TensorSpec(shape=(None, max_len_text-1), dtype=tf.int32)
             ),
            tf.TensorSpec(shape=(None, max_len_text-1, text_words+1), dtype=tf.int32)
        )).repeat()

    test_data = tf.data.Dataset.from_generator(
        lambda: generate_data(test_dataset, text_words),
        output_signature=(
            (tf.TensorSpec(shape=(None, max_len_title), dtype=tf.int32),
             tf.TensorSpec(shape=(None, max_len_phon), dtype=tf.int32),
             tf.TensorSpec(shape=(None, max_len_text-1), dtype=tf.int32)
             ),
            tf.TensorSpec(shape=(None, max_len_text-1, text_words+1), dtype=tf.int32)
        )).repeat()

    best_hyperparameters = HyperParameters()
    best_hyperparameters.Fixed('embedding_dim', value=128)
    best_hyperparameters.Fixed('lstm_units', value=128)
    best_hyperparameters.Fixed('learning_rate', value=0.001)

    if use_tuner == True:
        tuner = BayesianOptimization(
            v_model.build_model,
            objective='val_accuracy',
            max_trials=100,
            executions_per_trial=1,
            directory='tuner',
            project_name='sonnets',
            num_initial_points=10
        )
        tuner.search(train_data,
                     validation_data=test_data,
                     epochs=5,
                     steps_per_epoch=train_steps_per_epoch,
                     validation_steps=val_steps_per_epoch
                     )
        best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0]

    model = v_model.build_model(best_hyperparameters)

    if os.path.exists(MODEL_PATH):
        model.load_weights(MODEL_PATH)

    save_options = tf.saved_model.SaveOptions()
    checkpoint = ModelCheckpoint(MODEL_PATH,
                                 save_freq='epoch',
                                 monitor='val_accuracy',
                                 save_best_only=True,
                                 mode="max",
                                 save_weights_only=False,
                                 # options=save_options,
                                 )
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, min_lr=0.00001)
    v_generator = verset_generator.VersetGenerator(title="SONNET POUR LA MORT DE SON AMIE",
                                                   phon="ʁ a s",
                                                   title_tokenizer=title_tokenizer,
                                                   text_tokenizer=text_tokenizer,
                                                   phon_tokenizer=phon_tokenizer,
                                                   title_words = title_words,
                                                   max_len_text=max_len_text,
                                                   max_len_phon=max_len_phon,
                                                   max_len_title=max_len_title)

    model.fit(train_data,
              validation_data=test_data,
              epochs=100,
              steps_per_epoch=train_steps_per_epoch,
              validation_steps=val_steps_per_epoch,
              callbacks=[checkpoint, reduce_lr,v_generator])