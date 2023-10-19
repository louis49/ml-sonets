import os

from keras import layers, Input, Model, optimizers
from keras.src.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras_tuner import BayesianOptimization, HyperParameters, Objective
from tensorflow.python.data import Dataset
import tensorflow as tf

from twomodels.data import Data, MODEL_1_SEQ_BLACK_PATH, MODEL_1_SEQ_BLACK_TEST_PATH
from twomodels.epoch_callback import EpochCallback

BATCH_SIZE = 10
MODEL_1_PATH = "model/phon_model.h5"

class PhonModel():
    def __init__(self, data: Data):
        self.data = data

    def build_model(self, hp):
        embedding_dim = hp.Int("embedding_dim", min_value=8, max_value=512, step=8, default=184)
        lstm_units = hp.Int("lstm_units", min_value=8, max_value=512, step=8, default=168)
        learning_rate = hp.Float('learning_rate', min_value=1e-5, max_value=1e-2, sampling='log', default=0.001)

        # Entrées
        title_input = Input(shape=(self.data.title_max_size,), dtype="int32", name="title_input")  # Aucune longueur fixe, dépend des données
        decoder_input = Input(shape=(self.data.phon_max_size*14-1,), dtype="int32", name="decoder_input")  # Aucune longueur fixe

        # Embeddings
        title_embedding_layer = layers.Embedding(self.data.title_words, embedding_dim)  # supposant que 'title_words' est le nombre total de mots uniques dans les titres
        decoder_embedding_layer = layers.Embedding(self.data.phon_words, embedding_dim)  # supposant que 'phon_words' est le nombre total de phonèmes uniques

        title_embedded = title_embedding_layer(title_input)
        decoder_embedded = decoder_embedding_layer(decoder_input)

        # LSTM pour les entrées de titre
        title_lstm = layers.LSTM(lstm_units, return_sequences=True, return_state=True)
        title_outputs, title_state_h, title_state_c = title_lstm(title_embedded)  # Les états ici sont pour le contexte

        # Décodeur LSTM
        decoder_lstm = layers.LSTM(lstm_units, return_sequences=True)
        decoder_outputs = decoder_lstm(decoder_embedded, initial_state=[title_state_h, title_state_c])

        decoder_dense = layers.TimeDistributed(layers.Dense(self.data.phon_words+1, activation='softmax'))
        decoder_outputs = decoder_dense(decoder_outputs)

        # Création du modèle
        model = Model(inputs=[title_input, decoder_input], outputs=decoder_outputs)

        # Compilation du modèle
        model.compile(optimizer=optimizers.legacy.Adam(learning_rate=learning_rate),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        return model

    def train(self, use_tuner=False):
        def generator_wrapper_train():
            current_epoch = epoch_callback.current_epoch
            for data in self.data.generate_data(BATCH_SIZE, epoch_size_train, current_epoch, True):
                yield data

        def generator_wrapper_test():
            current_epoch = epoch_callback.current_epoch
            for data in self.data.generate_data(BATCH_SIZE, epoch_size_test, current_epoch, False):
                yield data

        epoch_callback = EpochCallback()

        epoch_size_train = self.data.count_tfrecord_samples(MODEL_1_SEQ_BLACK_PATH)
        epoch_size_test = self.data.count_tfrecord_samples(MODEL_1_SEQ_BLACK_TEST_PATH)

        #f = generator_wrapper_train()

        #k = self.data.generate_data(BATCH_SIZE, epoch_size_train, 1, True)

        train_data = Dataset.from_generator(
            generator_wrapper_train,
            output_signature=(
                (tf.TensorSpec(shape=(None, self.data.title_max_size), dtype=tf.int32),
                 tf.TensorSpec(shape=(None, self.data.phon_max_size*14-1), dtype=tf.int32),
                 ),
                tf.TensorSpec(shape=(None, self.data.phon_max_size*14-1, self.data.phon_words+1), dtype=tf.float32)
            )).repeat()

        test_data = Dataset.from_generator(
            generator_wrapper_test,
            output_signature=(
                (tf.TensorSpec(shape=(None, self.data.title_max_size), dtype=tf.int32),
                 tf.TensorSpec(shape=(None, self.data.phon_max_size*14-1), dtype=tf.int32),
                 ),
                tf.TensorSpec(shape=(None, self.data.phon_max_size*14-1, self.data.phon_words+1), dtype=tf.float32)
            )).repeat()

        best_hyperparameters = HyperParameters()
        best_hyperparameters.Fixed('embedding_dim', value=128)
        best_hyperparameters.Fixed('lstm_units', value=128)
        best_hyperparameters.Fixed('learning_rate', value=0.001)

        if use_tuner == True:
            tuner = BayesianOptimization(
                self.build_model,
                objective=Objective("val_accuracy", direction="max"),
                max_trials=100,
                executions_per_trial=1,
                directory='tuner',
                project_name='sonnets',
                num_initial_points=10
            )
            tuner.search(train_data,
                         validation_data=test_data,
                         epochs=20,
                         steps_per_epoch=epoch_size_train//BATCH_SIZE,
                         validation_steps=epoch_size_test//BATCH_SIZE
                         )
            best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0]

        model = self.build_model(best_hyperparameters)

        if os.path.exists(MODEL_1_PATH):
            model.load_weights(MODEL_1_PATH)

        save_options = tf.saved_model.SaveOptions()
        checkpoint = ModelCheckpoint(MODEL_1_PATH,
                                     save_freq='epoch',
                                     monitor='val_accuracy',
                                     save_best_only=True,
                                     mode="max",
                                     save_weights_only=False,
                                     # options=save_options,
                                     )
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, min_lr=0.00001)

        model.fit(train_data,
                  validation_data=test_data,
                  epochs=200,
                  steps_per_epoch = epoch_size_train//BATCH_SIZE,
                  validation_steps = epoch_size_test//BATCH_SIZE,
                  callbacks=[epoch_callback, checkpoint, reduce_lr])