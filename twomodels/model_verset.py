import os

from keras import layers, Input, Model, optimizers, regularizers
from keras.src.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.src.layers import Dense
from keras_tuner import BayesianOptimization, HyperParameters, Objective
from tensorflow.python.data import Dataset
import tensorflow as tf
import keras.backend as k

from data import Data, MODEL_2_SEQ_BLACK_PATH, MODEL_2_SEQ_BLACK_TEST_PATH, MODEL_2_SEQ_WHITE_PATH, MODEL_2_SEQ_WHITE_TEST_PATH
from generator_callback_verset import VersetGenerator
from tensorflow.python.keras.callbacks import Callback

BATCH_SIZE_TRAIN = 200
BATCH_SIZE_TEST = 100
MODEL_2_PATH = "model/verset_model.h5"

global CURRENT_EPOCH

class EpochCallback(Callback):
    def __init__(self):
        super().__init__()
        #print('EpochCallback Init : {}'.format(0))
        #self.current_epoch = 0

    def on_epoch_begin(self, epoch, logs=None):
        global CURRENT_EPOCH
        CURRENT_EPOCH = epoch

class VersetModel():
    def __init__(self, data: Data):
        self.data = data

    def build_model(self, hp):
        model = self.model(hp)

        return model

    def model(self,hp):

        def perplexity(y_true, y_pred):
            return k.exp(k.mean(k.categorical_crossentropy(y_true, y_pred)))

        def f1_score(y_true, y_pred):
            # Calculez la précision
            true_positives = k.sum(k.round(k.clip(y_true * y_pred, 0, 1)))
            predicted_positives = k.sum(k.round(k.clip(y_pred, 0, 1)))
            precision = true_positives / (predicted_positives + k.epsilon())

            # Calculez le rappel
            possible_positives = k.sum(k.round(k.clip(y_true, 0, 1)))
            recall = true_positives / (possible_positives + k.epsilon())

            # Calculez le score F1
            f1_val = 2 * (precision * recall) / (precision + recall + k.epsilon())

            return f1_val

        def find_last_word_index(y):
            # Convertir les prédictions de one-hot à des indices de classe
            y_indices = k.argmax(y, axis=-1)

            # Trouver l'index du dernier mot non nul dans chaque séquence
            mask = k.cast(k.not_equal(y_indices, 0), k.floatx())  # Masque pour identifier les mots non nuls
            last_word_indices = k.sum(mask, axis=1) - 1  # Index du dernier mot non nul
            last_word_indices = k.switch(last_word_indices < 0, k.zeros_like(last_word_indices), last_word_indices) # Pour traiter le cas d'aucun mot trouvé
            return last_word_indices

        def custom(y_true, y_pred):
            # Obtenir les indices du dernier mot significatif
            last_word_indices_true = find_last_word_index(y_true)

            # Extraire les indices de classe pour le dernier mot significatif de y_true
            y_true_last_word = tf.gather_nd(
                k.argmax(y_true, axis=-1),
                tf.stack([
                    tf.range(tf.shape(y_true)[0]),
                    tf.cast(last_word_indices_true, tf.int32)
                ], axis=1)
            )

            # Extraire les indices de classe pour le mot correspondant dans y_pred
            y_pred_last_word = tf.gather_nd(
                k.argmax(y_pred, axis=-1),
                tf.stack([
                    tf.range(tf.shape(y_pred)[0]),
                    tf.cast(last_word_indices_true, tf.int32)
                ], axis=1)
            )

            # Comparer les mots (étiquettes) générés et réels pour le dernier mot significatif
            correct_predictions = k.cast(k.equal(y_pred_last_word, y_true_last_word), k.floatx())

            # Calculer le score (par exemple, précision) pour le dernier mot significatif
            score = k.mean(correct_predictions)
            return score

        lstm_units = hp.Int("verset_lstm_units", min_value=8, max_value=512, step=8, default=128)
        embedding_dim_title = hp.Int("verset_encoder_title_embedding_dim", min_value=8, max_value=512, step=8, default=128)
        embedding_dim_decoder = hp.Int("verset_decoder_embedding_dim", min_value=8, max_value=512, step=8, default=128)
        learning_rate = hp.Float('learning_rate', min_value=1e-5, max_value=1e-2, sampling='log', default=0.001)

        attention_encoder = False#hp.Boolean("verset_attention_encoder", default=False)
        attention_decoder = False#hp.Boolean("verset_attention_decoder", default=False)

        num_heads_encoder = 8 #hp.Int("verset_num_heads_encoder", min_value=2, max_value=24, step=2, default=12, parent_name="verset_attention_encoder", parent_values=True)
        num_heads_decoder = 8 #hp.Int("verset_num_heads_decoder", min_value=2, max_value=24, step=2, default=12, parent_name="verset_attention_decoder", parent_values=True)

        lstm_layer_decoder = False #hp.Boolean("verset_lstm_layer_decoder", default=False)

        # Entrées
        encoder_input_title = Input(shape=(self.data.title_max_size,), dtype="int32", name="title_input")
        encoder_input_phons = Input(shape=(self.data.phon_max_size, self.data.phon_words + 1), dtype="float32", name="phons_input")
        encoder_input_line = Input(shape=(1,), dtype="int32", name="line_input")

        encoder_embedding_title = layers.Embedding(self.data.title_words, embedding_dim_title)(encoder_input_title)

        encoder_output_title, forward_h_title, forward_c_title, backward_h_title, backward_c_title = layers.Bidirectional(layers.LSTM(lstm_units, return_state=True, return_sequences=True))(encoder_embedding_title)
        state_h_title = layers.Concatenate()([forward_h_title, backward_h_title])
        state_c_title = layers.Concatenate()([forward_c_title, backward_c_title])

        encoder_output_phons, forward_h_phons, forward_c_phons, backward_h_phons, backward_c_phons = layers.Bidirectional(layers.LSTM(lstm_units, return_state=True, return_sequences=True))(encoder_input_phons)
        state_h_phons = layers.Concatenate()([forward_h_phons, backward_h_phons])
        state_c_phons = layers.Concatenate()([forward_c_phons, backward_c_phons])

        line_processing = Dense(units=2*lstm_units, activation="relu")(encoder_input_line)

        combined_state_h = layers.Concatenate()([state_h_title, state_h_phons, line_processing])
        combined_state_c = layers.Concatenate()([state_c_title, state_c_phons, line_processing])

        line_processing_expanded = tf.expand_dims(line_processing, axis=1)
        encoder_output = layers.Concatenate(axis=1)([encoder_output_title, encoder_output_phons, line_processing_expanded])

        if attention_encoder:
            encoder_attention = layers.MultiHeadAttention(num_heads=num_heads_encoder, key_dim=max(6*lstm_units//num_heads_encoder, 1))
            encoder_attention_output = encoder_attention(query=encoder_output, key=encoder_output, value=encoder_output)
            encoder_output = layers.Concatenate(axis=-1)([encoder_output, encoder_attention_output])

        decoder_input = Input(shape=(self.data.text_max_size - 1,), dtype="int32", name="decoder_input")
        decoder_embedding = layers.Embedding(self.data.text_words + 1, embedding_dim_decoder)(decoder_input)

        decoder_output = layers.LSTM(6 * lstm_units, return_sequences=True, return_state=False)(decoder_embedding, initial_state=[combined_state_h, combined_state_c])

        if attention_decoder:
            attention = layers.MultiHeadAttention(num_heads=num_heads_decoder, key_dim=max((6*lstm_units) // num_heads_decoder, 1))
            attention_output = attention(query=decoder_output, key=encoder_output, value=encoder_output)
            decoder_output = layers.Concatenate(axis=-1)([decoder_output, attention_output])
            decoder_output = layers.LayerNormalization()(decoder_output)

        if lstm_layer_decoder:
            decoder_output = layers.LSTM(6 * lstm_units,
                                         return_sequences=True,
                                         # kernel_regularizer = regularizers.l1_l2(l1=1e-5, l2=1e-4),
                                         # recurrent_regularizer = regularizers.l1_l2(l1=1e-5, l2=1e-4),
                                         # bias_regularizer = regularizers.l1_l2(l1=1e-5, l2=1e-4)
                                         )(decoder_output)

        decoder_dense = layers.TimeDistributed(layers.Dense(self.data.text_words + 1, activation='softmax'))
        decoder_output = decoder_dense(decoder_output)

        # Création du modèle
        model = Model(inputs=[encoder_input_title, encoder_input_phons, encoder_input_line, decoder_input], outputs=decoder_output, name="VersetModel")
        model.compile(optimizer=optimizers.legacy.Adam(learning_rate=learning_rate),
                      loss='categorical_crossentropy',
                      metrics=['accuracy', custom])

        return model

    def train(self, use_tuner=False):
        def generator_wrapper_train():
            #print("Run generator_wrapper_train - epoch : {}".format(CURRENT_EPOCH))
            for data in self.data.generate_data_verset(BATCH_SIZE_TRAIN, epoch_size_train // BATCH_SIZE_TRAIN, CURRENT_EPOCH, True, False):
                yield data

        def generator_wrapper_train_tuner():
            #print("Run generator_wrapper_train - epoch : {}".format(CURRENT_EPOCH))
            for data in self.data.generate_data_verset(BATCH_SIZE_TRAIN, epoch_size_train // BATCH_SIZE_TRAIN, CURRENT_EPOCH, True, True):
                yield data

        def generator_wrapper_test():
            for data in self.data.generate_data_verset(BATCH_SIZE_TEST, epoch_size_test // BATCH_SIZE_TEST, CURRENT_EPOCH, False, False):
                yield data

        def generator_wrapper_test_tuner():
            for data in self.data.generate_data_verset(BATCH_SIZE_TEST, epoch_size_test // BATCH_SIZE_TEST, CURRENT_EPOCH, False, False):
                yield data


        epoch_callback = EpochCallback()

        epoch_size_train = 2141447 // 100 #self.data.count_tfrecord_samples(MODEL_2_SEQ_BLACK_PATH) # #
        epoch_size_test = 4381 #self.data.count_tfrecord_samples(MODEL_2_SEQ_BLACK_TEST_PATH) # #

        train_data = Dataset.from_generator(
            generator_wrapper_train,
            output_signature=(
                (tf.TensorSpec(shape=(None, self.data.title_max_size), dtype=tf.int32),
                 tf.TensorSpec(shape=(None, self.data.phon_max_size, self.data.phon_words + 1), dtype=tf.float32),
                 tf.TensorSpec(shape=(None, 1), dtype=tf.int32),  #input_line
                 tf.TensorSpec(shape=(None, self.data.text_max_size - 1), dtype=tf.int32),
                 ),
                tf.TensorSpec(shape=(None, self.data.text_max_size - 1, self.data.text_words+1), dtype=tf.float32)
            )).repeat()

        train_data_tuner = Dataset.from_generator(
            generator_wrapper_train_tuner,
            output_signature=(
                (tf.TensorSpec(shape=(None, self.data.title_max_size), dtype=tf.int32),
                 tf.TensorSpec(shape=(None, self.data.phon_max_size, self.data.phon_words + 1), dtype=tf.float32),
                 tf.TensorSpec(shape=(None, 1), dtype=tf.int32),  # input_line
                 tf.TensorSpec(shape=(None, self.data.text_max_size - 1), dtype=tf.int32),
                 ),
                tf.TensorSpec(shape=(None, self.data.text_max_size - 1, self.data.text_words + 1), dtype=tf.float32)
            )).repeat()

        test_data = Dataset.from_generator(
            generator_wrapper_test,
            output_signature=(
                (tf.TensorSpec(shape=(None, self.data.title_max_size), dtype=tf.int32),
                 tf.TensorSpec(shape=(None, self.data.phon_max_size, self.data.phon_words + 1), dtype=tf.float32),
                 tf.TensorSpec(shape=(None, 1), dtype=tf.int32), #input_line
                 tf.TensorSpec(shape=(None, self.data.text_max_size - 1), dtype=tf.int32),
                 ),
                tf.TensorSpec(shape=(None, self.data.text_max_size - 1, self.data.text_words+1), dtype=tf.float32)
            )).repeat()

        test_data_tuner = Dataset.from_generator(
            generator_wrapper_test,
            output_signature=(
                (tf.TensorSpec(shape=(None, self.data.title_max_size), dtype=tf.int32),
                 tf.TensorSpec(shape=(None, self.data.phon_max_size, self.data.phon_words + 1), dtype=tf.float32),
                 tf.TensorSpec(shape=(None, 1), dtype=tf.int32),  # input_line
                 tf.TensorSpec(shape=(None, self.data.text_max_size - 1), dtype=tf.int32),
                 ),
                tf.TensorSpec(shape=(None, self.data.text_max_size - 1, self.data.text_words + 1), dtype=tf.float32)
            )).repeat()

        best_hyperparameters = HyperParameters()
        best_hyperparameters.Fixed('verset_lstm_units', value=32)
        best_hyperparameters.Fixed('verset_encoder_title_embedding_dim', value=8)
        best_hyperparameters.Fixed('verset_decoder_embedding_dim', value=32)

        best_hyperparameters.Fixed('verset_num_heads_encoder', value=8)  # 10
        best_hyperparameters.Fixed('verset_num_heads_decoder', value=8)  # 10
        best_hyperparameters.Fixed('verset_attention_encoder', value=False)
        best_hyperparameters.Fixed('verset_attention_decoder', value=False)
        best_hyperparameters.Fixed('verset_lstm_layer_decoder', value=False)

        best_hyperparameters.Fixed('learning_rate', value=0.001)
        #best_hyperparameters.Fixed('drop_out', value=0.38573)
        #best_hyperparameters.Fixed('regularizer', value=0.0001)


        if use_tuner:
            tuner = BayesianOptimization(
                self.build_model,
                objective=[Objective("val_accuracy", direction="max"),
                           Objective("val_custom", direction="max"),
                           ],
                max_trials=100,
                executions_per_trial=1,
                directory='tuner',
                project_name='versets',
                num_initial_points=10
            )
            tuner.search(train_data_tuner,
                         validation_data=test_data_tuner,
                         epochs=5,
                         steps_per_epoch=epoch_size_train//BATCH_SIZE_TRAIN,
                         validation_steps=epoch_size_test//BATCH_SIZE_TEST,
                         callbacks=[epoch_callback]
                         )
            best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0]

        model = self.build_model(best_hyperparameters)

        #if os.path.exists(MODEL_2_PATH):
        #    model.load_weights(MODEL_2_PATH)

        save_options = tf.saved_model.SaveOptions()
        checkpoint = ModelCheckpoint(MODEL_2_PATH,
                                     save_freq='epoch',
                                     monitor='val_accuracy',
                                     save_best_only=True,
                                     mode="max",
                                     save_weights_only=False,
                                     # options=save_options,
                                     )
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=10, min_lr=0.000001)

        verset_generator = VersetGenerator(model, data=self.data)

        model.fit(train_data,
                  validation_data=test_data,
                  epochs=400,
                  steps_per_epoch = epoch_size_train//BATCH_SIZE_TRAIN,
                  validation_steps = epoch_size_test//BATCH_SIZE_TEST,
                  callbacks=[epoch_callback, checkpoint, reduce_lr, verset_generator])