from keras import layers, Input, Model, optimizers
from keras.src.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras_tuner import BayesianOptimization, HyperParameters, Objective
from tensorflow.python.data import Dataset
import tensorflow as tf
import keras.backend as k

from data import Data, MODEL_1_SEQ_BLACK_PATH, MODEL_1_SEQ_BLACK_TEST_PATH
from generator_callback_phon import PhonGenerator
from tensorflow.python.keras.callbacks import Callback

BATCH_SIZE_TRAIN = 400
BATCH_SIZE_TEST = 100

MODEL_1_PATH = "model/phon_model.h5"

global CURRENT_EPOCH


# CURRENT_EPOCH = 0

class EpochCallback(Callback):
    def __init__(self):
        super().__init__()

    def on_epoch_begin(self, epoch, logs=None):
        global CURRENT_EPOCH
        CURRENT_EPOCH = epoch


class PhonModel:
    def __init__(self, data: Data):
        self.data = data

    def model(self, hp):
        model = self.build_model(hp)

        return model

    def build_model(self, hp):

        def perplexity(y_true, y_pred):
            return k.exp(k.mean(k.categorical_crossentropy(y_true, y_pred)))

        def find_last_word_index(y):
            # Convertir les prédictions de one-hot à des indices de classe
            y_indices = k.argmax(y, axis=-1)

            # Trouver l'index du dernier mot non nul dans chaque séquence
            mask = k.cast(k.not_equal(y_indices, 0), k.floatx())  # Masque pour identifier les mots non nuls
            last_word_indices = k.sum(mask, axis=1) - 1  # Index du dernier mot non nul
            last_word_indices = k.switch(last_word_indices < 0, k.zeros_like(last_word_indices), last_word_indices)  # Pour traiter le cas d'aucun mot trouvé
            return last_word_indices

        def custom(y_true, y_pred):
            # Obtenir les indices du dernier mot significatif
            last_word_indices_true = find_last_word_index(y_true)

            # Extraire les indices de classe pour le dernier mot significatif de y_true
            y_true_last_word = tf.gather_nd(k.argmax(y_true, axis=-1), tf.stack(
                [tf.range(tf.shape(y_true)[0]), tf.cast(last_word_indices_true, tf.int32)], axis=1))

            # Extraire les indices de classe pour le mot correspondant dans y_pred
            y_pred_last_word = tf.gather_nd(k.argmax(y_pred, axis=-1), tf.stack(
                [tf.range(tf.shape(y_pred)[0]), tf.cast(last_word_indices_true, tf.int32)], axis=1))

            # Comparer les mots (étiquettes) générés et réels pour le dernier mot significatif
            correct_predictions = k.cast(k.equal(y_pred_last_word, y_true_last_word), k.floatx())

            # Calculer le score (par exemple, précision) pour le dernier mot significatif
            score = k.mean(correct_predictions)
            return score

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

        '''
        Best val_accuracy So Far: 0.8253488540649414
        |Best Value So Far |Hyperparameter
        |504               |phon_lstm_units
        |112               |phon_encoder_embedding_dim
        |216               |phon_decoder_embedding_dim
        |0.01              |learning_rate
        |0.2               |phon_drop_out_encoder
        |0.6               |phon_drop_out_decoder
        |9.2337e-05        |phon_l1_regularizer
        |1.0802e-05        |phon_l2_regularizer
        |True              |phon_attention_encoder
        |False             |phon_attention_decoder
        |2                 |phon_num_heads_encoder
        |False             |lstm_layer_decoder
        '''
        lstm_units = hp.Int("phon_lstm_units", min_value=8, max_value=512, step=8, default=128)
        encoder_embedding_dim = hp.Int("phon_encoder_embedding_dim", min_value=8, max_value=512, step=8, default=8)
        decoder_embedding_dim = hp.Int("phon_decoder_embedding_dim", min_value=8, max_value=512, step=8, default=128)
        learning_rate = hp.Float('learning_rate', min_value=1e-5, max_value=1e-2, sampling='log', default=0.001)
        # drop_out_encoder = hp.Float('phon_drop_out_encoder', min_value=0.0, max_value=0.5, step=0.05, default=0.38)
        drop_out_decoder = hp.Float('phon_drop_out_decoder', min_value=0.0, max_value=0.5, step=0.05, default=0.38)
        # l1_regularizer = hp.Float('phon_l1_regularizer', min_value=1e-5, max_value=1e-2, sampling='log', default=0.0001)
        # l2_regularizer = hp.Float('phon_l2_regularizer', min_value=1e-5, max_value=1e-2, sampling='log', default=0.0001)

        attention_encoder = hp.Boolean("phon_attention_encoder", default=False)
        attention_decoder = hp.Boolean("phon_attention_decoder", default=False)
        num_heads_encoder = hp.Int("phon_num_heads_encoder", min_value=2, max_value=24, step=2, default=12,
                                   parent_name="phon_attention_encoder", parent_values=True)
        num_heads_decoder = hp.Int("phon_num_heads_decoder", min_value=2, max_value=24, step=2, default=12,
                                   parent_name="phon_attention_decoder", parent_values=True)

        lstm_layer_decoder = hp.Boolean("phon_lstm_layer_decoder", default=False)

        # Entrées
        encoder_input = Input(shape=(self.data.title_max_size,), dtype="int32", name="title_input")
        encoder_embedding = layers.Embedding(self.data.title_words, encoder_embedding_dim)(encoder_input)
        #encoder_embedding = layers.LayerNormalization()(encoder_embedding)
        # encoder_embedding = layers.Dropout(drop_out_encoder)(encoder_embedding)
        encoder_output, forward_h, forward_c, backward_h, backward_c = layers.Bidirectional(layers.LSTM(
            lstm_units,
            return_state=True,
            return_sequences=True,
            # dropout=drop_out_encoder,
            # recurrent_dropout=drop_out_encoder,
            # kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
            # recurrent_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
            # bias_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4)
        ))(encoder_embedding)
        state_h = layers.Concatenate()([forward_h, backward_h])
        state_c = layers.Concatenate()([forward_c, backward_c])
        encoder_states = [state_h, state_c]

        if attention_encoder:
            encoder_attention = layers.MultiHeadAttention(num_heads=num_heads_encoder,
                                                          key_dim=max(encoder_embedding_dim // num_heads_encoder, 1))
            encoder_attention_output = encoder_attention(query=encoder_output, key=encoder_output, value=encoder_output)
            encoder_output = layers.Concatenate(axis=-1)([encoder_output, encoder_attention_output])
            encoder_output = layers.LayerNormalization()(encoder_output)

        decoder_input = Input(shape=(self.data.phon_max_size * 14 + 1,), dtype="int32", name="decoder_input")
        decoder_embedding = layers.Embedding(self.data.phon_words, decoder_embedding_dim)(decoder_input)
        #decoder_embedding = layers.LayerNormalization()(decoder_embedding)
        decoder_embedding = layers.Dropout(drop_out_decoder)(decoder_embedding)

        decoder_output = layers.Bidirectional(layers.LSTM(
            2 * lstm_units,
            return_sequences=True,
            return_state=False,
        ))(decoder_embedding, initial_state=[forward_h, forward_c, backward_h, backward_c])

        if attention_decoder:
            attention = layers.MultiHeadAttention(num_heads=num_heads_decoder,
                                                  key_dim=max(decoder_embedding_dim // num_heads_decoder, 1))
            attention_output = attention(query=decoder_output, key=encoder_output, value=encoder_output)
            decoder_output = layers.Concatenate(axis=-1)([decoder_output, attention_output])
            #decoder_output = layers.LayerNormalization()(decoder_output)

        if lstm_layer_decoder:
            decoder_output = layers.LSTM(2 * lstm_units,
                                         return_sequences=True,
                                         )(decoder_output)

        decoder_dense = layers.TimeDistributed(layers.Dense(self.data.phon_words + 1, activation='softmax'))
        decoder_output = decoder_dense(decoder_output)

        # Création du modèle
        model = Model(inputs=[encoder_input, decoder_input], outputs=decoder_output, name="PhonModel")

        model.compile(optimizer=optimizers.legacy.Adam(learning_rate=learning_rate),
                      loss='categorical_crossentropy',
                      metrics=['accuracy', custom])  # 'Precision', 'Recall',perplexity, f1_score, perplexity

        return model

    def train(self, use_tuner=False):
        def generator_wrapper_train():
            # print("Run generator_wrapper_train - epoch : {}".format(CURRENT_EPOCH))
            for data in self.data.generate_data_phon(BATCH_SIZE_TRAIN, epoch_size_train // BATCH_SIZE_TRAIN,
                                                     CURRENT_EPOCH, True):
                yield data

        def generator_wrapper_test():
            for data in self.data.generate_data_phon(BATCH_SIZE_TEST, epoch_size_test // BATCH_SIZE_TEST, CURRENT_EPOCH,
                                                     False):
                yield data

        epoch_callback = EpochCallback()

        epoch_size_train = 264301 // 10#self.data.count_tfrecord_samples(MODEL_1_SEQ_BLACK_PATH)
        epoch_size_test = 536 #self.data.count_tfrecord_samples(MODEL_1_SEQ_BLACK_TEST_PATH)

        train_data = Dataset.from_generator(
            generator_wrapper_train,
            output_signature=(
                (tf.TensorSpec(shape=(None, self.data.title_max_size), dtype=tf.int32),
                 tf.TensorSpec(shape=(None, self.data.phon_max_size * 14 + 1), dtype=tf.int32),
                 ),
                tf.TensorSpec(shape=(None, self.data.phon_max_size * 14 + 1, self.data.phon_words + 1),
                              dtype=tf.float32)
            )).repeat()

        test_data = Dataset.from_generator(
            generator_wrapper_test,
            output_signature=(
                (tf.TensorSpec(shape=(None, self.data.title_max_size), dtype=tf.int32),
                 tf.TensorSpec(shape=(None, self.data.phon_max_size * 14 + 1), dtype=tf.int32),
                 ),
                tf.TensorSpec(shape=(None, self.data.phon_max_size * 14 + 1, self.data.phon_words + 1),
                              dtype=tf.float32)
            )).repeat()

        best_hyperparameters = HyperParameters()
        best_hyperparameters.Fixed('phon_lstm_units', value=128)  # 168 #128
        best_hyperparameters.Fixed('phon_encoder_embedding_dim', value=8)  # 184
        best_hyperparameters.Fixed('phon_decoder_embedding_dim', value=128)  # 184 #192
        best_hyperparameters.Fixed('learning_rate', value=0.005)  # 0.001
        # best_hyperparameters.Fixed('phon_drop_out_encoder', value=0.2)  # 0.38573
        best_hyperparameters.Fixed('phon_drop_out_decoder', value=0.33)
        best_hyperparameters.Fixed('phon_num_heads_encoder', value=8)  # 10
        best_hyperparameters.Fixed('phon_num_heads_decoder', value=8)  # 10
        best_hyperparameters.Fixed('phon_attention_encoder', value=False)
        best_hyperparameters.Fixed('phon_attention_decoder', value=True)
        best_hyperparameters.Fixed('phon_lstm_layer_decoder', value=True)

        if use_tuner:
            tuner = BayesianOptimization(
                self.model,
                objective=[Objective("val_accuracy", direction="max"),
                           Objective("val_custom", direction="max"),
                           ],
                max_trials=1000,
                executions_per_trial=1,
                directory='tuner',
                project_name='phons',
                num_initial_points=16,
                # hyperparameters=best_hyperparameters
            )
            tuner.search(train_data,
                         validation_data=test_data,
                         epochs=2,
                         steps_per_epoch=epoch_size_train // BATCH_SIZE_TRAIN,
                         validation_steps=epoch_size_test // BATCH_SIZE_TEST,
                         callbacks=[epoch_callback]
                         )
            best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0]

        model = self.model(best_hyperparameters)

        # if os.path.exists(MODEL_1_PATH):
        #    model.load_weights(MODEL_1_PATH)

        save_options = tf.saved_model.SaveOptions()
        checkpoint = ModelCheckpoint(MODEL_1_PATH,
                                     save_freq='epoch',
                                     monitor='val_accuracy',
                                     save_best_only=True,
                                     mode="max",
                                     save_weights_only=False,
                                     # options=save_options,
                                     )
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.90, patience=10, min_lr=0.000001)

        phon_generator = PhonGenerator(model, data=self.data)

        model.fit(train_data,
                  validation_data=test_data,
                  epochs=400,
                  steps_per_epoch=epoch_size_train // BATCH_SIZE_TRAIN,
                  validation_steps=epoch_size_test // BATCH_SIZE_TEST,
                  callbacks=[epoch_callback, checkpoint, reduce_lr, phon_generator])  # #checkpoint reduce_lr
