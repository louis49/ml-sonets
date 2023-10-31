import os

from keras import layers, Input, Model, optimizers, regularizers
from keras.src.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.src.layers import Dense
from keras_tuner import BayesianOptimization, HyperParameters, Objective
from tensorflow.python.data import Dataset
import tensorflow as tf

from data import Data, MODEL_2_SEQ_BLACK_PATH, MODEL_2_SEQ_BLACK_TEST_PATH
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
        model = self.model_simple(hp)

        return model

    def model_simple(self,hp):
        lstm_units = hp.Int("verset_lstm_units", min_value=128, max_value=512, step=8, default=512)
        embedding_dim_title = hp.Int("verset_encoder_title_embedding_dim", min_value=128, max_value=512, step=8, default=512)
        embedding_dim_decoder = hp.Int("verset_decoder_embedding_dim", min_value=128, max_value=512, step=8, default=512)
        learning_rate = hp.Float('learning_rate', min_value=1e-5, max_value=1e-2, sampling='log', default=0.001)

        # Entrées
        encoder_input_title = Input(shape=(self.data.title_max_size,), dtype="int32", name="title_input")
        encoder_input_phons = Input(shape=(self.data.phon_max_size, self.data.phon_words + 1), dtype="float32", name="phons_input")
        encoder_input_line = Input(shape=(1,), dtype="int32", name="line_input")

        encoder_embedding_title = layers.Embedding(self.data.title_words, embedding_dim_title)(encoder_input_title)

        encoder_output_title, forward_h_title, forward_c_title, backward_h_title, backward_c_title = layers.Bidirectional(layers.LSTM(lstm_units, return_state=True))(encoder_embedding_title)
        state_h_title = layers.Concatenate()([forward_h_title, backward_h_title])
        state_c_title = layers.Concatenate()([forward_c_title, backward_c_title])

        encoder_output_phons, forward_h_phons, forward_c_phons, backward_h_phons, backward_c_phons = layers.Bidirectional(layers.LSTM(lstm_units, return_state=True))(encoder_input_phons)
        state_h_phons = layers.Concatenate()([forward_h_phons, backward_h_phons])
        state_c_phons = layers.Concatenate()([forward_c_phons, backward_c_phons])

        line_processing = Dense(units=lstm_units, activation="relu")(encoder_input_line)

        combined_state_h = layers.Concatenate()([state_h_title, state_h_phons, line_processing])
        combined_state_c = layers.Concatenate()([state_c_title, state_c_phons, line_processing])

        decoder_input = Input(shape=(self.data.text_max_size - 1,), dtype="int32", name="decoder_input")
        decoder_embedding = layers.Embedding(self.data.text_words + 1, embedding_dim_decoder)(decoder_input)

        decoder_output = layers.LSTM(5 * lstm_units, return_sequences=True, return_state=False)(decoder_embedding, initial_state=[combined_state_h, combined_state_c])

        decoder_dense = layers.TimeDistributed(layers.Dense(self.data.text_words + 1, activation='softmax'))
        decoder_outputs = decoder_dense(decoder_output)

        # Création du modèle
        model = Model(inputs=[encoder_input_title, encoder_input_phons, encoder_input_line, decoder_input], outputs=decoder_outputs, name="VersetModel")
        model.compile(optimizer=optimizers.legacy.Adam(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])

        return model

    def model(self, hp):
        lstm_units = hp.Int("verset_lstm_units", min_value=128, max_value=512, step=8, default=512)
        encoder_title_embedding_dim = hp.Int("verset_encoder_title_embedding_dim", min_value=128, max_value=512, step=8, default=512)
        #encoder_line_embedding_dim = hp.Int("verset_encoder_line_embedding_dim", min_value=128, max_value=2048, step=128, default=1024)
        decoder_embedding_dim = hp.Int("verset_decoder_embedding_dim", min_value=128, max_value=512, step=8, default=512)
        #drop_out = hp.Float('drop_out', min_value=0.0, max_value=1.0, step=0.05, default=0.38)
        #regularizer = hp.Float('regularizer', min_value=1e-5, max_value=1e-2, sampling='log', default=0.0001)

        attention = hp.Boolean("verset_attention", default=False)
        num_heads_encoder_title = hp.Int("verset_num_heads_encoder_title", min_value=8, max_value=24, step=4, default=12, parent_name="verset_attention", parent_values=True)
        num_heads_encoder_phons = hp.Int("verset_num_heads_encoder_phons", min_value=8, max_value=24, step=4, default=12, parent_name="verset_attention", parent_values=True)
        num_heads_decoder = hp.Int("verset_num_heads_decoder", min_value=8, max_value=24, step=4, default=12, parent_name="verset_attention", parent_values=True)

        learning_rate = hp.Float('learning_rate', min_value=1e-5, max_value=1e-2, sampling='log', default=0.001)

        # Entrées
        encoder_input_title = Input(shape=(self.data.title_max_size,), dtype="int32", name="title_input")
        encoder_input_phons = Input(shape=(self.data.phon_max_size, self.data.phon_words + 1), dtype="float32", name="phons_input")
        encoder_input_line = Input(shape=(1,), dtype="int32", name="line_input")

        encoder_embedding_title = layers.Embedding(self.data.title_words, encoder_title_embedding_dim)(encoder_input_title)
        encoder_embedding_phons = encoder_input_phons #layers.Embedding(self.data.phon_words, encoder_phons_embedding_dim)(encoder_input_phons)

        line_processing = Dense(units=2*lstm_units, activation="relu")(
            encoder_input_line)

        #encoder_embedding_title = layers.LayerNormalization()(encoder_embedding_title)
        #encoder_embedding_title = layers.Dropout(drop_out)(encoder_embedding_title)

        #encoder_embedding_phons = layers.LayerNormalization()(encoder_embedding_phons)
        #encoder_embedding_phons = layers.Dropout(drop_out)(encoder_embedding_phons)

        encoder_output_title, forward_h_title, forward_c_title, backward_h_title, backward_c_title = layers.Bidirectional(layers.LSTM(
            lstm_units,
            return_state=True,
            return_sequences=True,
            # dropout=DROP_OUT,
            # recurrent_dropout=DROP_OUT,
            #kernel_regularizer=regularizers.l1_l2(regularizer),
            #recurrent_regularizer=regularizers.l1_l2(regularizer),
            #bias_regularizer=regularizers.l1_l2(regularizer)
        ))(encoder_embedding_title)
        state_h_title = layers.Concatenate()([forward_h_title, backward_h_title])
        state_c_title = layers.Concatenate()([forward_c_title, backward_c_title])

        encoder_output_phons, forward_h_phons, forward_c_phons, backward_h_phons, backward_c_phons = layers.Bidirectional(
            layers.LSTM(
                lstm_units,
                return_state=True,
                return_sequences=True,
                # dropout=DROP_OUT,
                # recurrent_dropout=DROP_OUT,
                #kernel_regularizer=regularizers.l1_l2(regularizer),
                #recurrent_regularizer=regularizers.l1_l2(regularizer),
                #bias_regularizer=regularizers.l1_l2(regularizer)
            ))(encoder_embedding_phons)
        state_h_phons = layers.Concatenate()([forward_h_phons, backward_h_phons])
        state_c_phons = layers.Concatenate()([forward_c_phons, backward_c_phons])

        combined_state_h = layers.Concatenate()([state_h_title, state_h_phons, line_processing])
        combined_state_c = layers.Concatenate()([state_c_title, state_c_phons, line_processing])

        combined_encoder_output = layers.Concatenate(axis=1)([encoder_output_title, encoder_output_phons])

        # key_dim multiple of num_heads ? hp value "ratio" ?
        #if attention:
        #    encoder_attention_title = layers.MultiHeadAttention(num_heads=num_heads_encoder_title, key_dim=encoder_title_embedding_dim//num_heads_encoder_title)
        #    encoder_attention_output_title = encoder_attention_title(query=encoder_output_title, key=encoder_output_title, value=encoder_output_title)
        #    encoder_output_title = layers.Concatenate(axis=-1)([encoder_output_title, encoder_attention_output_title])

        #    encoder_attention_phons = layers.MultiHeadAttention(num_heads=num_heads_encoder_phons, key_dim=self.data.phon_max_size//num_heads_encoder_phons)
        #    encoder_attention_output_phons = encoder_attention_phons(query=encoder_output_phons, key=encoder_output_phons, value=encoder_output_phons)
        #    encoder_output_phons = layers.Concatenate(axis=-1)([encoder_output_phons, encoder_attention_output_phons])



        decoder_input = Input(shape=(self.data.text_max_size-1,), dtype="int32", name="decoder_input")
        decoder_embedding = layers.Embedding(self.data.text_words+1, decoder_embedding_dim)(decoder_input)
        #decoder_embedding = layers.LayerNormalization()(decoder_embedding)
        #decoder_embedding = layers.Dropout(drop_out)(decoder_embedding)
        decoder_output, _, _ = layers.LSTM(
            6 * lstm_units,
            return_sequences=True,
            return_state=True,
            # dropout=DROP_OUT,
            # recurrent_dropout=DROP_OUT,
            #kernel_regularizer=regularizers.l1_l2(regularizer),
            #recurrent_regularizer=regularizers.l1_l2(regularizer),
            #bias_regularizer=regularizers.l1_l2(regularizer)
        )(decoder_embedding, initial_state=[combined_state_h, combined_state_c])

        # key_dim multiple of num_heads ? hp value "ratio" ?
        if attention:
            attention = layers.MultiHeadAttention(num_heads=num_heads_decoder, key_dim=decoder_embedding_dim//num_heads_decoder)
            attention_output = attention(query=decoder_output, key=combined_encoder_output, value=combined_encoder_output)
            decoder_output = layers.Concatenate(axis=-1)([decoder_output, attention_output])

        decoder_dense = layers.TimeDistributed(layers.Dense(self.data.text_words+1, activation='softmax'))
        decoder_outputs = decoder_dense(decoder_output)

        # Création du modèle
        model = Model(inputs=[encoder_input_title, encoder_input_phons, encoder_input_line, decoder_input], outputs=decoder_outputs, name="VersetModel")

        model.compile(optimizer=optimizers.legacy.Adam(learning_rate=learning_rate),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        return model

    def train(self, use_tuner=False):
        def generator_wrapper_train():
            #print("Run generator_wrapper_train - epoch : {}".format(CURRENT_EPOCH))
            for data in self.data.generate_data_verset(BATCH_SIZE_TRAIN, epoch_size_train // BATCH_SIZE_TRAIN, CURRENT_EPOCH, True):
                yield data

        def generator_wrapper_test():
            for data in self.data.generate_data_verset(BATCH_SIZE_TEST, epoch_size_test // BATCH_SIZE_TEST, CURRENT_EPOCH, False):
                yield data



        epoch_callback = EpochCallback()

        epoch_size_train = self.data.count_tfrecord_samples(MODEL_2_SEQ_BLACK_PATH)
        epoch_size_test = min(self.data.count_tfrecord_samples(MODEL_2_SEQ_BLACK_TEST_PATH), BATCH_SIZE_TEST)


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

        best_hyperparameters = HyperParameters()
        best_hyperparameters.Fixed('verset_lstm_units', value=128)
        best_hyperparameters.Fixed('verset_encoder_title_embedding_dim', value=128)
        best_hyperparameters.Fixed('verset_decoder_embedding_dim', value=128)
        #best_hyperparameters.Fixed('verset_attention', value=False)

        best_hyperparameters.Fixed('learning_rate', value=0.001)
        #best_hyperparameters.Fixed('drop_out', value=0.38573)
        #best_hyperparameters.Fixed('regularizer', value=0.0001)


        if use_tuner == True:
            tuner = BayesianOptimization(
                self.build_model,
                objective=Objective("val_accuracy", direction="max"),
                max_trials=100,
                executions_per_trial=1,
                directory='tuner',
                project_name='versets',
                num_initial_points=10
            )
            tuner.search(train_data,
                         validation_data=test_data,
                         epochs=10,
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
                  epochs=100,
                  steps_per_epoch = epoch_size_train//BATCH_SIZE_TRAIN,
                  validation_steps = epoch_size_test//BATCH_SIZE_TEST,
                  callbacks=[epoch_callback, checkpoint, reduce_lr, verset_generator])