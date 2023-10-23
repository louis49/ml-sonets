import os

from keras import layers, Input, Model, optimizers, regularizers
from keras.src.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras_tuner import BayesianOptimization, HyperParameters, Objective
from tensorflow.python.data import Dataset
import tensorflow as tf

from data import Data, MODEL_2_SEQ_BLACK_PATH, MODEL_2_SEQ_BLACK_TEST_PATH
from phon_generator_callback import PhonGenerator
from tensorflow.python.keras.callbacks import Callback

BATCH_SIZE = 10
MODEL_2_PATH = "model/verset_model.h5"

global CURRENT_EPOCH

class EpochCallback(Callback):
    def __init__(self):
        super().__init__()
        print('EpochCallback Init : {}'.format(0))
        #self.current_epoch = 0

    def on_epoch_begin(self, epoch, logs=None):
        global CURRENT_EPOCH
        CURRENT_EPOCH = epoch

class VersetModel():
    def __init__(self, data: Data):
        self.data = data

    def build_model(self, hp):
        lstm_units = hp.Int("lstm_units", min_value=8, max_value=128, step=8, default=168)
        embedding_dim = hp.Int("embedding_dim", min_value=8, max_value=128, step=8, default=184)
        learning_rate = hp.Float('learning_rate', min_value=1e-5, max_value=1e-2, sampling='log', default=0.001)
        drop_out = hp.Float('drop_out', min_value=0.0, max_value=1.0, step=0.05, default=0.38)
        regularizer = hp.Float('regularizer', min_value=1e-5, max_value=1e-2, sampling='log', default=0.0001)
        num_heads = hp.Int("num_heads", min_value=1, max_value=16, step=1, default=12)

        # Entrées
        encoder_input = Input(shape=(self.data.title_max_size,), dtype="int32", name="title_input")
        encoder_embedding = layers.Embedding(self.data.title_words, embedding_dim)(encoder_input)
        encoder_embedding = layers.LayerNormalization()(encoder_embedding)
        encoder_embedding = layers.Dropout(drop_out)(encoder_embedding)
        encoder_output, forward_h, forward_c, backward_h, backward_c = layers.Bidirectional(layers.LSTM(
            lstm_units,
            return_state=True,
            return_sequences=True,
            # dropout=DROP_OUT,
            # recurrent_dropout=DROP_OUT,
            kernel_regularizer=regularizers.l1_l2(regularizer),
            recurrent_regularizer=regularizers.l1_l2(regularizer),
            bias_regularizer=regularizers.l1_l2(regularizer)
        ))(encoder_embedding)
        state_h = layers.Concatenate()([forward_h, backward_h])
        state_c = layers.Concatenate()([forward_c, backward_c])
        encoder_states = [state_h, state_c]

        encoder_attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embedding_dim)
        encoder_attention_output = encoder_attention(query=encoder_output, key=encoder_output, value=encoder_output)
        encoder_output = layers.Concatenate(axis=-1)([encoder_output, encoder_attention_output])


        decoder_input = Input(shape=(self.data.phon_max_size*14+1,), dtype="int32", name="decoder_input")
        decoder_embedding = layers.Embedding(self.data.phon_words, embedding_dim)(decoder_input)
        decoder_embedding = layers.LayerNormalization()(decoder_embedding)
        decoder_embedding = layers.Dropout(drop_out)(decoder_embedding)
        decoder_output, _, _ = layers.LSTM(
            2 * lstm_units,
            return_sequences=True,
            return_state=True,
            # dropout=DROP_OUT,
            # recurrent_dropout=DROP_OUT,
            kernel_regularizer=regularizers.l1_l2(regularizer),
            recurrent_regularizer=regularizers.l1_l2(regularizer),
            bias_regularizer=regularizers.l1_l2(regularizer)
        )(decoder_embedding, initial_state=encoder_states)

        attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embedding_dim)
        attention_output = attention(query=decoder_output, key=encoder_output, value=encoder_output)

        decoder_concat = layers.Concatenate(axis=-1)([decoder_output, attention_output])

        decoder_dense = layers.TimeDistributed(layers.Dense(self.data.phon_words+1, activation='softmax'))
        decoder_outputs = decoder_dense(decoder_concat)

        # Création du modèle
        model = Model(inputs=[encoder_input, decoder_input], outputs=decoder_outputs)

        # Compilation du modèle
        model.compile(optimizer=optimizers.legacy.Adam(learning_rate=learning_rate),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        return model

    def train(self, use_tuner=False):
        def generator_wrapper_train():
            #print("Run generator_wrapper_train - epoch : {}".format(CURRENT_EPOCH))
            for data in self.data.generate_data_verset(BATCH_SIZE, epoch_size_train // BATCH_SIZE, CURRENT_EPOCH, True):
                yield data

        def generator_wrapper_test():
            for data in self.data.generate_data_verset(BATCH_SIZE, epoch_size_test // BATCH_SIZE, CURRENT_EPOCH, False):
                yield data

        epoch_callback = EpochCallback()

        epoch_size_train = self.data.count_tfrecord_samples(MODEL_2_SEQ_BLACK_PATH)
        epoch_size_test = self.data.count_tfrecord_samples(MODEL_2_SEQ_BLACK_TEST_PATH)

        train_data = Dataset.from_generator(
            generator_wrapper_train,
            output_signature=(
                (tf.TensorSpec(shape=(None, self.data.title_max_size), dtype=tf.int32),
                 tf.TensorSpec(shape=(None, self.data.phon_max_size), dtype=tf.int32),
                 tf.TensorSpec(shape=(None, self.data.title_max_size-1), dtype=tf.int32),
                 ),
                tf.TensorSpec(shape=(None, self.data.title_max_size-1, self.data.title_words+1), dtype=tf.float32)
            )).repeat()

        test_data = Dataset.from_generator(
            generator_wrapper_test,
            output_signature=(
                (tf.TensorSpec(shape=(None, self.data.title_max_size), dtype=tf.int32),
                 tf.TensorSpec(shape=(None, self.data.phon_max_size), dtype=tf.int32),
                 tf.TensorSpec(shape=(None, self.data.title_max_size - 1), dtype=tf.int32),
                 ),
                tf.TensorSpec(shape=(None, self.data.title_max_size - 1, self.data.title_words + 1), dtype=tf.float32)
            )).repeat()

        best_hyperparameters = HyperParameters()
        best_hyperparameters.Fixed('lstm_units', value=32)  # 168
        best_hyperparameters.Fixed('embedding_dim', value=32)  # 184
        best_hyperparameters.Fixed('learning_rate', value=0.001)  # 0.001
        best_hyperparameters.Fixed('drop_out', value=0.38573)  # 0.38573
        best_hyperparameters.Fixed('regularizer', value=0.0001)  # 0.0001
        best_hyperparameters.Fixed('num_heads', value=4)  # 10

        if use_tuner == True:
            tuner = BayesianOptimization(
                self.build_model,
                objective=Objective("val_loss", direction="min"),
                max_trials=100,
                executions_per_trial=1,
                directory='tuner',
                project_name='versets',
                num_initial_points=10
            )
            tuner.search(train_data,
                         validation_data=test_data,
                         epochs=10,
                         steps_per_epoch=epoch_size_train//BATCH_SIZE,
                         validation_steps=epoch_size_test//BATCH_SIZE,
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
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, min_lr=0.00001)

        phon_generator = PhonGenerator(model, data=self.data)

        model.fit(train_data,
                  validation_data=test_data,
                  epochs=200,
                  steps_per_epoch = epoch_size_train//BATCH_SIZE,
                  validation_steps = epoch_size_test//BATCH_SIZE,
                  callbacks=[epoch_callback, checkpoint, reduce_lr, phon_generator])