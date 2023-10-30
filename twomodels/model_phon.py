import os

from keras import layers, Input, Model, optimizers, regularizers
from keras.src.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras_tuner import BayesianOptimization, HyperParameters, Objective
from tensorflow.python.data import Dataset
import tensorflow as tf

from data import Data, MODEL_1_SEQ_BLACK_PATH, MODEL_1_SEQ_BLACK_TEST_PATH
from generator_callback_phon import PhonGenerator
from tensorflow.python.keras.callbacks import Callback

BATCH_SIZE_TRAIN = 250
BATCH_SIZE_TEST = 100

MODEL_1_PATH = "model/phon_model.h5"

global CURRENT_EPOCH
#CURRENT_EPOCH = 0

class EpochCallback(Callback):
    def __init__(self):
        super().__init__()

    def on_epoch_begin(self, epoch, logs=None):
        global CURRENT_EPOCH
        CURRENT_EPOCH = epoch

class PhonModel():
    def __init__(self, data: Data):
        self.data = data

    def model(self, hp):
        model = self.build_model(hp)

        return model

    def build_model(self, hp):
        lstm_units = hp.Int("phon_lstm_units", min_value=8, max_value=512, step=8, default=512)
        encoder_embedding_dim = hp.Int("phon_encoder_embedding_dim", min_value=8, max_value=512, step=8, default=512)
        decoder_embedding_dim = hp.Int("phon_decoder_embedding_dim", min_value=8, max_value=512, step=8, default=512)
        learning_rate = hp.Float('learning_rate', min_value=1e-5, max_value=1e-2, sampling='log', default=0.001)
        #drop_out_encoder = hp.Float('phon_drop_out_encoder', min_value=0.0, max_value=1.0, step=0.05, default=0.38)
        #drop_out_decoder = hp.Float('phon_drop_out_decoder', min_value=0.0, max_value=1.0, step=0.05, default=0.38)
        #regularizer = hp.Float('phon_regularizer', min_value=1e-5, max_value=1e-2, sampling='log', default=0.0001)

        attention_encoder = hp.Boolean("phon_attention_encoder", default=False)
        attention_decoder = hp.Boolean("phon_attention_decoder", default=False)
        num_heads_encoder = hp.Int("phon_num_heads_encoder", min_value=2, max_value=24, step=2, default=12, parent_name="phon_attention_encoder", parent_values=True)
        num_heads_decoder = hp.Int("phon_num_heads_decoder", min_value=2, max_value=24, step=2, default=12, parent_name="phon_attention_decoder", parent_values=True)

        # Entrées
        encoder_input = Input(shape=(self.data.title_max_size,), dtype="int32", name="title_input")
        encoder_embedding = layers.Embedding(self.data.title_words, encoder_embedding_dim)(encoder_input)
        #encoder_embedding = layers.LayerNormalization()(encoder_embedding)
        #encoder_embedding = layers.Dropout(drop_out_encoder)(encoder_embedding)
        encoder_output, forward_h, forward_c, backward_h, backward_c = layers.Bidirectional(layers.LSTM(
            lstm_units,
            return_state=True,
            return_sequences=True,
            #dropout=drop_out_encoder,
            #recurrent_dropout=drop_out_encoder,
            #kernel_regularizer=regularizers.l1_l2(regularizer),
            #recurrent_regularizer=regularizers.l1_l2(regularizer),
            #bias_regularizer=regularizers.l1_l2(regularizer)
        ))(encoder_embedding)
        state_h = layers.Concatenate()([forward_h, backward_h])
        state_c = layers.Concatenate()([forward_c, backward_c])
        encoder_states = [state_h, state_c]

        if attention_encoder:
            encoder_attention = layers.MultiHeadAttention(num_heads=num_heads_encoder, key_dim=encoder_embedding_dim//num_heads_encoder)
            encoder_attention_output = encoder_attention(query=encoder_output, key=encoder_output, value=encoder_output)
            encoder_output = layers.Concatenate(axis=-1)([encoder_output, encoder_attention_output])
            encoder_output = layers.LayerNormalization()(encoder_output)

        decoder_input = Input(shape=(self.data.phon_max_size*14+1,), dtype="int32", name="decoder_input")
        decoder_embedding = layers.Embedding(self.data.phon_words, decoder_embedding_dim)(decoder_input)
        #decoder_embedding = layers.LayerNormalization()(decoder_embedding)
        #decoder_embedding = layers.Dropout(drop_out_decoder)(decoder_embedding)
        decoder_output, _, _ = layers.LSTM(
            2 * lstm_units,
            return_sequences=True,
            return_state=True,
            #dropout=drop_out_decoder,
            #recurrent_dropout=drop_out_decoder,
            #kernel_regularizer=regularizers.l1_l2(regularizer),
            #recurrent_regularizer=regularizers.l1_l2(regularizer),
            #bias_regularizer=regularizers.l1_l2(regularizer)
        )(decoder_embedding, initial_state=encoder_states)

        if attention_decoder:
            attention = layers.MultiHeadAttention(num_heads=num_heads_decoder, key_dim=int(decoder_embedding_dim//num_heads_decoder))
            attention_output = attention(query=decoder_output, key=encoder_output, value=encoder_output)
            decoder_output = layers.Concatenate(axis=-1)([decoder_output, attention_output])
            decoder_output = layers.LayerNormalization()(decoder_output)

        '''
        
        decoder_output, _, _ = layers.LSTM(
            2 * lstm_units,
            return_sequences=True,
            return_state=True,
            #dropout=drop_out_decoder,
            #recurrent_dropout=drop_out_decoder,
            #kernel_regularizer=regularizers.l1_l2(regularizer),
            #recurrent_regularizer=regularizers.l1_l2(regularizer),
            #bias_regularizer=regularizers.l1_l2(regularizer)
        )(decoder_output)
        '''
        decoder_dense = layers.TimeDistributed(layers.Dense(self.data.phon_words+1, activation='softmax'))
        decoder_outputs = decoder_dense(decoder_output)

        # Création du modèle
        model = Model(inputs=[encoder_input, decoder_input], outputs=decoder_outputs, name="PhonModel")

        model.compile(optimizer = optimizers.legacy.Adam(learning_rate=learning_rate),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        return model

    def train(self, use_tuner=False):
        def generator_wrapper_train():
            #print("Run generator_wrapper_train - epoch : {}".format(CURRENT_EPOCH))
            for data in self.data.generate_data_phon(BATCH_SIZE_TRAIN, epoch_size_train // BATCH_SIZE_TRAIN, CURRENT_EPOCH, True):
                yield data

        def generator_wrapper_test():
            for data in self.data.generate_data_phon(BATCH_SIZE_TEST, epoch_size_test // BATCH_SIZE_TEST, CURRENT_EPOCH, False):
                yield data

        epoch_callback = EpochCallback()

        epoch_size_train = self.data.count_tfrecord_samples(MODEL_1_SEQ_BLACK_PATH)
        epoch_size_test = min(self.data.count_tfrecord_samples(MODEL_1_SEQ_BLACK_TEST_PATH), BATCH_SIZE_TEST)

        train_data = Dataset.from_generator(
            generator_wrapper_train,
            output_signature=(
                (tf.TensorSpec(shape=(None, self.data.title_max_size), dtype=tf.int32),
                 tf.TensorSpec(shape=(None, self.data.phon_max_size*14+1), dtype=tf.int32),
                 ),
                tf.TensorSpec(shape=(None, self.data.phon_max_size*14+1, self.data.phon_words+1), dtype=tf.float32)
            )).repeat()

        test_data = Dataset.from_generator(
            generator_wrapper_test,
            output_signature=(
                (tf.TensorSpec(shape=(None, self.data.title_max_size), dtype=tf.int32),
                 tf.TensorSpec(shape=(None, self.data.phon_max_size*14+1), dtype=tf.int32),
                 ),
                tf.TensorSpec(shape=(None, self.data.phon_max_size*14+1, self.data.phon_words+1), dtype=tf.float32)
            )).repeat()

        best_hyperparameters = HyperParameters()
        best_hyperparameters.Fixed('phon_lstm_units', value=336)  # 168 #128
        best_hyperparameters.Fixed('phon_encoder_embedding_dim', value=200)  # 184
        best_hyperparameters.Fixed('phon_decoder_embedding_dim', value=120)  # 184 #192
        best_hyperparameters.Fixed('phon_learning_rate', value=0.0005)  # 0.001
        #best_hyperparameters.Fixed('phon_drop_out_encoder', value=0.33)  # 0.38573
        #best_hyperparameters.Fixed('phon_drop_out_decoder', value=0.33)  # 0.38573
        #best_hyperparameters.Fixed('phon_regularizer', value=0.001)  # 0.0001
        best_hyperparameters.Fixed('num_heads_encoder', value=16)  # 10
        best_hyperparameters.Fixed('num_heads_decoder', value=16)  # 10
        best_hyperparameters.Fixed('attention_encoder', value=False)
        best_hyperparameters.Fixed('attention_decoder', value=False)


        if use_tuner == True:
            tuner = BayesianOptimization(
                self.model,
                objective=Objective("val_accuracy", direction="max"),
                max_trials=1000,
                executions_per_trial=1,
                directory='tuner',
                project_name='phons',
                num_initial_points=16,
                #hyperparameters=best_hyperparameters
            )
            tuner.search(train_data,
                         validation_data=test_data,
                         epochs=5,
                         steps_per_epoch=epoch_size_train//BATCH_SIZE_TRAIN,
                         validation_steps=epoch_size_test//BATCH_SIZE_TEST,
                         callbacks=[epoch_callback]
                         )
            best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0]

        model = self.model(best_hyperparameters)

        #if os.path.exists(MODEL_1_PATH):
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
                  epochs=1000,
                  steps_per_epoch=epoch_size_train//BATCH_SIZE_TRAIN,
                  validation_steps=epoch_size_test//BATCH_SIZE_TEST,
                  callbacks=[epoch_callback, checkpoint, reduce_lr, phon_generator])# #checkpoint reduce_lr