import time
import tensorflow as tf
import keras.backend as k
import numpy as np
from tensorflow.python.data import Dataset
from keras import Input, layers, optimizers, Model, callbacks, regularizers
from keras_tuner import BayesianOptimization, HyperParameters, Objective, GridSearch
from tensorboard.plugins.hparams import api as hp
from tensorflow.python.keras.callbacks import Callback
from keras.preprocessing.sequence import pad_sequences
#{"ongoing_trials": {"tuner0": "0000"},
BATCH_SIZE_LEARN = 4000
BATCH_SIZE_TEST = 500
MODEL_PATH = "./model/model.keras"
LOGS_FOLDER_PATH = "./logs/"


class StopIfNoDescendingSlopeAndMaxAccuracy(Callback):
    def __init__(self, max_epochs=3, threshold=1.0):
        super(StopIfNoDescendingSlopeAndMaxAccuracy, self).__init__()
        self.max_epochs = max_epochs
        self.val_losses = []
        self.threshold = threshold

    def on_epoch_end(self, epoch, logs=None):
        self.val_losses.append(logs.get('val_loss'))
        if epoch + 1 >= self.max_epochs:
            slope = np.polyfit(range(self.max_epochs), self.val_losses[-self.max_epochs:], 1)[0]
            if slope >= 0:
                print(
                    f"Arrêt de l'entraînement car la pente de val_loss sur {self.max_epochs} epochs n'est pas descendante")
                self.model.stop_training = True
            else:
                print(f"Slope = {slope}")

        current_accuracy = logs.get("accuracy")
        if current_accuracy and current_accuracy >= self.threshold:
            print(f"Arrêt de l'entraînement, car l'accuracy a dépassé {self.threshold}")
            self.model.stop_training = True


class TensorBoard(callbacks.TensorBoard):
    def __init__(self, log_dir, epochs=3, **kwargs,):
        # Initialisation du constructeur de TensorBoard
        super(TensorBoard, self).__init__(log_dir=log_dir, histogram_freq=1, write_graph=False, write_images=False, write_steps_per_second=False, update_freq=1, profile_batch=0, embeddings_freq=0, embeddings_metadata=None, **kwargs)
        self.epochs = epochs
        self.val_losses = []
        self.losses = []

    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch, logs)  # Appel de la méthode parente
        self.val_losses.append(logs.get('val_loss'))
        self.losses.append(logs.get('loss'))
        diff_accuracy = logs.get('accuracy')-logs.get('val_accuracy')
        val_slope, slope = 0, 0
        if epoch + 1 >= self.epochs:
            val_slope = np.polyfit(range(self.epochs), self.val_losses[-self.epochs:], 1)[0]
            slope = np.polyfit(range(self.epochs), self.losses[-self.epochs:], 1)[0]
        with self._val_writer.as_default():
            tf.summary.scalar('epoch_slope', val_slope, step=epoch)
        with self._train_writer.as_default():
            tf.summary.scalar('epoch_slope', slope, step=epoch)
            tf.summary.scalar('epoch_diff_accuracy', diff_accuracy, step=epoch)


class PhonemeGenerator(Callback):
    def __init__(self, data, every_epoch=1):
        super().__init__()
        self.data = data
        self.every_epoch = every_epoch

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.every_epoch == 0 : #and epoch != 0
            generated_phon_learn = self.generate_text(self.data.word_learn)
            generated_phon_valid = self.generate_text(self.data.word_valid)

            print("\nLEARN : {} -> {} ({})\nVALID : {} -> {} ({})".format(
                self.data.word_learn,  generated_phon_learn, self.data.phon_learn,
                self.data.word_valid, generated_phon_valid, self.data.phon_valid,
            ))


    def generate_text(self, input):
        tokenized_word = self.data.word_tokenizer.texts_to_sequences([input])[0]
        padded_word = pad_sequences([tokenized_word], maxlen=self.data.word_max_size, padding='post')
        seed_text = '$'
        output_sequence = self.data.phons_tokenizer.texts_to_sequences([seed_text])[0]

        sequence = ''
        for i in range(self.data.phons_max_size - 1):
            current_sequence = pad_sequences([output_sequence], maxlen=self.data.phons_max_size - 1, padding='post')
            input_data = [np.array(padded_word), np.array(current_sequence)]
            predictions = self.model.predict(input_data, verbose=False)
            next_word_idx = np.argmax(predictions[0], axis=-1)
            if next_word_idx[i] == 0:  # Si c'est un padding, on continue
                continue

            output_sequence.append(next_word_idx[i])

            next_word = self.data.phons_tokenizer.index_word[next_word_idx[i]]

            if next_word == '#':
                break

            # Ajout du mot prédit au verset
            seed_text += '' + next_word
        return seed_text.strip().replace('$', '')


class Learn():

    def __init__(self, data):
        print("Init Learn")
        self.data = data

    def model(self, hyperparameters):

        def find_last_word_index(y):
            # Convertir les prédictions de one-hot à des indices de classe
            y_indices = k.argmax(y, axis=-1)

            # Trouver l'index du dernier mot non nul dans chaque séquence
            mask = k.cast(k.not_equal(y_indices, 0), k.floatx())  # Masque pour identifier les mots non nuls
            last_word_indices = k.sum(mask, axis=1) - 1  # Index du dernier mot non nul
            last_word_indices = k.switch(last_word_indices < 0, k.zeros_like(last_word_indices),
                                         last_word_indices)  # Pour traiter le cas d'aucun mot trouvé
            return last_word_indices

        def sequence(y_true, y_pred):
            y_true_indices = k.argmax(y_true, axis=-1)
            y_pred_indices = k.argmax(y_pred, axis=-1)

            # Créer un masque pour ignorer les éléments padding (zéros)
            mask = k.cast(k.not_equal(y_true_indices, 0), k.floatx())

            # Calculer la précision pour chaque élément et appliquer le masque
            correct_predictions = k.cast(k.equal(y_true_indices, y_pred_indices), k.floatx()) * mask

            # Calculer la précision moyenne sur l'ensemble de la séquence
            score = k.sum(correct_predictions) / k.sum(mask)
            return score

        def last_word(y_true, y_pred):
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

        learning_rate = hyperparameters.Float('learning_rate', min_value=1e-6, max_value=0.01, sampling='log', default=0.001)

        lstm_units = hyperparameters.Int("lstm_units", min_value=32, max_value=1024, step=32, default=128) #
        embedding_dim_encoder_input = hyperparameters.Int("embedding_dim_encoder_input", min_value=32, max_value=1024, step=32, default=128) #, default=128
        embedding_dim_decoder_input = hyperparameters.Int("embedding_dim_decoder_input", min_value=32, max_value=1024, step=32, default=128) #default=128

        lstm_layer_decoder = False #hyperparameters.Boolean("lstm_layer_decoder", default=False)
        lstm_units_decoder = hyperparameters.Int("lstm_units_decoder", min_value=32, max_value=1024, step=32, parent_values=True) #parent_name="lstm_layer_decoder",

        drop_out_encoder_input = 0. #hyperparameters.Float('drop_out_encoder_input', min_value=0, max_value=0.95, step=0.05, default=0.05)
        drop_out_decoder_input = 0. #hyperparameters.Float('drop_out_decoder_input', min_value=0, max_value=0.95, step=0.05, default=0.05)
        drop_out_decoder_output = 0. #hyperparameters.Float('drop_out_decoder_output', min_value=0, max_value=0.95, step=0.05, default=0.0)

        reg_r1_encoder = 0.0 #hyperparameters.Float('reg_r1_encoder', min_value=1e-6, max_value=0.1, sampling='log', default=0.001)
        reg_r2_encoder = 0.0 #hyperparameters.Float('reg_r2_encoder', min_value=1e-6, max_value=0.1, sampling='log', default=0.001)
        reg_r1_decoder = 0.0 #hyperparameters.Float('reg_r1_decoder', min_value=1e-6, max_value=0.1, sampling='log', default=0.001)
        reg_r2_decoder = 0.0 #hyperparameters.Float('reg_r2_decoder', min_value=1e-6, max_value=0.1, sampling='log', default=0.001)

        attention_encoder = False #hyperparameters.Boolean("attention_encoder", default=False)
        attention_decoder = False #hyperparameters.Boolean("attention_decoder", default=True)

        num_head_attention_encoder = hyperparameters.Int("num_head_attention_encoder", min_value=1, max_value=256, step=1, default=24, parent_values=True)  # ,parent_name="attention_encoder"
        num_head_attention_decoder = hyperparameters.Int("num_head_attention_decoder", min_value=1, max_value=256, step=1, default=24, parent_values=True)  # ,parent_name="attention_decoder",

        gru = False  # hyperparameters.Boolean("gru", default=False)
        gru_output = False  # hyperparameters.Boolean("gru_output", default=False)

        encoder_input = Input(shape=(self.data.word_max_size), dtype="int32", name="encoder_input")
        encoder_embedding = layers.Embedding(self.data.word_index_size, embedding_dim_encoder_input, name="encoder_embedding", mask_zero=True)(encoder_input) #
        encoder_embedding = layers.LayerNormalization(name="encoder_norm")(encoder_embedding)
        encoder_embedding = layers.Dropout(drop_out_encoder_input, name="encoder_dropout")(encoder_embedding)

        if gru:
            encoder_output1, memory_state = layers.GRU(lstm_units,
                                                      return_state=True,
                                                      return_sequences=True,
                                                      kernel_regularizer=regularizers.L1L2(reg_r1_encoder, reg_r2_encoder),
                                                      )(encoder_embedding)

            encoder_output2, memory_state = layers.GRU(lstm_units,
                                                      return_state=True,
                                                      return_sequences=True,
                                                      kernel_regularizer=regularizers.L1L2(reg_r1_encoder, reg_r2_encoder),
                                                      )(encoder_output1)

            encoder_output, memory_state = layers.GRU(lstm_units,
                                                      return_state=True,
                                                      return_sequences=True,
                                                      kernel_regularizer=regularizers.L1L2(reg_r1_encoder, reg_r2_encoder),
                                                      )(encoder_output2)

        else:
            encoder_output, memory_state, carry_state = layers.LSTM(lstm_units,
                                                                    return_state=True,
                                                                    return_sequences=True,
                                                                    kernel_regularizer=regularizers.L1L2(reg_r1_encoder, reg_r2_encoder),
                                                                    )(encoder_embedding)

            encoder_output, memory_state, carry_state = layers.LSTM(lstm_units,
                                                                    return_state=True,
                                                                    return_sequences=True,
                                                                    kernel_regularizer=regularizers.L1L2(reg_r1_encoder, reg_r2_encoder),
                                                                    )(encoder_output)
        
            encoder_output, memory_state, carry_state = layers.LSTM(lstm_units,
                                                                    return_state=True,
                                                                    return_sequences=True,
                                                                    kernel_regularizer=regularizers.L1L2(reg_r1_encoder, reg_r2_encoder),
                                                                    )(encoder_output)

        if attention_encoder:
            key_dim = max(lstm_units // num_head_attention_encoder, 1)
            attention = layers.MultiHeadAttention(num_heads=num_head_attention_encoder, key_dim=key_dim, name="encoder_attention")
            attention_output = attention(query=encoder_output, key=encoder_output, value=encoder_output)

            #attention = layers.AdditiveAttention(name="encoder_attention")
            #attention_output = attention([encoder_output, encoder_output, encoder_output])

            encoder_output = layers.Concatenate(axis=-1, name="attention_encoder_concat")([encoder_output, attention_output])

        decoder_input = Input(shape=(self.data.phons_max_size - 1), dtype="int32", name="decoder_input")
        decoder_embedding = layers.Embedding(self.data.phons_index_size, embedding_dim_decoder_input, name="decoder_embedding", mask_zero=True)(decoder_input) #
        decoder_embedding = layers.LayerNormalization(name="decoder_input_norm")(decoder_embedding)
        decoder_embedding = layers.Dropout(drop_out_decoder_input, name="decoder_input_dropout")(decoder_embedding)

        if gru:
            decoder_output = layers.GRU(lstm_units,
                                        return_state=False,
                                        return_sequences=True,
                                        kernel_regularizer=regularizers.L1L2(reg_r1_decoder, reg_r2_decoder),
                                        )(decoder_embedding, initial_state=[memory_state])
        else:
            decoder_output = layers.LSTM(lstm_units,
                                         return_state=False,
                                         return_sequences=True,
                                         kernel_regularizer=regularizers.L1L2(reg_r1_decoder, reg_r2_decoder),
                                         )(decoder_embedding, initial_state=[memory_state, carry_state])

        if attention_decoder:
            key_dim = max(lstm_units // num_head_attention_decoder, 1)
            attention = layers.MultiHeadAttention(num_heads=num_head_attention_decoder, key_dim=key_dim, name="decoder_attention")
            attention_output = attention(query=decoder_output, key=encoder_output, value=encoder_output)

            #attention = layers.Attention(name="decoder_attention")
            #attention_output = attention([decoder_output, encoder_output])

            decoder_output = layers.Concatenate(axis=-1, name="attention_decoder_concat")([decoder_output, attention_output])
            # decoder_output = layers.LayerNormalization(name="attention_norm")(decoder_output)
            # decoder_output = layers.Dropout(drop_out_decoder_output, name="attention_dropout")(decoder_output)

        if lstm_layer_decoder:
            if gru_output:
                decoder_output = layers.GRU(lstm_units_decoder, return_state=False, return_sequences=True)(decoder_output)
            else:
                decoder_output = layers.LSTM(lstm_units_decoder, return_state=False, return_sequences=True)(decoder_output)

        # decoder_output = layers.LayerNormalization(name="decoder_output_norm")(decoder_output)
        decoder_output = layers.Dropout(drop_out_decoder_output, name="decoder_output_dropout")(decoder_output)

        decoder_output = layers.TimeDistributed(
            layers.Dense(self.data.phons_index_size, activation='softmax', name="output_dense"),
            name="output_time_distributed")(decoder_output)

        model = Model(inputs=[
            encoder_input,
            decoder_input],
            outputs=decoder_output,
            name="Word2PhonemesModel")
        model.compile(optimizer="rmsprop" , #optimizers.legacy.Adam(learning_rate=learning_rate)
                      loss='categorical_crossentropy',#categorical_crossentropy
                      metrics=['accuracy', last_word, sequence],
                      )

        return model

    def train(self, tuner=False):

        def generator_wrapper_train():
            for data in self.data.generate_data(BATCH_SIZE_LEARN, train=True):
                yield data

        def generator_wrapper_test():
            for data in self.data.generate_data(BATCH_SIZE_TEST, train=False):
                yield data

        train_data = Dataset.from_generator(
            generator_wrapper_train,
            output_signature=(
                (tf.TensorSpec(shape=(None, self.data.word_max_size), dtype=tf.int32),
                 tf.TensorSpec(shape=(None, self.data.phons_max_size - 1), dtype=tf.int32),
                 ),
                tf.TensorSpec(shape=(None, self.data.phons_max_size - 1, self.data.phons_index_size), dtype=tf.float32)
            )).repeat()

        test_data = Dataset.from_generator(
            generator_wrapper_test,
            output_signature=(
                (tf.TensorSpec(shape=(None, self.data.word_max_size), dtype=tf.int32),
                 tf.TensorSpec(shape=(None, self.data.phons_max_size - 1), dtype=tf.int32),
                 ),
                tf.TensorSpec(shape=(None, self.data.phons_max_size - 1, self.data.phons_index_size), dtype=tf.float32)
            )).repeat()

        best_hyperparameters = HyperParameters()

        best_hyperparameters.Fixed('lstm_units', value=256)
        best_hyperparameters.Fixed('embedding_dim_encoder_input', value=256)
        best_hyperparameters.Fixed('embedding_dim_decoder_input', value=256)
        best_hyperparameters.Fixed('learning_rate', value=0.00001)
        best_hyperparameters.Fixed('lstm_layer_decoder', value=True)
        best_hyperparameters.Fixed('lstm_units_decoder', value=256)
        best_hyperparameters.Fixed('drop_out_encoder_input', value=0.)
        best_hyperparameters.Fixed('drop_out_decoder_input', value=0.)
        best_hyperparameters.Fixed('drop_out_decoder_output', value=0.)
        best_hyperparameters.Fixed('reg_r1_encoder', value=0.0)
        best_hyperparameters.Fixed('reg_r2_encoder', value=0.0)
        best_hyperparameters.Fixed('reg_r1_decoder', value=0.0)
        best_hyperparameters.Fixed('reg_r2_decoder', value=0.0)
        best_hyperparameters.Fixed('attention_encoder', value=True)
        best_hyperparameters.Fixed('num_head_attention_encoder', value=8)
        best_hyperparameters.Fixed('attention_decoder', value=True)
        best_hyperparameters.Fixed('num_head_attention_decoder', value=8)

        hparams_dict = {
            'lstm_units': best_hyperparameters.get('lstm_units'),
            'lstm_units_decoder': best_hyperparameters.get('lstm_units_decoder'),
            'embedding_dim_encoder_input': best_hyperparameters.get('embedding_dim_encoder_input'),
            'embedding_dim_decoder_input': best_hyperparameters.get('embedding_dim_decoder_input'),
            'learning_rate': best_hyperparameters.get('learning_rate'),
            'lstm_layer_decoder': best_hyperparameters.get('lstm_layer_decoder'),
            'drop_out_encoder_input': best_hyperparameters.get('drop_out_encoder_input'),
            'drop_out_decoder_input': best_hyperparameters.get('drop_out_decoder_input'),
            'drop_out_decoder_output': best_hyperparameters.get('drop_out_decoder_output'),
            'reg_r1_encoder': best_hyperparameters.get('reg_r1_encoder'),
            'reg_r2_encoder': best_hyperparameters.get('reg_r2_encoder'),
            'reg_r1_decoder': best_hyperparameters.get('reg_r1_decoder'),
            'reg_r2_decoder': best_hyperparameters.get('reg_r2_decoder'),
            'attention_encoder': best_hyperparameters.get('attention_encoder'),
            'num_head_attention_encoder': best_hyperparameters.get('num_head_attention_encoder'),
            'attention_decoder': best_hyperparameters.get('attention_decoder'),
            'num_head_attention_decoder': best_hyperparameters.get('num_head_attention_decoder'),
        }

        generator_callback = PhonemeGenerator(data=self.data, every_epoch=1)

        stop_callback = StopIfNoDescendingSlopeAndMaxAccuracy(max_epochs=5, threshold=1.0)

        epoch_size_train = self.data.records_learn // 10
        epoch_size_test = self.data.records_test

        if tuner:
            log_dir = LOGS_FOLDER_PATH + 'tuner'
            tensorboard_callback = TensorBoard(log_dir=log_dir,  epochs=5)  # callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=False) #
            #tensorboard_callback = callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=False) #
            # slope_callback = Slope(log_dir, epochs=10)
            '''
            tuner = GridSearch(
                self.model,
                directory='tuner',
                project_name='phonemes',
                executions_per_trial=1,
                max_trials=1000,
                objective=Objective("val_accuracy", direction="max")
            )
            '''

            tuner = BayesianOptimization(
                self.model,
                objective=  # [
                # Objective("val_sequence", direction="max"),
                Objective("val_accuracy", direction="max"),
                # ],
                max_trials=1000,
                executions_per_trial=1,
                directory='tuner',
                project_name='phonemes',
                num_initial_points=20,
                # overwrite=True,
            )

            tuner.search(train_data,
                         validation_data=test_data,
                         epochs=2,
                         steps_per_epoch=epoch_size_train // BATCH_SIZE_LEARN,
                         validation_steps=epoch_size_test // BATCH_SIZE_TEST,
                         callbacks=[tensorboard_callback, generator_callback, ],
                         # stop_callback # tensorboard_callback #accuracy_threshold_callback
                         verbose=1,
                         )
        else:
            log_dir = LOGS_FOLDER_PATH + '/runs/' + str(int(time.time()))

        with tf.summary.create_file_writer(log_dir).as_default():
            hp.hparams(hparams_dict)

        checkpoint_callback = callbacks.ModelCheckpoint(MODEL_PATH, save_freq='epoch', save_weights_only=False)

        model = self.model(best_hyperparameters)

        tensorboard_callback = TensorBoard(log_dir, epochs=5)
        model.fit(train_data,
                  validation_data=test_data,
                  epochs=1000,
                  steps_per_epoch=epoch_size_train // BATCH_SIZE_LEARN,
                  validation_steps=epoch_size_test // BATCH_SIZE_TEST,
                  callbacks=[
                      tensorboard_callback,
                      checkpoint_callback,
                      generator_callback,
                  ])


