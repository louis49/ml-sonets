import os

from keras.src.callbacks import Callback, ModelCheckpoint, ReduceLROnPlateau
from keras_tuner import HyperParameters, BayesianOptimization, Objective
from tensorflow.python.data import Dataset

from data import Data, MODEL_3_SEQ_BLACK_PATH, MODEL_3_SEQ_BLACK_TEST_PATH
from keras import layers, Input, Model, optimizers, regularizers
from model_verset import VersetModel, MODEL_2_PATH
from model_phon import PhonModel, MODEL_1_PATH
import tensorflow as tf
from generator_callback_sonnet import SonnetGenerator

BATCH_SIZE_TRAIN = 100
BATCH_SIZE_TEST = 10

MODEL_3_PATH = "model/sonnet_model.h5"

global CURRENT_EPOCH
CURRENT_EPOCH = 0
class EpochCallback(Callback):
    def __init__(self):
        super().__init__()

    def on_epoch_begin(self, epoch, logs=None):
        global CURRENT_EPOCH
        CURRENT_EPOCH = epoch

class SonnetModel():
    def __init__(self, data: Data):
        self.data = data

    def build_model(self, hp):
        model = self.model(hp)
        #learning_rate = hp.Float('learning_rate', min_value=1e-5, max_value=1e-2, sampling='log', default=0.001)
        #model.compile(optimizer=optimizers.legacy.Adam(learning_rate=learning_rate),
        #              loss='categorical_crossentropy',
        #              metrics=['accuracy'])
        return model

    def model(self, hp):

        learning_rate = hp.Float('learning_rate', min_value=1e-5, max_value=1e-2, sampling='log', default=0.001)

        # Créer une instance de chaque modèle.
        phon_modeler = PhonModel(data=self.data)
        phon_model = phon_modeler.build_model(hp)

        #phon_model.load_weights(MODEL_1_PATH)

        verset_modeler = VersetModel(data=self.data)
        verset_model = verset_modeler.model(hp)

        #verset_model.load_weights(MODEL_2_PATH)

        # Définir les entrées.
        input_title = Input(shape=(self.data.title_max_size,), dtype="int32", name="title_input")
        input_phons = Input(shape=(self.data.phon_max_size * 14 + 1, self.data.phon_words + 1), dtype="float32", name="phons_input")  # Les 14 rimes.
        input_texts = Input(shape=(self.data.text_max_size * 14 - 14,), dtype="int32", name="verses_input")  # Les 14 versets potentiels.

        # Traiter les entrées.
        processed_phons = input_phons#[:,:-1] // Le dernier élément a été enlevé lors du generate_data
        processed_texts = input_texts#[:, 1:] // On garde les 14 versets qui contiennent les tags

        processed_phons_max = tf.argmax(processed_phons, axis=-1)

        # Utiliser le Modèle 1 pour obtenir les rimes encodées.
        output_phons = phon_model([input_title, processed_phons_max]) #processed_phons_max contient le $ (start), encoded_phons, contient le € (end)

        # Diviser les rimes encodées pour les utiliser individuellement.
        encoded_phons_split = tf.split(output_phons[:, :-1,:], num_or_size_splits=14, axis=1) # On supprime le dernier élémentqui est le tag de fin € et on découpe en 14 morceaux
        encoded_verses_split = tf.split(processed_texts, num_or_size_splits=14, axis=1) # On découpe en 14 morceaux

        # Utiliser le Modèle 2 pour générer des versets à partir de chaque rime.
        generated_verses = []
        for i in range(14):
            line_number = tf.constant([[i]], dtype=tf.int32)
            input_phon = encoded_phons_split[i]
            input_versets = encoded_verses_split[i]
            generated_verse = verset_model([input_title, input_phon, line_number, input_versets])
            generated_verses.append(generated_verse)

        output_texts = layers.Concatenate(axis=1, name="VersetsModel")(generated_verses)

        model = Model(inputs=[input_title, input_phons, input_texts], outputs=[output_phons, output_texts], name="SonnetModel")

        model.compile(optimizer=optimizers.legacy.Adam(learning_rate=learning_rate),
                      loss={
                          'PhonModel': 'categorical_crossentropy',
                          'VersetsModel': 'categorical_crossentropy',
                      },
                      metrics={
                          'PhonModel': ['accuracy'],
                          'VersetsModel': ['accuracy'],
                      },
                      loss_weights={
                          'PhonModel': 1.0,
                          'VersetsModel': 1.0,
                      })

        return model

    def train(self, use_tuner=False):

        def generator_wrapper_train():
            #print("Run generator_wrapper_train - epoch : {}".format(CURRENT_EPOCH))
            for data in self.data.generate_data_sonnet(BATCH_SIZE_TRAIN, epoch_size_train // BATCH_SIZE_TRAIN, CURRENT_EPOCH, True):
                yield data

        def generator_wrapper_test():
            for data in self.data.generate_data_sonnet(BATCH_SIZE_TEST, epoch_size_test // BATCH_SIZE_TEST, CURRENT_EPOCH, False):
                yield data

        epoch_callback = EpochCallback()

        epoch_size_train = self.data.count_tfrecord_samples(MODEL_3_SEQ_BLACK_PATH)
        epoch_size_test = min(self.data.count_tfrecord_samples(MODEL_3_SEQ_BLACK_TEST_PATH), BATCH_SIZE_TEST)

        # k = self.data.generate_data_sonnet(BATCH_SIZE_TEST, epoch_size_test // BATCH_SIZE_TEST, CURRENT_EPOCH, False)

        train_data = Dataset.from_generator(
            generator_wrapper_train,
            output_signature=(
                (tf.TensorSpec(shape=(None, self.data.title_max_size), dtype=tf.int32),
                 tf.TensorSpec(shape=(None, self.data.phon_max_size * 14 + 1, self.data.phon_words + 1), dtype=tf.float32),
                 tf.TensorSpec(shape=(None, self.data.text_max_size * 14 - 14), dtype=tf.int32),
                 ),
                (tf.TensorSpec(shape=(None, self.data.phon_max_size * 14 + 1, self.data.phon_words + 1), dtype=tf.float32),
                 tf.TensorSpec(shape=(None, self.data.text_max_size * 14 - 14, self.data.text_words + 1), dtype=tf.float32)
                 )
            )).repeat()

        test_data = Dataset.from_generator(
            generator_wrapper_test,
            output_signature=(
                (tf.TensorSpec(shape=(None, self.data.title_max_size), dtype=tf.int32),
                 tf.TensorSpec(shape=(None, self.data.phon_max_size*14+1, self.data.phon_words + 1), dtype=tf.float32),
                 tf.TensorSpec(shape=(None, self.data.text_max_size * 14 - 14), dtype=tf.int32),
                 ),
                (tf.TensorSpec(shape=(None, self.data.phon_max_size*14+1, self.data.phon_words + 1), dtype=tf.float32),
                 tf.TensorSpec(shape=(None, self.data.text_max_size * 14 - 14, self.data.text_words + 1), dtype=tf.float32)
                 )
            )).repeat()

        best_hyperparameters = HyperParameters()

        best_hyperparameters.Fixed('verset_lstm_units', value=1)
        best_hyperparameters.Fixed('verset_encoder_title_embedding_dim', value=1)
        best_hyperparameters.Fixed('verset_decoder_embedding_dim', value=1)

        best_hyperparameters.Fixed('phon_lstm_units', value=1)
        best_hyperparameters.Fixed('phon_encoder_embedding_dim', value=1)
        best_hyperparameters.Fixed('phon_decoder_embedding_dim', value=1)

        best_hyperparameters.Fixed('learning_rate', value=0.001)

        #best_hyperparameters.Fixed('drop_out', value=0.38573)
        #best_hyperparameters.Fixed('regularizer', value=0.0001)
        #best_hyperparameters.Fixed('num_heads', value=5)

        if use_tuner == True:
            tuner = BayesianOptimization(
                self.build_model,
                objective=[Objective("val_VersetsModel_accuracy", direction="max"),
                           Objective("val_PhonModel_accuracy", direction="max")],
                max_trials=100,
                executions_per_trial=1,
                directory='tuner',
                project_name='sonnets',
                num_initial_points=10
            )
            tuner.search(train_data,
                         validation_data=test_data,
                         epochs=5,
                         steps_per_epoch=epoch_size_train // BATCH_SIZE_TRAIN,
                         validation_steps=epoch_size_test // BATCH_SIZE_TEST,
                         callbacks=[epoch_callback]
                         )
            best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0]

        model = self.build_model(best_hyperparameters)

        save_options = tf.saved_model.SaveOptions()
        checkpoint = ModelCheckpoint(MODEL_3_PATH,
                                     save_freq='epoch',
                                     monitor='val_accuracy',
                                     save_best_only=True,
                                     mode="max",
                                     save_weights_only=False,
                                     # options=save_options,
                                     )
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.90, patience=10, min_lr=0.000001)

        sonnet_generator = SonnetGenerator(model, data=self.data)

        if os.path.exists(MODEL_3_PATH):
            model.load_weights(MODEL_3_PATH)
            sonnet_generator.generate_text("Amour fou")

        model.fit(train_data,
                  validation_data=test_data,
                  epochs=200,
                  steps_per_epoch=epoch_size_train // BATCH_SIZE_TRAIN,
                  validation_steps=epoch_size_test // BATCH_SIZE_TEST,
                  callbacks=[epoch_callback, checkpoint, reduce_lr, sonnet_generator])
